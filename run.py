import os
import sys
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm

import pickle

import jax
from jax.experimental.checkify import checkify, index_checks
from jax import numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt

import torch
import einops

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from torch.utils.tensorboard import SummaryWriter 
import numpy as np
from functools import partial

from score_sde.models.flow import SDEPushForward
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref

from context_generator.utils.misc import load_config, inf_iterator
from context_generator.datasets.pdbredo_chain import get_pdbredo_chain_dataset
from context_generator.utils.data import PaddingCollate
from context_generator.utils.protein.constants import chi_pi_periodic, AA
from context_generator.models.da_encoder import RDEEncoder
from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder

log = logging.getLogger(__name__)
writer = SummaryWriter('train')
context_gen_cfg = f"./context_generator/configs/train/diff.yml"

# jax.config.update('jax_array', True)

def aggregate_sidechain_accuracy(aa, chi_pred, chi_native, chi_mask):
    aa = aa.reshape(-1)
    chi_mask = chi_mask.reshape(-1, 4)
    diff = torch.min(
        (chi_pred - chi_native) % (2 * np.pi),
        (chi_native - chi_pred) % (2 * np.pi),
    )   # (N, L, 4)
    diff = torch.rad2deg(diff)
    diff = diff.reshape(-1, 4)

    diff_flip = torch.min(
        ( (chi_pred + np.pi) - chi_native) % (2 * np.pi),
        (chi_native - (chi_pred + np.pi) ) % (2 * np.pi),
    )
    diff_flip = torch.rad2deg(diff_flip)
    diff_flip = diff_flip.reshape(-1, 4)
    
    acc = [{j:[] for j in range(1, 4+1)} for i in range(20)]
    for i in range(aa.size(0)):
        for j in range(4):
            chi_number = j+1
            if not chi_mask[i, j].item(): continue
            if chi_pi_periodic[AA(aa[i].item())][chi_number-1]:
                diff_this = min(diff[i, j].item(), diff_flip[i, j].item())
            else:
                diff_this = diff[i, j].item()
            acc[aa[i].item()][chi_number].append(diff_this)
    
    table = np.full((20, 4), np.nan)
    for i in range(20):
        for j in range(1, 4+1):
            if len(acc[i][j]) > 0:
                table[i, j-1] = np.mean(acc[i][j])
    return table

def make_sidechain_accuracy_table_image(tag: str, diff: np.ndarray):
    from Bio.PDB.Polypeptide import index_to_three
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    rows = [index_to_three(i) for i in range(20)]
    cell_text = diff.tolist()
    fig, ax = plt.subplots(dpi=200)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(tag)
    ax.table(
        cellText=cell_text,
        colLabels=columns,
        rowLabels=rows,
        loc='center'
    )
    return fig

def run(cfg):
    def train(train_state):
        loss = instantiate(
            cfg.loss, pushforward=pushforward, model=model, eps=cfg.eps, train=True
        )
        train_step_fn = get_ema_loss_step_fn(loss, optimizer=optimiser, train=True)
        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        t = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()
        total_train_time = 0
        for step in t:
            # data, context = next(train_ds)
            batch = next(train_it)
            data = batch.pop('chi_native')
            for k, v in list(batch.items()):
                if isinstance(v, list) or isinstance(v, int):
                    _ = batch.pop(k)
                    # print(v_del)
            
            data = einops.rearrange(jnp.stack([jnp.sin(data), jnp.cos(data)], axis=-1), 'b l m n -> b l (m n)')
            xbatch = {"data": data, "context": batch}
            
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), xbatch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan")
                return train_state, False

            if step % 50 == 0:
                logger.log_metrics({"train/loss": loss}, step)
                t.set_description(f"Loss: {loss:.3f}")

                writer.add_scalar("train/loss", np.array(loss), step)

            if step > 0 and step % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time
                ckpt_path_by_step = os.path.join(ckpt_path, f'step_{step}')
                os.makedirs(ckpt_path_by_step, exist_ok=True)
                save(ckpt_path_by_step, train_state)
                save(ckpt_path, train_state)

                if cfg.train_plot:
                    generate_plots(train_state, "val", step=step)
                train_time = timer()

        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, True

    def generate_plots(train_state, stage, step=None):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler_kwargs = dict(N=100, eps=cfg.eps, predictor="GRW")
        sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)
        
        chi_pred, chi_native, chi_masked_flag, chi_corrupt_flag, aa_all = [], [], [], [], []

        for i, vbatch in enumerate(tqdm(val_dl, desc='Validate', dynamic_ncols=True)):
            data = vbatch.pop('chi_native')
            chi_native.append(data)

            for k, v in list(vbatch.items()):
                if isinstance(v, list) or isinstance(v, int):
                    _ = vbatch.pop(k)    

            chi_masked_flag.append(
                vbatch['chi_masked_flag'][..., None] * vbatch['chi_mask']
            )              
            chi_corrupt_flag.append(
                vbatch['chi_corrupt_flag'][..., None] * vbatch['chi_mask']
            )        
            aa_all.append(vbatch['aa'])    

            shape = (int(data.shape[0]*data.shape[1]),)
            rng, next_rng = jax.random.split(rng)
            #HACK
            xs = sampler(next_rng, shape, vbatch)
            prop_in_M = data_manifold.belongs(xs, atol=1e-4).mean()
            log.info(f"Prop samples in M = {100 * prop_in_M.item():.1f}%") 
            xs = jnp.stack(
                [jnp.arctan2(xs[..., 0], xs[..., 1]), jnp.arctan2(xs[..., 2], xs[..., 3]), jnp.arctan2(xs[..., 4], xs[..., 5]), jnp.arctan2(xs[..., 6], xs[..., 7])],
                axis=-1,
            )
            chi_pred.append(np.asarray(xs).reshape(data.shape[0], data.shape[1], -1))
        
        chi_pred, chi_native = np.concatenate(chi_pred, axis=0), np.concatenate(chi_native, axis=0)
        chi_masked_flag = np.concatenate(chi_masked_flag, axis=0)
        chi_corrupt_flag = np.concatenate(chi_corrupt_flag, axis=0)
        aa_all = np.concatenate(aa_all, axis=0)
        chi_pred, chi_native = torch.tensor(chi_pred), torch.tensor(chi_native)
        chi_masked_flag = torch.tensor(chi_masked_flag)
        chi_corrupt_flag = torch.tensor(chi_corrupt_flag)
        aa_all = torch.tensor(aa_all)
        log.info(f"Data prepared")
        acc_table_masked = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_masked_flag)
        acc_table_corrupt = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_corrupt_flag)
        print(acc_table_masked)
        print(acc_table_corrupt)
        writer.add_figure(
            'val/acc_table_masked', 
            make_sidechain_accuracy_table_image('masked', acc_table_masked), 
            global_step=step
        )
        writer.add_figure(
            'val/acc_table_corrupt',
            make_sidechain_accuracy_table_image('corrupt', acc_table_corrupt),
            global_step=step
        )
    
    
    def generate_samples(train_state, stage, step=None):
        log.info("Generating samples")
        rng = jax.random.PRNGKey(cfg.seed)

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler_kwargs = dict(N=100, eps=cfg.eps, predictor="GRW")
        sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)

        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        likelihood_fn = jax.jit(likelihood_fn)


        chi_pred, chi_native, chi_masked_flag, chi_corrupt_flag, aa_all = [], [], [], [], []
        for i, vbatch in enumerate(tqdm(val_dl, desc='Validate', dynamic_ncols=True)):
            output_dir = os.path.join("sample_results", vbatch['id'][0])
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'data.pickle'), 'wb') as f:
                pickle.dump(vbatch, f)
            data = vbatch.pop('chi_native')
            chi_native.append(data)

            for k, v in list(vbatch.items()):
                if isinstance(v, list) or isinstance(v, int):
                    _ = vbatch.pop(k)
            
            chi_masked_flag.append(
                vbatch['chi_masked_flag'][..., None] * vbatch['chi_mask']
            )      

            with open(os.path.join(output_dir, 'masked_flag.pickle'), 'wb') as f:
                pickle.dump(vbatch['chi_masked_flag'][..., None] * vbatch['chi_mask'], f)        
            chi_corrupt_flag.append(
                vbatch['chi_corrupt_flag'][..., None] * vbatch['chi_mask']
            )
            aa_all.append(vbatch['aa'])

            shape = (int(data.shape[0]*data.shape[1]),)

            xs_all = []
            logp_step_all = []

            for _ in range(5):
                rng, next_rng = jax.random.split(rng)
                xs = sampler(next_rng, shape, vbatch)
                prop_in_M = data_manifold.belongs(xs, atol=1e-4).mean()
                log.info(f"Prop samples in M = {100 * prop_in_M.item():.1f}%")
                logp_step, nfe = likelihood_fn(xs.reshape(data.shape[0], data.shape[1], -1), vbatch)
                xs_all.append(xs)
                logp_step_all.append(logp_step)
            
            xs_all = np.stack(xs_all, axis=0)
            xs_all = np.stack(
                [np.arctan2(xs_all[..., 0], xs_all[..., 1]), np.arctan2(xs_all[..., 2], xs_all[..., 3]), np.arctan2(xs_all[..., 4], xs_all[..., 5]), np.arctan2(xs_all[..., 6], xs_all[..., 7])],
                axis=-1,
            )
            xs_all = xs_all.reshape(xs_all.shape[0], data.shape[0], data.shape[1], -1)

            logp_step_all = np.stack(logp_step_all, axis=0)
            logp_step_all = logp_step_all.reshape(logp_step_all.shape[0], data.shape[0], data.shape[1], )

            xs = torch.tensor(xs_all)
            logprobs = torch.tensor(logp_step_all)

            logprobs_max, smp_idx = logprobs.max(dim=0)    # (N, L)
            smp_idx = smp_idx[None, :, :, None].repeat(1, 1, 1, 4)  # (1, N, L, 4)
            xs = torch.gather(xs, dim=0, index=smp_idx).squeeze(0)

            with open(os.path.join(output_dir, 'xs.pickle'), 'wb') as f:
                pickle.dump(xs.numpy(), f)
            chi_pred.append(xs.numpy())
        
        
        chi_pred, chi_native = np.concatenate(chi_pred, axis=0), np.concatenate(chi_native, axis=0)

        chi_masked_flag = np.concatenate(chi_masked_flag, axis=0)
        chi_corrupt_flag = np.concatenate(chi_corrupt_flag, axis=0)
        aa_all = np.concatenate(aa_all, axis=0)
        chi_pred, chi_native = torch.tensor(chi_pred), torch.tensor(chi_native)
        chi_masked_flag = torch.tensor(chi_masked_flag)
        chi_corrupt_flag = torch.tensor(chi_corrupt_flag)
        aa_all = torch.tensor(aa_all)
        log.info(f"Data prepared")
        acc_table_masked = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_masked_flag)
        acc_table_corrupt = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_corrupt_flag)
        print(acc_table_masked)
        print(acc_table_corrupt)
        writer.add_figure(
            'val/acc_table_masked', 
            make_sidechain_accuracy_table_image('masked', acc_table_masked), 
            global_step=step
        )
        writer.add_figure(
            'val/acc_table_corrupt',
            make_sidechain_accuracy_table_image('corrupt', acc_table_corrupt),
            global_step=step
        )
    ### Main
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    if cfg.mode == "test":
        ckpt_path = "./tmp/checkpoint/SidechainDiff_ckpt"
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info("Stage : Instantiate model")
    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    log.info("Stage : Instantiate dataset")
    rng, next_rng = jax.random.split(rng)

    cg_config, _ = load_config(context_gen_cfg)
    train_ds = get_pdbredo_chain_dataset(cg_config.data.train)
    val_ds = get_pdbredo_chain_dataset(cg_config.data.val)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=PaddingCollate())
    train_it = inf_iterator(train_dl)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=PaddingCollate())
    log.info('Train %d | Val %d' % (len(train_ds), len(val_ds)))

    log.info("Stage : Instantiate vector field model")

    # @partial(checkify, errors=index_checks)
    def model(y, t, context):
        """Vector field s_\theta: y, t, context -> T_y M"""
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        encoder = RDEEncoder(
            single_encoder=PerResidueEncoder(
                feat_dim=cg_config.model.encoder.node_feat_dim,
                max_num_atoms=5
            ),
            masked_bias=hk.Embed(
                vocab_size=2,
                embed_dim=cg_config.model.encoder.node_feat_dim,
            ),
            pair_encoder=ResiduePairEncoder(
                feat_dim=cg_config.model.encoder.pair_feat_dim,
                max_num_atoms=5,    # N, CA, C, O, CB,
            ),
            attn_encoder=GAEncoder(
                **cg_config.model.encoder
            )
        )

        enc_context = encoder(context)
        enc_context = einops.rearrange(enc_context, 'b l d -> (b l) d')
        t_expanded = jnp.expand_dims(t.reshape(-1), -1)
        
        enc_context = jnp.concatenate([t_expanded, enc_context], axis=-1)
        if len(y.shape) == 3:
            y = einops.rearrange(y, 'b l d -> (b l) d')
        
        return score(y, enc_context)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    ibatch = next(train_it)
    data = ibatch.pop('chi_native')
    for k, v in list(ibatch.items()):
        if isinstance(v, list) or isinstance(v, int):
            _ = ibatch.pop(k)
    data = einops.rearrange(jnp.stack([jnp.sin(data), jnp.cos(data)], axis=-1), 'b l m n -> b l (m n)')

    t = jnp.zeros((data.shape[0], data.shape[1], 1))
    params, state = model.init(rng=next_rng, y=transform.inv(data), t=t, context=ibatch)

    log.info("Stage : Instantiate optimiser")
    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))


    opt_state = optimiser.init(params)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        train_state = restore(ckpt_path)
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_state=opt_state,
            model_state=state,
            step=0,
            params=params,
            ema_rate=cfg.ema_rate,
            params_ema=params,
            rng=next_rng,  # TODO: we should actually use this for reproducibility
        )
        save(ckpt_path, train_state)


    if cfg.mode == "train" or cfg.mode == "all":
        if train_state.step == 0 and cfg.test_plot:
            generate_plots(train_state, "test", step=-1)
        log.info("Stage : Training")
        train_state, success = train(train_state)
    if cfg.mode == "test" or (cfg.mode == "all" and success):
        if cfg.test_plot:
            generate_samples(train_state, "test", step=-1)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")
