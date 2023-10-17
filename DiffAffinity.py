import os
import sys
import shutil
import argparse
import pandas as pd
import torch.utils.tensorboard
from tqdm.auto import tqdm
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from score_sde.utils import TrainState, save, restore

from context_generator.models.da_encoder import RDEEncoder
from context_generator.utils.misc import BlackHole, load_config, get_logger, get_new_log_dir, current_milli_time
from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder
from context_generator.models.da_ddg import DDG_RDE_Network
from context_generator.utils.protein.constants import BBHeavyAtom
from context_generator.utils.skempi import SkempiDatasetManager, per_complex_corr
from context_generator.utils.train import log_losses, ScalarMetricAccumulator

rng = jax.random.PRNGKey(42)

class Freezable_TrainState(NamedTuple):
    trainable_params: hk.Params
    non_trainable_params: hk.Params
    opt_state: optax.OptState
    # state: hk.State



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--num_cvfolds', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='./logs_skempi')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--idx_cvfolds', type=int, required=True)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    rde_config, rde_config_name = load_config("context_generator/configs/train/diff.yml")


    class CrossValidation(object):

        def __init__(self, params, state, num_cvfolds):
            super().__init__()
            self.num_cvfolds = num_cvfolds
            self.params = params
            self.state = state
            self.models = []
            self.optimisers = []
            for i in range(num_cvfolds):
                optimiser = optax.chain(optax.adam(config.train.optimizer.lr), optax.clip_by_global_norm(1.0))
                self.optimisers.append(optimiser)
                trainable_params, non_trainable_params = hk.data_structures.partition(
                    lambda m,n,p: "trainable" in m, self.params)
                opt_state = optimiser.init(self.params)

                self.models.append(
                    Freezable_TrainState(
                        trainable_params=trainable_params,
                        non_trainable_params=non_trainable_params,
                        opt_state=opt_state,
                        state=self.state
                    )
                )

        def get(self, fold):
            return self.models[fold], self.optimisers[fold]

        def set(self, fold, new_train_state:NamedTuple):
            self.models[fold] = new_train_state

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None

    else:
        if args.resume:
            log_dir = get_new_log_dir(args.logdir, prefix='%s(%d)-resume' % (config_name, args.num_cvfolds,), tag=args.tag)
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s(%d)' % (config_name, args.num_cvfolds,), tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SkempiDatasetManager(
        config, 
        num_cvfolds=args.num_cvfolds,
        num_workers=args.num_workers,
        logger=logger,
    )
    # Resume Encoder

    # Create the model

    def _forward(batch: dict):

        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        ddg_rde_wt = DDG_RDE_Network(
            context_encoder=RDEEncoder(
                single_encoder=PerResidueEncoder(
                    feat_dim=rde_config.model.encoder.node_feat_dim,
                    max_num_atoms=5
                ),
                masked_bias=hk.Embed(
                    vocab_size=2,
                    embed_dim=rde_config.model.encoder.node_feat_dim,
                ),
                pair_encoder=ResiduePairEncoder(
                    feat_dim=rde_config.model.encoder.pair_feat_dim,
                    max_num_atoms=5
                ),
                attn_encoder=GAEncoder(
                    **rde_config.model.encoder
                )              
            ),
            cfg=config,
            single_encoder_ddg=PerResidueEncoder(
                feat_dim=config.model.encoder.node_feat_dim,
                max_num_atoms=5,
                name="trainable_per_residue_encoder"
            ),
            pair_encoder_ddg=ResiduePairEncoder(
                feat_dim=config.model.encoder.pair_feat_dim,
                max_num_atoms=5,
                name="trainable_residue_pair_encoder"
            ),
            attn_encoder_ddg=GAEncoder(
                **config.model.encoder,
                name="trainable_ga_encoder"
            )
        )
        ddg_rde_mt = ddg_rde_wt

        h_wt = ddg_rde_wt(batch_wt)
        h_mt = ddg_rde_mt(batch_mt)

        H_mt, H_wt = h_mt.max(axis=1), h_wt.max(axis=1)
        dim = config.model.encoder.node_feat_dim
        ddg_readout = hk.Sequential([
            hk.Linear(dim, name="trainable_linear"), jax.nn.relu,
            hk.Linear(dim, name="trainable_linear_1"), jax.nn.relu,
            hk.Linear(1, name="trainable_linear_2")
        ])
        ddg_pred = ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = ddg_readout(H_wt - H_mt).squeeze(-1)
        return ddg_pred, ddg_pred_inv

    model = hk.transform(_forward)

    rng, next_rng = jax.random.split(rng)
    ibatch = next(dataset_mgr.get_train_iterator(0))
    for k, v in list(ibatch.items()):
        if isinstance(v, list) or isinstance(v, int):
            _ = ibatch.pop(k)  
    params = model.init(rng=next_rng, batch=ibatch)
    encoder_train_state = restore(config.model.SidechainDiff_checkpoint)
    encoder_params = encoder_train_state.params
    for k in list(encoder_params.keys()):
        if k.startswith('torus_generator'):
            _ = encoder_params.pop(k)
    
    params = hk.data_structures.merge(params, encoder_params)
    params = jax.device_put(params)
    for k in list(encoder_params.keys()):
        assert k in list(params.keys())

    trainable_params, non_trainable_params = hk.data_structures.partition(
                        lambda m,n,p: "trainable" in m or "ddg_rde__network" in m, params)

    optimiser = optax.chain(optax.adam(config.train.optimizer.lr), optax.clip_by_global_norm(1.0))

    opt_state = optimiser.init(trainable_params)

    train_state = Freezable_TrainState(
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
        opt_state=opt_state
    )

    print(len(list(train_state.non_trainable_params.keys())))

    def loss_fn(trainable_params, non_trainable_params, batch):
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        ddg_pred, ddg_pred_inv = model.apply(params, None, batch)
        loss = (jnp.mean(optax.l2_loss(ddg_pred, batch['ddG'])) + jnp.mean(optax.l2_loss(ddg_pred_inv, -batch['ddG']))) / 2
        return loss
    
    @jax.jit
    def train_step(train_state: Freezable_TrainState, batch_dict):
        trainable_params, non_trainable_params, opt_state = train_state
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss, trainable_params_grads = loss_and_grad_fn(
            trainable_params,
            non_trainable_params,
            batch_dict)
        updates, new_opt_state = optimiser.update(trainable_params_grads, opt_state)
        new_trainable_params = optax.apply_updates(trainable_params, updates)

        train_state = Freezable_TrainState(
            trainable_params=new_trainable_params,
            non_trainable_params=non_trainable_params,
            opt_state=new_opt_state,
        )
        return train_state, loss

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(args.idx_cvfolds), desc='Validate', dynamic_ncols=True)):
            trainable_params, non_trainable_params, opt_state = train_state
            params = hk.data_structures.merge(trainable_params, non_trainable_params)
            ddg_pred, ddg_pred_inv = model.apply(params, None, batch)
            loss = (jnp.mean(optax.l2_loss(ddg_pred, batch['ddG'])) + jnp.mean(optax.l2_loss(ddg_pred_inv, -batch['ddG']))) / 2

            loss = torch.tensor(np.array(loss))
            scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

            for complex, mutstr, ddg_true, ddg_pred in zip(batch['complex'], batch['mutstr'], batch['ddG'], np.array(ddg_pred)):
                results.append({
                    'complex': complex,
                    'mutstr': mutstr,
                    'num_muts': len(mutstr.split(',')),
                    'ddG': ddg_true,
                    'ddG_pred': ddg_pred
                })
        results = pd.DataFrame(results)
        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{it}.csv'), index=False)
        pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
        spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
        pearson_pc, spearman_pc = per_complex_corr(results)

        logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
        logger.info(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
        writer.add_scalar('val/all_pearson', pearson_all, it)
        writer.add_scalar('val/all_spearman', spearman_all, it)
        writer.add_scalar('val/pc_pearson', pearson_pc, it)
        writer.add_scalar('val/pc_spearman', spearman_pc, it)

        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', logger=logger, writer=writer)
        return avg_loss
    
    it_first = 1
    
    for it in range(it_first, config.train.max_iters + 1):
        
        if it % config.train.val_freq == 0:
            avg_val_loss = validate(it)
            if not args.debug:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                os.makedirs(ckpt_path, exist_ok=True)
                save(ckpt_path, train_state)

        fold = it % args.num_cvfolds
        if fold != args.idx_cvfolds:
            continue
        time_start = current_milli_time()
        xbatch = next(dataset_mgr.get_train_iterator(fold))
        for k, v in list(xbatch.items()):
            if isinstance(v, list) or isinstance(v, int):
                _ = xbatch.pop(k)
        
        train_state, loss = train_step(train_state, xbatch)

        time_backward_end = current_milli_time()
        # Logging
        scalar_dict = {}
        scalar_dict.update({
            'fold': fold,
            'time': (time_backward_end - time_start) / 1000,
        })
        loss = np.array(loss)
        loss_dict = {"regression": loss}
        log_losses(loss, loss_dict, scalar_dict, it=it//3, tag='train', logger=logger, writer=writer)              
