import os
import copy
import argparse
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm

from typing import Callable, NamedTuple
from einops import rearrange
import abc


import haiku as hk
import jax 
import optax


from score_sde.utils import TrainState, save, restore

from context_generator.utils.misc import load_config
from context_generator.utils.data import PaddingCollate
from context_generator.utils.train import *
from context_generator.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch
from context_generator.utils.protein.parsers import parse_biopython_structure
from context_generator.models.da_ddg import DDG_RDE_Network
from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder
from context_generator.models.da_encoder import RDEEncoder

from context_generator.utils.skempi import eval_skempi


rng = jax.random.PRNGKey(42)

class Freezable_TrainState(NamedTuple):
    trainable_params: hk.Params
    non_trainable_params: hk.Params
    opt_state: optax.OptState


class PMDataset(Dataset):

    def __init__(self, pdb_path, mutations):
        super().__init__()
        self.pdb_path = pdb_path

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.mutations = self._parse_mutations(mutations)
        self.transform = Compose([
            SelectAtom('backbone+CB'),
            SelectedRegionFixedSizePatch('mut_flag', 128)
        ])

    
    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_structure(self):
        if self.pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif self.pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        structure = parser.get_structure(None, self.pdb_path)
        data, seq_map = parse_biopython_structure(structure[0])
        self.data = data
        self.seq_map = seq_map

    @abc.abstractclassmethod
    def _parse_mutations(self, mutations):
        pass

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        data = self.clone_data()
        mut = self.mutations[index]
        mut_pos_idx = self.seq_map[mut['position']]

        data['mut_flag'] = torch.zeros(size=data['aa'].shape, dtype=torch.bool)
        data['mut_flag'][mut_pos_idx] = True
        data['aa_mut'] = data['aa'].clone()
        data['aa_mut'][mut_pos_idx] = one_to_index(mut['mt'])
        data = self.transform(data)
        data['ddG'] = mut['ddg'] # 预测7FAE时似乎不需要这个key
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )
        return data

class _7FAEDataset(PMDataset):

    def __init__(self, pdb_path, mutations):
        super().__init__(pdb_path, mutations)
    
    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq, ' ')
            if pos not in self.seq_map: continue

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'position': pos,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'position': pos,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

class _6M0JDataset(PMDataset):

    def __init__(self, pdb_path, mutations):
        super().__init__(pdb_path, mutations)

    def _parse_mutations(self, mutations):
        parsed = []
        df = pd.read_csv(mutations)
        for m, ddg in zip(df['mutation'], df['delta_bind']):
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq, ' ')
            if pos not in self.seq_map: continue

            parsed.append({
                'ddg': ddg,
                'position': pos,
                'wt': wt,
                'mt': mt
            })
        return parsed    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-o', '--output', type=str, default='./pm_results.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    config, _ = load_config(args.config)

    rde_config, rde_config_name = load_config("context_generator/configs/train/diff.yml")

    if config.name == "7FAE":
        dataset = _7FAEDataset(
            pdb_path = config.pdb,
            mutations = config.mutations,
        )
    elif config.name == "6M0J":
        dataset = _6M0JDataset(
            pdb_path = config.pdb,
            mutations = config.mutations,            
        )

    else:
        raise ValueError("Target protein not exists..")

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=PaddingCollate(), 
    )

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


        dim = config.model.encoder.node_feat_dim
        H_mt, H_wt = h_mt.max(axis=1), h_wt.max(axis=1)

        ddg_readout = hk.Sequential([
            hk.Linear(dim, name="trainable_linear"), jax.nn.relu,
            hk.Linear(dim, name="trainable_linear_1"), jax.nn.relu,
            hk.Linear(1, name="trainable_linear_2")
        ])
        ddg_pred = ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = ddg_readout(H_wt - H_mt).squeeze(-1)
        # loss = (jnp.power(ddg_pred, batch['ddG'], 2).mean() + jnp.power(ddg_pred_inv, batch['ddG'], 2).mean()) / 2
        return ddg_pred, ddg_pred_inv

    model = hk.transform(_forward)

    rng, next_rng = jax.random.split(rng)
    ibatch = next(iter(loader))
    params = model.init(rng=next_rng, batch=ibatch)

    train_state = restore(config.checkpoint)
    trainable_params, non_trainable_params, opt_state = train_state
    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    
    if config.name == "7FAE":
        result = []
        for batch in tqdm(loader):
            ddg_pred, ddg_pred_inv = model.apply(params, None, batch)
            for mutstr, ddG_pred in zip(batch['mutstr'], np.array(ddg_pred)):
                result.append(
                    {
                        'mutstr': mutstr,
                        'ddG_pred': ddG_pred,
                    }
                )
        result = pd.DataFrame(result)
        result = result.groupby('mutstr').mean().reset_index()
        result['rank'] = result['ddG_pred'].rank() / len(result)
        print(f'Results saved to {args.output}.')
        result.to_csv(args.output)

        if 'interest' in config and config.interest:
            print(result[result['mutstr'].isin(config.interest)])
    
    elif config.name == "6M0J":
        result = []
        for batch in tqdm(loader):
            ddg_pred, ddg_pred_inv = model.apply(params, None, batch)
            for mutstr, ddG_pred, ddG_true in zip(batch['mutstr'], np.array(ddg_pred), batch['ddG']):
                result.append(
                    {
                        'mutstr': mutstr,
                        'ddG_pred': ddG_pred,
                        'ddG': -ddG_true,
                        'complex': '6m0j'
                    }
                )
        result = pd.DataFrame(result)  
        result['method'] = 'DiffAffinity'
        result.to_csv(args.output)
        df_metrics = eval_skempi(result, 'all')
        print(df_metrics)      
