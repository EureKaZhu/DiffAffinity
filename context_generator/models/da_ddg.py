import jax
from jax.experimental.checkify import checkify, index_checks
import jax.numpy as jnp
import haiku as hk
import easydict

import einops
import dataclasses
from typing import Optional
from functools import partial

from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder
from context_generator.utils.protein.constants import BBHeavyAtom
from context_generator.models.da_encoder_nobias import RDEEncoderWOBias

@dataclasses.dataclass
class DDG_RDE_Network(hk.Module):

    context_encoder: RDEEncoderWOBias
    cfg: easydict.EasyDict
    single_encoder_ddg: PerResidueEncoder
    pair_encoder_ddg: ResiduePairEncoder
    attn_encoder_ddg: GAEncoder

    def __call__(self, batch:dict):
        batch = {k: v for k, v in batch.items()} 
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        x_pret = self.context_encoder(batch)

        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].astype('float32'))[:, :, None]
        
        x_single = self.single_encoder_ddg(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,            
        )     

        dim = self.cfg.model.encoder.node_feat_dim
        single_fusion = hk.Sequential([
            hk.Linear(dim), jax.nn.relu,
            hk.Linear(dim)            
        ])
        x = single_fusion(jnp.concatenate([x_single, x_pret], axis=-1))
        # x = single_fusion(jnp.concatenate([x_single,], axis=-1))

        mut_bias = hk.Embed(
            vocab_size=2,
            embed_dim=dim
        )
        b = mut_bias(batch['mut_flag'].astype('int32'))
        b = b * batch['mut_flag'][...,jnp.newaxis]
        x = x + b

        z = self.pair_encoder_ddg(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_atoms'], mask_atoms = batch['mask_atoms'],              
        )

        x = self.attn_encoder_ddg(
            pos_atoms = batch['pos_atoms'], 
            res_feat = x, pair_feat = z, 
            mask = mask_residue                
        )

        return x

        

