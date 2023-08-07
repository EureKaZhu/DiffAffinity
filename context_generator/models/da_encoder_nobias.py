import jax
from jax.experimental.checkify import checkify, index_checks
import jax.numpy as jnp
import haiku as hk

import einops
import dataclasses
from typing import Optional
from functools import partial

from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder
from context_generator.utils.protein.constants import BBHeavyAtom

@dataclasses.dataclass
class RDEEncoderWOBias(hk.Module):

    single_encoder: PerResidueEncoder
    # masked_bias: hk.Embed
    pair_encoder: ResiduePairEncoder
    attn_encoder: GAEncoder

    def __call__(
            self,
            batch: dict,
    ):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi_corrupt']
        
        x = self.single_encoder(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,            
        )
        # b = self.masked_bias(batch['chi_masked_flag'].astype('int32'))
        # b = b * batch['chi_masked_flag'][...,jnp.newaxis]
        # x = x + b
        z = self.pair_encoder(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_atoms'], mask_atoms = batch['mask_atoms'],            
        )

        x = self.attn_encoder(
            pos_atoms = batch['pos_atoms'], 
            res_feat = x, pair_feat = z, 
            mask = mask_residue
        )
        return x 
