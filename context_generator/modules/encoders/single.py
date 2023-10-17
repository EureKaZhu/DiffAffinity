import jax
import jax.numpy as jnp
import haiku as hk

import dataclasses
from typing import Optional

from context_generator.modules.common.layers import AngularEncoding

@dataclasses.dataclass
class PerResidueEncoder(hk.Module):

    feat_dim: int
    max_num_atoms: int
    max_aa_types: int = 22
    name: Optional[str] = None
    
    def __call__(self, aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)
        """
        N, L = aa.shape
        # Amino acid identity features
        # aa_feat = self.aatype_embed(aa) # (N, L, feat)
        aa_feat = hk.Embed(vocab_size=self.max_aa_types, embed_dim=self.feat_dim)(aa)

        # Dihedral features
        dihedral = jnp.concatenate(
            [phi[..., None], psi[..., None], chi],
            axis=-1
        ) # (N, L, 6)
        dihedral_mask = jnp.concatenate(
            [phi_mask[..., None], psi_mask[..., None], chi_mask],
            axis=-1
        ) # (N, L, 6)
        # dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 6, feat)
        dihedral_feat = AngularEncoding()(dihedral[..., None]) * dihedral_mask[..., None]
        dihedral_feat = dihedral_feat.reshape(N, L, -1)
        # Mix
        # out_feat = self.mlp(jnp.concatenate([aa_feat, dihedral_feat], axis=-1)) # (N, L, F)
        mlp = hk.Sequential([
            hk.Linear(self.feat_dim * 2), jax.nn.relu,
            hk.Linear(self.feat_dim), jax.nn.relu,
            hk.Linear(self.feat_dim), jax.nn.relu,
            hk.Linear(self.feat_dim)            
        ])
        out_feat = mlp(jnp.concatenate([aa_feat, dihedral_feat], axis=-1))
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat