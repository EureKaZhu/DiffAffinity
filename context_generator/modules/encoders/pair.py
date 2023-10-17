import jax
import jax.numpy as jnp
import haiku as hk

import dataclasses
from typing import Optional

from context_generator.modules.common.layers import AngularEncoding
from context_generator.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from context_generator.utils.protein.constants import BBHeavyAtom

@dataclasses.dataclass
class ResiduePairEncoder(hk.Module):

    feat_dim: int
    max_num_atoms: int
    max_aa_types: int = 22
    max_relpos: int = 32
    name: Optional[str] = None

    def __call__(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.shape
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # Pair identities
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        # feat_aapair = self.aa_pair_embed(aa_pair)
        aa_pair_embed = hk.Embed(vocab_size=self.max_aa_types*self.max_aa_types, embed_dim=self.feat_dim)
        feat_aapair = aa_pair_embed(aa_pair)

        # Relative positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = jax.lax.clamp(
            min=-self.max_relpos,
            x = res_nb[:,:,None] - res_nb[:,None,:], 
            max=self.max_relpos
        ) # (N, L, L)
        # feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]
        relpos_embed = hk.Embed(vocab_size=2*self.max_relpos+1, embed_dim=self.feat_dim)
        feat_relpos = relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]
        
        # Distances
        d = angstrom_to_nm(jnp.linalg.norm(
            pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            axis=-1, ord=2,
        )).reshape(N, L, L, -1) # (N, L, L, A*A)
        # c = jax.nn.softplus(self.aapair_to_distcoef(aa_pair))   # (N, L, L, A*A)
        aapair_to_distcoef = hk.Embed(
            vocab_size=self.max_aa_types*self.max_aa_types,
            embed_dim=self.max_num_atoms*self.max_num_atoms,
            w_init=hk.initializers.Constant(0.0)
        )
        c = jax.nn.softplus(aapair_to_distcoef(aa_pair))   # (N, L, L, A*A)
        d_gauss = jnp.exp(-1 * c * d**2)
        mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, -1)
        # feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        distance_embed = hk.Sequential([
            hk.Linear(self.feat_dim), jax.nn.relu,
            hk.Linear(self.feat_dim), jax.nn.relu
        ])
        feat_dist = distance_embed(d_gauss * mask_atom_pair)

        # HACK
        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        # feat_dihed = self.dihedral_embed(dihed)
        feat_dihed = AngularEncoding()(dihed)


        # All
        feat_all = jnp.concatenate([feat_aapair, feat_relpos, feat_dist, feat_dihed], axis=-1)
        # feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        out_mlp = hk.Sequential([
            hk.Linear(self.feat_dim), jax.nn.relu,
            hk.Linear(self.feat_dim), jax.nn.relu,
            hk.Linear(self.feat_dim)
        ])
        feat_all = out_mlp(feat_all)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all