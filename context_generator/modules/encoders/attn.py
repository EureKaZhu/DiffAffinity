import jax
from jax.experimental.checkify import checkify, index_checks
import jax.numpy as jnp
import haiku as hk

import numpy as np

import einops
import dataclasses
from typing import Optional
from functools import partial

from context_generator.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from context_generator.modules.common.layers import mask_zero
from context_generator.utils.protein.constants import BBHeavyAtom
import logging
log = logging.getLogger(__name__)

def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.shape
    mask_row = mask.reshape(N, L, 1, 1).repeat(logits.shape[-2], -2).repeat(logits.shape[-1], -1)
    mask_pair = mask_row * mask_row.transpose(0,2,1,3)

    logits = jnp.where(mask_pair, logits, logits - inf)
    alpha = jax.nn.softmax(logits, axis=2)  # (N, L, L, num_heads)
    alpha = jnp.where(mask_row, alpha, jnp.zeros_like(alpha))
    return alpha

def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.shape)[:-1] + [n_heads, n_ch]
    return x.reshape(*s)

@dataclasses.dataclass
class GABlock(hk.Module):

    node_feat_dim: int
    pair_feat_dim: int
    value_dim: int = 32
    query_key_dim: int = 32
    num_query_points: int = 8
    num_value_points: int = 8
    num_heads: int = 12
    bias: bool = False
    name: Optional[str] = None

    def _node_logits(self, x):
        proj_query = hk.Linear(self.query_key_dim * self.num_heads, with_bias=self.bias)
        proj_key = hk.Linear(self.query_key_dim * self.num_heads, with_bias=self.bias)
        query_l = _heads(proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (jnp.expand_dims(query_l, 2) * jnp.expand_dims(key_l, 1) *
                       (1 / jnp.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        proj_pair_bias = hk.Linear(self.num_heads, with_bias=self.bias)
        logits_pair = proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.shape
        # Query
        proj_query_point = hk.Linear(self.num_query_points*self.num_heads*3, with_bias=self.bias)
        query_points = _heads(proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points) # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1) # (N, L, n_heads, n_pnts*3)
        # Key
        proj_key_point = hk.Linear(self.num_query_points*self.num_heads*3, with_bias=self.bias)
        key_points = _heads(proj_key_point(x), self.num_heads * self.num_query_points,
                            3) # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)
        # Q-K Product
        sum_sq_dist = ((query_s[:,:,jnp.newaxis,...] - key_s[:,jnp.newaxis,...]) ** 2).sum(-1) # (N, L, L, n_heads)
        spatial_coef = hk.get_parameter(name="spatial_coef", 
                                        shape=[1,1,1,self.num_heads],
                                        init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.)))
        gamma = jax.nn.softplus(spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * jnp.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha[...,jnp.newaxis] * z[...,jnp.newaxis,:] # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(axis=2) # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        proj_value = hk.Linear(self.value_dim * self.num_heads, with_bias=self.bias)
        value_l = _heads(proj_value(x), self.num_heads, self.query_key_dim) # (N, L, n_heads, v_ch)
        feat_node = alpha[...,jnp.newaxis] * value_l[:,jnp.newaxis,...]  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(axis=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)
      
    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.shape
        proj_value_point = hk.Linear(self.num_value_points*self.num_heads*3, with_bias=self.bias)
        value_points = _heads(proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points[:,jnp.newaxis,...]  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(axis=2)  # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = jnp.linalg.norm(feat_points, axis=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, axis=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)
        feat_spatial = jnp.concatenate([
            feat_points.reshape(N, L, -1),
        ], axis=-1)
        return feat_spatial
    
    def __call__(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)

        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        out_transform = hk.Linear(self.node_feat_dim)
        feat_all = out_transform(jnp.concatenate([feat_p2n, feat_node, feat_spatial], axis=-1))  # (N, L, F)

        feat_all = mask_zero(mask[...,jnp.newaxis], feat_all)
        layer_norm_1 = hk.LayerNorm(axis=-1,
                                    create_offset=True,
                                    create_scale=True,)

        x_updated = layer_norm_1(x + feat_all)
        layer_norm_2 = hk.LayerNorm(axis=-1,
                                    create_offset=True,
                                    create_scale=True,)
        mlp_transition = hk.Sequential([
            hk.Linear(self.node_feat_dim), jax.nn.relu,
            hk.Linear(self.node_feat_dim), jax.nn.relu,
            hk.Linear(self.node_feat_dim)
        ])
        x_updated = layer_norm_2(x_updated + mlp_transition(x_updated))

        return x_updated
 
@dataclasses.dataclass
class GAEncoder(hk.Module):
    node_feat_dim: int
    pair_feat_dim: int
    num_layers: int
    name: Optional[str] = None

    def __call__(self,
                 pos_atoms,
                 res_feat,
                 pair_feat,
                 mask):
        R = construct_3d_basis(
            pos_atoms[:,:,BBHeavyAtom.CA],
            pos_atoms[:,:,BBHeavyAtom.C],
            pos_atoms[:,:,BBHeavyAtom.N]
        )
        t = pos_atoms[:,:,BBHeavyAtom.CA]
        t = angstrom_to_nm(t)
        for _ in range(self.num_layers):
            block = GABlock(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim)
            res_feat = block(R, t, res_feat, pair_feat, mask)
        return res_feat