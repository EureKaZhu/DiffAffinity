import os
import pickle

import jax
import numpy as np
import jax.numpy as jnp
import jax.lib.xla_bridge as xb
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten

from .typing import ScoreFunction
import einops


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_estimate_div_fn(fn: ScoreFunction):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(y: jnp.ndarray, t: float, context: jnp.ndarray, eps: jnp.ndarray):
        eps = eps.reshape(eps.shape[0], -1)
        grad_fn = lambda y: jnp.sum(fn(y, t, context) * eps)
        grad_fn_eps = jax.grad(grad_fn)(y).reshape(y.shape[0], -1)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(eps.shape))))

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(y: jnp.ndarray, t: float, context: dict):
        if len(y.shape) ==3:
            N, L , _ = y.shape
            # y_shape = y.shape
            # dim = np.prod(y_shape[1:])
            # t = jnp.expand_dims(t.reshape(-1), axis=-1)
            y = jnp.expand_dims(y, 1)  # NOTE: need leading batch dim after vmap
            # if context is not None:
            #     context = jnp.expand_dims(context, 1)
            t = jnp.expand_dims(t, 1)
            context_expanded = {k:jnp.expand_dims(v, 1) for k, v in context.items() if (not isinstance(v,list)) and (not isinstance(v,int))}
            jac = jax.vmap(jax.jacrev(fn, argnums=0))(y, t, context_expanded)
            # jac = jax.jacrev(fn, argnums=0)(y, t, context)
            # jac = jac[jnp.arange(y_shape[0]),:,jnp.arange(y_shape[0]),:]
            # print("jac", jac.shape)
            
            jac = jnp.stack([jac[i,jnp.arange(L),:,:,jnp.arange(L),:] for i in range(N)])
            # print(jac.shape)
            # print(jac.squeeze().shape)
            jac = einops.rearrange(jac.squeeze(), 'b l m n -> (b l) m n')
            chi_mask = einops.rearrange(jnp.stack([context['chi_mask'], context['chi_mask']], axis=-1), 'b l n c -> (b l) (n c)')
            chi_mask2d = einops.einsum(jnp.expand_dims(chi_mask,axis=2), jnp.expand_dims(chi_mask,axis=1), 'h i j, h j k -> h i k')
            chi_corrupt_flag = einops.rearrange(context['chi_corrupt_flag'], 'b l -> (b l)')[...,jnp.newaxis, jnp.newaxis]
            chi_complete = einops.rearrange(context['chi_complete'], 'b l -> (b l)')[...,jnp.newaxis, jnp.newaxis]
            div_mask = chi_mask2d & chi_corrupt_flag & chi_complete
            # print(jac.shape)
            # print(div_mask.shape)
            jac = jac * div_mask
            # jax.debug.print("div loss {}", jnp.trace(jac, axis1=-1, axis2=-2))
            return jnp.trace(jac, axis1=-1, axis2=-2)
        else:
            assert len(y.shape) == 2
            y_shape = y.shape
            dim = np.prod(y_shape[1:])
            # t = jnp.expand_dims(t.reshape(-1), axis=-1)
            jac = jax.jacrev(fn, argnums=0)(y, t, context)
            # print(jac.shape)
            jac = jac[np.arange(y_shape[0]),:,np.arange(y_shape[0]),:]
            # print(jac.shape)

            jac = jac.reshape([y_shape[0], dim, dim])
            chi_mask = einops.rearrange(jnp.stack([context['chi_mask'], context['chi_mask']], axis=-1), 'b l n c -> (b l) (n c)')
            chi_mask2d = einops.einsum(jnp.expand_dims(chi_mask,axis=2), jnp.expand_dims(chi_mask,axis=1), 'h i j, h j k -> h i k')
            chi_corrupt_flag = einops.rearrange(context['chi_corrupt_flag'], 'b l -> (b l)')[...,jnp.newaxis, jnp.newaxis]
            chi_complete = einops.rearrange(context['chi_complete'], 'b l -> (b l)')[...,jnp.newaxis, jnp.newaxis]
            div_mask = chi_mask2d & chi_corrupt_flag & chi_complete

            jac = jac * div_mask
            return jnp.trace(jac, axis1=-1, axis2=-2)



    return div_fn


def to_flattened_numpy(x: jnp.ndarray) -> np.ndarray:
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x: np.ndarray, shape: tuple) -> jnp.ndarray:
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


# Borrowed from flax
def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.

    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: `jax.local_devices()`).
    Returns:
      A new pytree containing the replicated arrays.
    """
    if devices is None:
        # match the default device assignments used in pmap:
        # for single-host, that's the XLA default device assignment
        # for multi-host, it's the order of jax.local_devices()
        if jax.process_count() == 1:
            devices = [
                d
                for d in xb.get_backend().get_default_device_assignment(
                    jax.device_count()
                )
                if d.process_index == jax.process_index()
            ]
        else:
            devices = jax.local_devices()

    return jax.device_put_replicated(tree, devices)


# Borrowed from flax
def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return tree_map(lambda x: x[0], tree)


def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return tree_unflatten(treedef, flat_state)
