import jax
import jax.numpy as jnp

from context_generator.utils.protein.constants import BBHeavyAtom

def angstrom_to_nm(x):
    return x / 10

def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = jnp.cross(v0, v1, axis=-1)
    n1 = u1 / jnp.linalg.norm(u1, axis=-1, keepdims=True)
    u2 = jnp.cross(v0, v2, axis=-1)
    n2 = u2 / jnp.linalg.norm(u2, axis=-1, keepdims=True)
    sgn = jnp.sign( (jnp.cross(v1, v2, axis=-1) * v0).sum(-1) )
    dihed = sgn * jnp.arccos( (n1 * n2).sum(-1) )
    dihed = jnp.nan_to_num(dihed)
    return dihed

def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        # pos_C[:,:,None].expand(N, L, L, 3), 
        # pos_N[:,None,:].expand(N, L, L, 3), 
        # pos_CA[:,None,:].expand(N, L, L, 3), 
        # pos_C[:,None,:].expand(N, L, L, 3)
        pos_C[:,:,None].repeat(L, axis=2),
        pos_N[:,None,:].repeat(L, axis=1),
        pos_CA[:,None,:].repeat(L, axis=1),
        pos_C[:,None,:].repeat(L, axis=1),
    )
    ir_psi = dihedral_from_four_points(
        # pos_N[:,:,None].expand(N, L, L, 3), 
        # pos_CA[:,:,None].expand(N, L, L, 3), 
        # pos_C[:,:,None].expand(N, L, L, 3), 
        # pos_N[:,None,:].expand(N, L, L, 3)
        pos_N[:,:,None].repeat(L, axis=2), 
        pos_CA[:,:,None].repeat(L, axis=2), 
        pos_C[:,:,None].repeat(L, axis=2), 
        pos_N[:,None,:].repeat(L, axis=1)
    )
    ir_dihed = jnp.stack([ir_phi, ir_psi], axis=-1)
    return ir_dihed

def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.shape[-1] == 3
    q_size = q.shape
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).swapaxes(-1,-2)   # (N, L, *, 3) -> (N, L, 3, *)
    p = jnp.matmul(R.swapaxes(-1,-2), (q - jnp.expand_dims(t,-1)))  # (N, L, 3, *)
    p = p.swapaxes(-1,-2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p

def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.shape[-1] == 3
    p_size = p.shape
    N, L = p_size[0], p_size[1]

    p = p.reshape(N, L, -1, 3).swapaxes(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = jnp.matmul(R, p) + t[...,jnp.newaxis]    # (N, L, 3, *)
    q = q.swapaxes(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q

def normalize_vector(v, axis, eps=1e-6):
    return v / (jnp.linalg.norm(v, ord=2, axis=axis, keepdims=True) + eps)

def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, axis=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, axis=-1)
    e2 = normalize_vector(u2, axis=-1)

    e3 = jnp.cross(e1, e2, axis=-1)    # (N, L, 3)

    mat = jnp.concatenate([
        e1[...,jnp.newaxis], e2[...,jnp.newaxis], e3[...,jnp.newaxis]
    ], axis=-1)  # (N, L, 3, 3_index)

    return mat

def project_v2v(v, e, axis):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(axis=axis, keepdims=True) * e