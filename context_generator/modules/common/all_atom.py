import jax
import jax.numpy as jnp
import numpy as np

from typing import Dict, Optional

from context_generator.modules.model import r3
from context_generator.utils.protein import residue_constants
from context_generator.utils.protein import parsers

def squared_difference(x, y):
  return jnp.square(x - y)

def get_chi_atom_indices():
  """Returns atom indices needed to compute chi angles for all residue types.

  Returns:
    A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
    in the order specified in residue_constants.restypes + unknown residue type
    at the end. For chi angles which are not defined on the residue, the
    positions indices are by default set to 0.
  """
  chi_atom_indices = []
  for residue_name in residue_constants.restypes:
    residue_name = residue_constants.restype_1to3[residue_name]
    residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
    atom_indices = []
    for chi_angle in residue_chi_angles:
      atom_indices.append(
          [residue_constants.atom_order[atom] for atom in chi_angle])
    for _ in range(4 - len(atom_indices)):
      atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
    chi_atom_indices.append(atom_indices)

  chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

  return jnp.asarray(chi_atom_indices)

def atom37_to_torsion_angles(
    aatype: jnp.ndarray,  # (B, N)
    all_atom_pos: jnp.ndarray,  # (B, N, 37, 3)
    all_atom_mask: jnp.ndarray,  # (B, N, 37)
    placeholder_for_undefined=False,
) -> Dict[str, jnp.ndarray]:
  """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

  The 7 torsion angles are in the order
  '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
  here pre_omega denotes the omega torsion angle between the given amino acid
  and the previous amino acid.

  Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_pos: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
    placeholder_for_undefined: flag denoting whether to set masked torsion
      angles to zero.
  Returns:
    Dict containing:
      * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
        2 dimensions denote sin and cos respectively
      * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
        with the angle shifted by pi for all chi angles affected by the naming
        ambiguities.
      * 'torsion_angles_mask': Mask for which chi angles are present.
  """

  # Map aatype > 20 to 'Unknown' (20).
  aatype = jnp.minimum(aatype, 20)

  # Compute the backbone angles.
  num_batch, num_res = aatype.shape

  pad = jnp.zeros([num_batch, 1, 37, 3], jnp.float32)
  prev_all_atom_pos = jnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

  pad = jnp.zeros([num_batch, 1, 37], jnp.float32)
  prev_all_atom_mask = jnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

  # For each torsion angle collect the 4 atom positions that define this angle.
  # shape (B, N, atoms=4, xyz=3)
  pre_omega_atom_pos = jnp.concatenate(
      [prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
       all_atom_pos[:, :, 0:2, :]  # this N, CA
      ], axis=-2)
  phi_atom_pos = jnp.concatenate(
      [prev_all_atom_pos[:, :, 2:3, :],  # prev C
       all_atom_pos[:, :, 0:3, :]  # this N, CA, C
      ], axis=-2)
  psi_atom_pos = jnp.concatenate(
      [all_atom_pos[:, :, 0:3, :],  # this N, CA, C
       all_atom_pos[:, :, 4:5, :]  # this O
      ], axis=-2)

  # Collect the masks from these atoms.
  # Shape [batch, num_res]
  pre_omega_mask = (
      jnp.prod(prev_all_atom_mask[:, :, 1:3], axis=-1)  # prev CA, C
      * jnp.prod(all_atom_mask[:, :, 0:2], axis=-1))  # this N, CA
  phi_mask = (
      prev_all_atom_mask[:, :, 2]  # prev C
      * jnp.prod(all_atom_mask[:, :, 0:3], axis=-1))  # this N, CA, C
  psi_mask = (
      jnp.prod(all_atom_mask[:, :, 0:3], axis=-1) *  # this N, CA, C
      all_atom_mask[:, :, 4])  # this O

  # Collect the atoms for the chi-angles.
  # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
  chi_atom_indices = get_chi_atom_indices()
  # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
  atom_indices = parsers.batched_gather(
      params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0)
  # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
  chis_atom_pos = parsers.batched_gather(
      params=all_atom_pos, indices=atom_indices, axis=-2,
      batch_dims=2)

  # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
  chi_angles_mask = list(residue_constants.chi_angles_mask)
  chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
  chi_angles_mask = jnp.asarray(chi_angles_mask)

  # Compute the chi angle mask. I.e. which chis angles exist according to the
  # aatype. Shape [batch, num_res, chis=4].
  chis_mask = parsers.batched_gather(params=chi_angles_mask, indices=aatype,
                                   axis=0, batch_dims=0)

  # Constrain the chis_mask to those chis, where the ground truth coordinates of
  # all defining four atoms are available.
  # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
  chi_angle_atoms_mask = parsers.batched_gather(
      params=all_atom_mask, indices=atom_indices, axis=-1,
      batch_dims=2)
  # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
  chi_angle_atoms_mask = jnp.prod(chi_angle_atoms_mask, axis=[-1])
  chis_mask = chis_mask * (chi_angle_atoms_mask).astype(jnp.float32)

  # Stack all torsion angle atom positions.
  # Shape (B, N, torsions=7, atoms=4, xyz=3)
  torsions_atom_pos = jnp.concatenate(
      [pre_omega_atom_pos[:, :, None, :, :],
       phi_atom_pos[:, :, None, :, :],
       psi_atom_pos[:, :, None, :, :],
       chis_atom_pos
      ], axis=2)

  # Stack up masks for all torsion angles.
  # shape (B, N, torsions=7)
  torsion_angles_mask = jnp.concatenate(
      [pre_omega_mask[:, :, None],
       phi_mask[:, :, None],
       psi_mask[:, :, None],
       chis_mask
      ], axis=2)

  # Create a frame from the first three atoms:
  # First atom: point on x-y-plane
  # Second atom: point on negative x-axis
  # Third atom: origin
  # r3.Rigids (B, N, torsions=7)
  torsion_frames = r3.rigids_from_3_points(
      point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
      origin=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
      point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]))

  # Compute the position of the forth atom in this frame (y and z coordinate
  # define the chi angle)
  # r3.Vecs (B, N, torsions=7)
  forth_atom_rel_pos = r3.rigids_mul_vecs(
      r3.invert_rigids(torsion_frames),
      r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))

  # Normalize to have the sin and cos of the torsion angle.
  # jnp.ndarray (B, N, torsions=7, sincos=2)
  torsion_angles_sin_cos = jnp.stack(
      [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
  torsion_angles_sin_cos /= jnp.sqrt(
      jnp.sum(jnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True)
      + 1e-8)

  # Mirror psi, because we computed it from the Oxygen-atom.
  torsion_angles_sin_cos *= jnp.asarray(
      [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

  # Create alternative angles for ambiguous atom names.
  chi_is_ambiguous = parsers.batched_gather(
      jnp.asarray(residue_constants.chi_pi_periodic), aatype)
  mirror_torsion_angles = jnp.concatenate(
      [jnp.ones([num_batch, num_res, 3]),
       1.0 - 2.0 * chi_is_ambiguous], axis=-1)
  alt_torsion_angles_sin_cos = (
      torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

  if placeholder_for_undefined:
    # Add placeholder torsions in place of undefined torsion angles
    # (e.g. N-terminus pre-omega)
    placeholder_torsions = jnp.stack([
        jnp.ones(torsion_angles_sin_cos.shape[:-1]),
        jnp.zeros(torsion_angles_sin_cos.shape[:-1])
    ], axis=-1)
    torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
    alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

  return {
      'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, N, 7, 2)
      'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
      'torsion_angles_mask': torsion_angles_mask  # (B, N, 7)
  }

def torsion_angles_to_frames(
        aatype: jnp.ndarray, # (B, N)
        backb_to_global: r3.Rigids, # (B, N)
        torsion_angles_sin_cos: jnp.ndarray  # (B, N, 7, 2)
) -> r3.Rigids:  # (B, N, 8)
  """Compute rigid group frames from torsion angles.
  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
  Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"
  Args:
    aatype: aatype for each residue
    backb_to_global: Rigid transformations describing transformation from
      backbone frame to global frame.
    torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
  Returns:
    Frames corresponding to all the Sidechain Rigid Transforms
  """

  # convert_dict = {
  #     0: 0, 1: 4, 2: 3, 3: 6, 4: 13,
  #     5: 7, 6: 8, 7: 9, 8: 11, 9: 10,
  #     10: 12, 11: 2, 12: 14, 13: 5, 14: 1,
  #     15: 15, 16: 16, 17: 19, 18: 17, 19: 18,
  #     20: 20, 21: 20
  # }
  # vectorized_map = np.vectorize(convert_dict.get)
  # aa_af2paradigm = vectorized_map(aatype)

  # Gather the default frames for all rigid groups
  # r3.Rigids with shape (B, N, 8)
  m = parsers.batched_gather(residue_constants.restype_rigid_group_default_frame, aatype)
  default_frames = r3.rigids_from_tensor4x4(m)

  # Create the rotation matrices according to the given angles (each frame is
  # defined such that its rotation is around the x-axis).  
  sin_angles = torsion_angles_sin_cos[...,0]
  cos_angles = torsion_angles_sin_cos[...,1]

  batch_size, num_residues, = aatype.shape
  sin_angles = jnp.concatenate([jnp.zeros([batch_size, num_residues, 1]), sin_angles], axis=-1)
  cos_angles = jnp.concatenate([jnp.ones([batch_size, num_residues, 1]), cos_angles], axis=-1)

  zeros = jnp.zeros_like(sin_angles)
  ones = jnp.ones_like(sin_angles) 

  # all_rots are r3.Rots with shape (B, N, 8)
  all_rots = r3.Rots(ones, zeros, zeros,
                      zeros, cos_angles, -sin_angles,
                      zeros, sin_angles, cos_angles)   
  
  # Apply rotations to the frames.
  all_frames = r3.rigids_mul_rots(default_frames, all_rots)

  # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
  # the previous frame. So chain them up accordingly.
  chi2_frame_to_frame = jax.tree_util.tree_map(lambda x: x[..., 5], all_frames)
  chi3_frame_to_frame = jax.tree_util.tree_map(lambda x: x[..., 6], all_frames)
  chi4_frame_to_frame = jax.tree_util.tree_map(lambda x: x[..., 7], all_frames)

  chi1_frame_to_backb = jax.tree_util.tree_map(lambda x: x[..., 4], all_frames)
  chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,
                                              chi2_frame_to_frame)
  chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,
                                              chi3_frame_to_frame)
  chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,
                                              chi4_frame_to_frame)  

  # Recombine them to a r3.Rigids with shape (B, N, 8).
  def _concat_frames(xall, x5, x6, x7):
    return jnp.concatenate(
        [xall[..., 0:5], x5[..., None], x6[..., None], x7[..., None]], axis=-1)

  all_frames_to_backb = jax.tree_util.tree_map(
      _concat_frames,
      all_frames,
      chi2_frame_to_backb,
      chi3_frame_to_backb,
      chi4_frame_to_backb)

  # Create the global frames.
  # shape (B, N, 8)

  all_frames_to_global = r3.rigids_mul_rigids(
      jax.tree_util.tree_map(lambda x: x[..., None], backb_to_global),
      all_frames_to_backb)
  
  return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(
    aatype: jnp.ndarray,  # (B, N)
    all_frames_to_global: r3.Rigids  # (B, N, 8)
) -> r3.Vecs:  # (B, N, 14)
  """Put atom literature positions (atom14 encoding) in each rigid group.

  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

  Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
  Returns:
    Positions of all atom coordinates in global frame.
  """

  # convert_dict = {
  #     0: 0, 1: 4, 2: 3, 3: 6, 4: 13,
  #     5: 7, 6: 8, 7: 9, 8: 11, 9: 10,
  #     10: 12, 11: 2, 12: 14, 13: 5, 14: 1,
  #     15: 15, 16: 16, 17: 19, 18: 17, 19: 18,
  #     20: 20, 21: 20
  # }
  # vectorized_map = np.vectorize(convert_dict.get)
  # aa_af2paradigm = vectorized_map(aatype)

  # Pick the appropriate transform for every atom.
  residx_to_group_idx = parsers.batched_gather(
      residue_constants.restype_atom14_to_rigid_group, aatype)
  group_mask = jax.nn.one_hot(
      residx_to_group_idx, num_classes=8)  # shape (B, N, 14, 8)

  # r3.Rigids with shape (B, N, 14)
  map_atoms_to_global = jax.tree_util.tree_map(
      lambda x: jnp.sum(x[..., None, :] * group_mask, axis=-1),
      all_frames_to_global)

  # Gather the literature atom positions for each residue.
  # r3.Vecs with shape (B, N, 14)
  lit_positions = r3.vecs_from_tensor(
      parsers.batched_gather(
          residue_constants.restype_atom14_rigid_group_positions, aatype))

  # Transform each atom from its local frame to the global frame.
  # r3.Vecs with shape (B, N, 14)
  pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

  # Mask out non-existing atoms.
  mask = parsers.batched_gather(residue_constants.restype_atom14_mask, aatype)
  pred_positions = jax.tree_util.tree_map(lambda x: x * mask, pred_positions)

  return pred_positions, mask


def within_residue_violations(
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_exists: jnp.ndarray,  # (N, 14)
    atom14_dists_lower_bound: jnp.ndarray,  # (N, 14, 14)
    atom14_dists_upper_bound: jnp.ndarray,  # (N, 14, 14)
    tighten_bounds_for_loss=0.0,
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes within residues.

  This is a loss penalizing any steric violations or clashes of non-bonded atoms
  in a given peptide. This loss corresponds to the part with
  the same residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_dists_lower_bound: Lower bound on allowed distances.
    atom14_dists_upper_bound: Upper bound on allowed distances
    tighten_bounds_for_loss: Extra factor to tighten loss

  Returns:
    Dict containing:
      * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
      * 'per_atom_clash_mask': mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 4
  assert len(atom14_atom_exists.shape) == 3
  assert len(atom14_dists_lower_bound.shape) == 4
  assert len(atom14_dists_upper_bound.shape) == 4

  # Compute the mask for each residue.
  # shape (N, 14, 14)
  dists_masks = (1. - jnp.eye(14, 14)[None, None])
  dists_masks *= (atom14_atom_exists[:, :, :, None] *
                  atom14_atom_exists[:, :, None, :])
  
  rotamer_square_mask = np.array([False]*5+[True]*9)[...,None]+np.array([False]*5+[True]*9)[None]
  dists_masks *= rotamer_square_mask[None,None]

  # Distance matrix
  # shape (N, 14, 14)
  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, :, :, None, :],
          atom14_pred_positions[:, :, None, :, :]),
      axis=-1))

  # Compute the loss.
  # shape (N, 14, 14)
  dists_to_low_error = jax.nn.relu(
      atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
  dists_to_high_error = jax.nn.relu(
      dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
  loss = dists_masks * (dists_to_low_error + dists_to_high_error)

  # Compute the per atom loss sum.
  # shape (N, 14)
  per_atom_loss_sum = (jnp.sum(loss, axis=2) +
                       jnp.sum(loss, axis=3))

  # print(per_atom_loss_sum.shape)
  # Compute the violations mask.
  # shape (N, 14, 14)
  violations = dists_masks * ((dists < atom14_dists_lower_bound) |
                              (dists > atom14_dists_upper_bound))

  # Compute the per atom violations.
  # shape (N, 14)
  per_atom_violations = jnp.maximum(
      jnp.max(violations, axis=2), jnp.max(violations, axis=3))

  return {'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
          'per_atom_violations': per_atom_violations  # shape (N, 14)
         }


def between_residue_clash_loss(
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_exists: jnp.ndarray,  # (N, 14)
    atom14_atom_radius: jnp.ndarray,  # (N, 14)
    residue_index: jnp.ndarray,  # (N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes between residues.

  This is a loss penalizing any steric clashes due to non bonded atoms in
  different peptides coming too close. This loss corresponds to the part with
  different residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_atom_radius: Van der Waals radius for each atom.
    residue_index: Residue index for given amino acid.
    overlap_tolerance_soft: Soft tolerance factor.
    overlap_tolerance_hard: Hard tolerance factor.

  Returns:
    Dict containing:
      * 'mean_loss': average clash loss
      * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
      * 'per_atom_clash_mask': mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 4
  assert len(atom14_atom_exists.shape) == 3
  assert len(atom14_atom_radius.shape) == 3
  assert len(residue_index.shape) == 2

  # Create the distance matrix.
  # (N, N, 14, 14)
  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, :, None, :, None, :],
          atom14_pred_positions[:, None, :, None, :, :]),
      axis=-1))

  # Create the mask for valid distances.
  # shape (N, N, 14, 14)
  dists_mask = (atom14_atom_exists[:, :, None, :, None] *
                atom14_atom_exists[:, None, :, None, :])
  rotamer_square_mask = np.array([False]*5+[True]*9)[...,None]+np.array([False]*5+[True]*9)[None]
  dists_mask *= rotamer_square_mask[None,None,None]
  # Mask out all the duplicate entries in the lower triangular matrix.
  # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
  # are handled separately.
  dists_mask *= (
      residue_index[:, :, None, None, None] != residue_index[:, None, :, None, None])

  # Backbone C--N bond between subsequent residues is no clash.
  c_one_hot = jax.nn.one_hot(2, num_classes=14)
  n_one_hot = jax.nn.one_hot(0, num_classes=14)
  neighbour_mask = ((residue_index[:, :, None, None, None] +
                     1) == residue_index[:, None, :, None, None])
  c_n_bonds = neighbour_mask * c_one_hot[None, None, None, :,
                                         None] * n_one_hot[None, None, None, None, :]
  dists_mask *= (1. - c_n_bonds)

  # Disulfide bridge between two cysteines is no clash.
  cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
  cys_sg_one_hot = jax.nn.one_hot(cys_sg_idx, num_classes=14)
  disulfide_bonds = (cys_sg_one_hot[None, None, None, :, None] *
                     cys_sg_one_hot[None, None, None, None, :])
  dists_mask *= (1. - disulfide_bonds)

  # Compute the lower bound for the allowed distances.
  # shape (N, N, 14, 14)
  dists_lower_bound = dists_mask * (atom14_atom_radius[:, :, None, :, None] +
                                    atom14_atom_radius[:, None, :, None, :])

  # Compute the error.
  # shape (N, N, 14, 14)
  dists_to_low_error = dists_mask * jax.nn.relu(
      dists_lower_bound - overlap_tolerance_soft - dists)

  # Compute the mean loss.
  # shape ()
  mean_loss = (jnp.sum(dists_to_low_error)
               / (1e-6 + jnp.sum(dists_mask)))

  # Compute the per atom loss sum.
  # shape (N, 14)
  per_atom_loss_sum = (jnp.sum(dists_to_low_error, axis=[1, 3]) +
                       jnp.sum(dists_to_low_error, axis=[2, 4]))

  # Compute the hard clash mask.
  # shape (N, N, 14, 14)
  clash_mask = dists_mask * (
      dists < (dists_lower_bound - overlap_tolerance_hard))

  # Compute the per atom clash.
  # shape (N, 14)
  per_atom_clash_mask = jnp.maximum(
      jnp.max(clash_mask, axis=[1, 3]),
      jnp.max(clash_mask, axis=[2, 4]))

  return {'mean_loss': mean_loss,  # shape ()
          'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
          'per_atom_clash_mask': per_atom_clash_mask  # shape (N, 14)
         }
