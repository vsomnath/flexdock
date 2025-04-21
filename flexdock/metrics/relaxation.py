import numpy as np
import torch
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from flexdock.geometry.ops import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rigid_transform_kabsch,
)
from flexdock.data.conformers.exceptions import time_limit
from flexdock.data.conformers.molecule import get_symmetry_rmsd

from flexdock.data.feature.protein import get_nearby_residue_mask
from flexdock.models.tensor_ops import index_expand_and_select


BOND_TYPES = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 1.5,
}


def symmetry_adjusted_rmsd(mol, coords, ref_coords, align=False):
    try:
        with time_limit(10):
            G = nx.Graph()
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(), index=atom.GetIdx(), atomic_num=atom.GetAtomicNum()
                )
            for bond in mol.GetBonds():
                G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            dists = torch.linalg.norm(coords[:, None] - coords[None, :], dim=-1)  # noqa
            rmsds, isomorphisms = [], []
            graph_matcher = GraphMatcher(
                G,
                G,
                node_match=lambda node1, node2: node1["atomic_num"]
                == node2["atomic_num"],
            )
            if align:
                Rs, trs = [], []
            for isomorphism in graph_matcher.isomorphisms_iter():
                isomorphism = torch.tensor(
                    [isomorphism[idx] for idx in range(len(isomorphism))],
                    device=coords.device,
                )
                perm_coords = coords.index_select(-2, isomorphism)
                if align:
                    R, tr = rigid_transform_kabsch(perm_coords, ref_coords)
                    perm_coords = perm_coords @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
                    Rs.append(R)
                    trs.append(tr)
                rmsd = torch.sqrt(
                    torch.mean(
                        torch.sum((perm_coords - ref_coords) ** 2, dim=-1), dim=-1
                    )
                )
                rmsds.append(rmsd)
                isomorphisms.append(isomorphism)
            rmsds = torch.stack(rmsds)
            min_rmsd_idx = torch.argmin(rmsds, dim=0)
            min_rmsds = torch.gather(rmsds, 0, min_rmsd_idx.unsqueeze(0)).squeeze()
            best_isomorphisms = torch.stack(isomorphisms)[min_rmsd_idx]
            best_perm_coords = torch.gather(
                coords.expand(*best_isomorphisms.shape, 3),
                -2,
                best_isomorphisms.unsqueeze(-1).expand(*best_isomorphisms.shape, 3),
            )
            if align:
                Rs = torch.stack(Rs)
                trs = torch.stack(trs)
                best_Rs = torch.gather(
                    Rs,
                    0,
                    min_rmsd_idx.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(Rs.shape[1:])
                    .unsqueeze(0),
                ).squeeze()
                best_trs = torch.gather(
                    trs,
                    0,
                    min_rmsd_idx.unsqueeze(-1).expand(trs.shape[1:]).unsqueeze(0),
                ).squeeze()
                return min_rmsds, best_perm_coords, best_Rs, best_trs
            return min_rmsds, best_perm_coords
    except:
        if align:
            R, tr = rigid_transform_kabsch(coords, ref_coords)
            coords = coords @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
        rmsds = torch.sqrt(
            torch.mean(torch.sum((coords - ref_coords) ** 2, dim=-1), dim=-1)
        )
        if align:
            return rmsds, coords, R, tr
        return rmsds, coords


def get_bond_index(mol):
    return torch.tensor(mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))).T


def get_angle_index(mol):
    return torch.tensor(mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))).T


def get_torsion_index(mol):
    return torch.tensor(mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*~*"))).T


def get_ring_index(mol):
    ring_5_index, ring_6_index = [], []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 5:
            ring_5_index.append(list(ring))
        elif len(ring) == 6:
            ring_6_index.append(list(ring))
    ring_5_index = torch.tensor(ring_5_index)
    ring_6_index = torch.tensor(ring_6_index)
    return ring_5_index, ring_6_index


def get_bond_lengths(pos, bond_index):
    r_ij = pos.index_select(-2, bond_index[0]) - pos.index_select(-2, bond_index[1])
    r_ij_norm = torch.linalg.norm(r_ij, dim=-1)

    return r_ij_norm


def get_bond_angles(pos, angle_index):
    r_ij = pos.index_select(-2, angle_index[0]) - pos.index_select(-2, angle_index[1])
    r_kj = pos.index_select(-2, angle_index[2]) - pos.index_select(-2, angle_index[1])

    r_ij_norm = torch.linalg.norm(r_ij, axis=-1)
    r_kj_norm = torch.linalg.norm(r_kj, axis=-1)

    r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)
    r_hat_kj = r_kj / r_kj_norm.unsqueeze(-1)

    cos_theta = (r_hat_ij.unsqueeze(-2) @ r_hat_kj.unsqueeze(-1)).squeeze()
    theta = torch.arccos(cos_theta)

    return theta


def get_torsion_angles(pos, torsion_index):
    r_ij = pos.index_select(-2, torsion_index[0]) - pos.index_select(
        -2, torsion_index[1]
    )
    r_kj = pos.index_select(-2, torsion_index[2]) - pos.index_select(
        -2, torsion_index[1]
    )
    r_kl = pos.index_select(-2, torsion_index[2]) - pos.index_select(
        -2, torsion_index[3]
    )

    n_ijk = torch.cross(r_ij, r_kj)
    n_jkl = torch.cross(r_kj, r_kl)

    r_kj_norm = torch.linalg.norm(r_kj, dim=-1)  # noqa: F841
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

    sign_phi = torch.sign(
        r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl).unsqueeze(-1)
    ).squeeze()
    phi = sign_phi * torch.arccos(
        torch.clamp(
            (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze()
            / (n_ijk_norm * n_jkl_norm),
            -1 + 1e-7,
            1 - 1e-7,
        )
    )
    return phi


def get_puckering_coordinates(pos, ring_index, use_polar=False):
    N = ring_index.shape[-1]
    if use_polar:
        assert N == 6

    ring_pos = index_expand_and_select(pos, -2, ring_index)
    ring_centroid = torch.mean(ring_pos, dim=-2, keepdims=True)
    ring_pos -= ring_centroid

    H = ring_pos.swapaxes(-1, -2) @ ring_pos
    U, S, Vt = torch.linalg.svd(H)
    n = Vt.swapaxes(-1, -2).index_select(-1, torch.tensor(2, device=Vt.device))
    n = n / torch.linalg.norm(n, dim=-1, keepdims=True)
    z = (ring_pos @ n).squeeze(-1)

    m = torch.arange(2, (N - 1) // 2 + 1, device=pos.device).reshape(-1, 1)
    increments = (
        2 * torch.pi * m * torch.arange(N, device=pos.device).reshape(1, -1) / N
    )
    q_sin_phi = (
        -1
        * ((2 / N) ** (1 / 2))
        * torch.sum(z.unsqueeze(-2) * torch.sin(increments), dim=-1).squeeze(-1)
    )
    q_cos_phi = ((2 / N) ** (1 / 2)) * torch.sum(
        z.unsqueeze(-2) * torch.cos(increments), dim=-1
    ).squeeze(-1)

    phi = torch.arctan2(q_sin_phi, q_cos_phi)
    q = q_sin_phi / torch.sin(phi)

    if N % 2 == 0:
        q_N_2 = N ** (-1 / 2) * torch.sum(
            z * (-(1 ** torch.arange(N, device=pos.device))), dim=-1
        )
        if use_polar:
            Q = q**2 + q_N_2**2
            theta = torch.arctan2(q, q_N_2)
            return phi, theta, Q
        return phi, q, q_N_2
    return phi, q


def get_cross_dists(atom_pos, lig_pos, atom_rec_index):
    lig_atom_dists = torch.linalg.norm(
        lig_pos.unsqueeze(-2) - atom_pos.unsqueeze(-3), dim=-1
    )
    lig_rec_dists = torch.scatter_reduce(
        torch.full(
            (*lig_atom_dists.shape[:-1], atom_rec_index.max() + 1),
            torch.inf,
            dtype=lig_atom_dists.dtype,
            device=lig_atom_dists.device,
        ),
        dim=-1,
        src=lig_atom_dists,
        index=atom_rec_index.expand(lig_atom_dists.shape),
        reduce="amin",
        include_self=False,
    )
    return lig_rec_dists


def compute_ligand_alignment_metrics(pos, ref_pos, mol, return_aligned_pos=True):
    if isinstance(pos, np.ndarray):
        pos = torch.tensor(pos, dtype=torch.float32)
        ref_pos = torch.tensor(ref_pos, dtype=torch.float32)

    mol = Chem.RemoveHs(mol)

    rmsds, _ = symmetry_adjusted_rmsd(mol, pos, ref_pos)

    centroids = pos.mean(dim=-2, keepdims=True)
    ref_centroids = ref_pos.mean(dim=-2, keepdims=True)
    centered_pos = pos - centroids
    centered_ref_pos = ref_pos - ref_centroids
    centered_rmsds, _ = symmetry_adjusted_rmsd(mol, centered_pos, centered_ref_pos)
    tr_mags = torch.linalg.norm(centroids - ref_centroids, dim=-1).squeeze(dim=-1)

    aligned_rmsds, aligned_pos, Rs, _ = symmetry_adjusted_rmsd(
        mol, pos, centered_ref_pos, align=True
    )
    rot_mags = torch.abs(torch.linalg.norm(matrix_to_axis_angle(Rs), dim=-1))
    rot_mags = torch.min(rot_mags, 2 * torch.pi - rot_mags)

    # This should be the same as rmsds but computed slower, keeping just to be sure
    try:
        lig_scrmsds = get_symmetry_rmsd(mol, pos.cpu().numpy(), ref_pos.cpu().numpy())
    except:
        lig_scrmsds = (
            torch.sqrt(torch.mean(torch.sum((pos - ref_pos) ** 2, dim=-1), dim=-1))
            .cpu()
            .numpy()
        )

    metric_dict = {
        "lig_rmsds": rmsds.cpu().numpy(),
        "lig_scrmsds": lig_scrmsds,
        "lig_centered_rmsds": centered_rmsds.cpu().numpy(),
        "lig_aligned_rmsds": aligned_rmsds.cpu().numpy(),
        "lig_tr_mags": tr_mags.cpu().numpy(),
        "lig_rot_mags": rot_mags.cpu().numpy(),
    }
    if return_aligned_pos:
        return metric_dict, aligned_pos
    else:
        return metric_dict


def compute_protein_alignment_metrics(
    pos, ref_pos, nearby_atom_mask, ca_mask, c_mask, n_mask
):
    if isinstance(pos, np.ndarray):
        pos = torch.tensor(pos, dtype=torch.float32)
        ref_pos = torch.tensor(pos, dtype=torch.float32)
        nearby_atom_mask = torch.tensor(nearby_atom_mask, dtype=torch.bool)
        ca_mask = torch.tensor(ca_mask, dtype=torch.bool)
        c_mask = torch.tensor(c_mask, dtype=torch.bool)
        n_mask = torch.tensor(n_mask, dtype=torch.bool)

    aa_rmsds = torch.sqrt(
        torch.mean(
            torch.sum(
                (pos[..., nearby_atom_mask, :] - ref_pos[..., nearby_atom_mask, :])
                ** 2,
                dim=-1,
            ),
            dim=-1,
        )
    )
    bb_rmsds = torch.sqrt(
        torch.mean(
            torch.sum((pos[..., ca_mask, :] - ref_pos[..., ca_mask, :]) ** 2, dim=-1),
            dim=-1,
        )
    )

    ca_n_vecs = pos[..., n_mask, :] - pos[..., ca_mask, :]
    ca_c_vecs = pos[..., c_mask, :] - pos[..., ca_mask, :]
    ca_n_vecs = ca_n_vecs / torch.linalg.norm(ca_n_vecs, dim=-1, keepdims=True)
    ca_c_vecs = ca_c_vecs / torch.linalg.norm(ca_c_vecs, dim=-1, keepdims=True)
    z_vecs = torch.cross(ca_n_vecs, ca_c_vecs, dim=-1)
    z_vecs = z_vecs / torch.linalg.norm(z_vecs, dim=-1, keepdims=True)

    ref_ca_n_vecs = ref_pos[..., n_mask, :] - ref_pos[..., ca_mask, :]
    ref_ca_c_vecs = ref_pos[..., c_mask, :] - ref_pos[..., ca_mask, :]
    ref_ca_n_vecs = ref_ca_n_vecs / torch.linalg.norm(
        ref_ca_n_vecs, dim=-1, keepdims=True
    )
    ref_ca_c_vecs = ref_ca_c_vecs / torch.linalg.norm(
        ref_ca_c_vecs, dim=-1, keepdims=True
    )
    ref_z_vecs = torch.cross(ref_ca_n_vecs, ref_ca_c_vecs, dim=-1)
    ref_z_vecs = ref_z_vecs / torch.linalg.norm(ref_z_vecs, dim=-1, keepdims=True)

    axis_z = torch.cross(z_vecs, ref_z_vecs.expand_as(z_vecs), dim=-1)
    axis_z = axis_z / torch.linalg.norm(axis_z, dim=-1, keepdims=True)
    angle_z = torch.arccos(
        torch.clamp(
            torch.sum(z_vecs * ref_z_vecs, dim=-1, keepdims=True),
            min=-1 + 1e-12,
            max=1 - 1e-12,
        )
    )
    rot_vec_z = axis_z * angle_z
    rot_mat_z = axis_angle_to_matrix(rot_vec_z)

    ca_n_vecs = (ca_n_vecs.unsqueeze(-2) @ rot_mat_z.swapaxes(-1, -2)).squeeze()
    axis_x = torch.cross(ca_n_vecs, ref_ca_n_vecs.expand_as(ca_n_vecs), dim=-1)
    axis_x = axis_x / torch.linalg.norm(axis_x, dim=-1, keepdims=True)
    angle_x = torch.arccos(
        torch.clamp(
            torch.sum(ca_n_vecs * ref_ca_n_vecs, dim=-1, keepdims=True),
            min=-1 + 1e-12,
            max=1 - 1e-12,
        )
    )
    rot_vec_x = axis_x * angle_x
    rot_mat_x = axis_angle_to_matrix(rot_vec_x)

    rot_mat = rot_mat_x @ rot_mat_z
    rot_vec = matrix_to_axis_angle(rot_mat)
    bb_rot_mags = torch.abs(torch.linalg.norm(rot_vec, dim=-1))
    bb_rot_mags = torch.min(bb_rot_mags, 2 * torch.pi - bb_rot_mags)

    metric_dict = {
        "aa_rmsds": aa_rmsds.cpu().numpy(),
        "bb_rmsds": bb_rmsds.cpu().numpy(),
        "bb_rot_mags": bb_rot_mags.cpu().numpy(),
    }
    return metric_dict


def compute_ligand_geometry_metrics(pos, ref_pos, mol):
    mol = Chem.RemoveHs(mol)

    bond_index = get_bond_index(mol).to(pos.device)
    angle_index = get_angle_index(mol).to(pos.device)
    torsion_index = get_torsion_index(mol).to(pos.device)
    ring_5_index, ring_6_index = get_ring_index(mol)
    ring_5_index = ring_5_index.to(pos.device)
    ring_6_index = ring_6_index.to(pos.device)

    bond_lengths = get_bond_lengths(pos, bond_index)
    ref_bond_lengths = get_bond_lengths(ref_pos, bond_index)
    bond_length_diffs = torch.mean(torch.abs(bond_lengths - ref_bond_lengths), axis=-1)

    bond_angles = get_bond_angles(pos, angle_index)
    ref_bond_angles = get_bond_angles(ref_pos, angle_index)
    bond_angle_diffs = torch.abs(bond_angles - ref_bond_angles)
    bond_angle_diffs = torch.mean(
        torch.min(bond_angle_diffs, 2 * torch.pi - bond_angle_diffs), axis=-1
    )

    torsion_angles = get_torsion_angles(pos, torsion_index)
    ref_torsion_angles = get_torsion_angles(ref_pos, torsion_index)
    torsion_angle_diffs = torch.abs(torsion_angles - ref_torsion_angles)
    torsion_angle_diffs = torch.mean(
        torch.min(torsion_angle_diffs, 2 * torch.pi - torsion_angle_diffs), axis=-1
    )

    metric_dict = {}
    metric_dict["lig_bond_length_diffs"] = bond_length_diffs.cpu().numpy()
    metric_dict["lig_bond_angle_diffs"] = bond_angle_diffs.cpu().numpy()
    metric_dict["lig_torsion_angle_diffs"] = torsion_angle_diffs.cpu().numpy()

    if ring_5_index.shape[0] > 0:
        ring_5_phi, ring_5_q = get_puckering_coordinates(pos, ring_5_index)
        ref_ring_5_phi, ref_ring_5_q = get_puckering_coordinates(ref_pos, ring_5_index)
        ring_5_phi_diffs = torch.mean(torch.abs(ring_5_phi - ref_ring_5_phi), axis=-1)
        ring_5_q_diffs = torch.mean(torch.abs(ring_5_q - ref_ring_5_q), axis=-1)
        metric_dict["ring_5_phi_diffs"] = ring_5_phi_diffs.cpu().numpy()
        metric_dict["ring_5_q_diffs"] = ring_5_q_diffs.cpu().numpy()

    if ring_6_index.shape[0] > 0:
        ring_6_phi, ring_6_theta, ring_6_q = get_puckering_coordinates(
            pos, ring_6_index, use_polar=True
        )
        ref_ring_6_phi, ref_ring_6_theta, ref_ring_6_q = get_puckering_coordinates(
            ref_pos, ring_6_index, use_polar=True
        )
        ring_6_phi_diffs = torch.mean(torch.abs(ring_6_phi - ref_ring_6_phi), axis=-1)
        ring_6_theta_diffs = torch.mean(
            torch.abs(ring_6_theta - ref_ring_6_theta), axis=-1
        )
        ring_6_q_diffs = torch.mean(torch.abs(ring_6_q - ref_ring_6_q), axis=-1)
        metric_dict["ring_6_phi_diffs"] = ring_6_phi_diffs.cpu().numpy()
        metric_dict["ring_6_theta_diffs"] = ring_6_theta_diffs.cpu().numpy()
        metric_dict["ring_6_q_diffs"] = ring_6_q_diffs.cpu().numpy()

    return metric_dict


def compute_lddt_pli(
    lig_pos,
    ref_lig_pos,
    atom_pos,
    ref_atom_pos,
    atom_rec_index,
    inclusion_radius,
    per_atom=False,
):
    contact_mask = get_nearby_residue_mask(
        ref_atom_pos,
        ref_lig_pos,
        atom_rec_index,
        cutoff=inclusion_radius,
        min_residues=0,
    )
    contacts = torch.argwhere(contact_mask).squeeze()
    dists = get_cross_dists(atom_pos, lig_pos, atom_rec_index).index_select(
        -1, contacts
    )
    ref_dists = get_cross_dists(ref_atom_pos, lig_pos, atom_rec_index).index_select(
        -1, contacts
    )
    dists_l1 = torch.abs(dists - ref_dists)
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dists_l1.device)
    reduce_axes = (-1,) if per_atom else (-2, -1)
    score = (dists_l1.unsqueeze(-1) < thresholds).float().mean(dim=-1).mean(reduce_axes)
    return {"lddt_pli": score.cpu().numpy()}


def compute_posebusters_geometry_metrics(
    lig_pos,
    edge_index,
    lower_bounds,
    upper_bounds,
    bond_mask,
    angle_mask,
    bond_buffer=0.25,
    angle_buffer=0.25,
    steric_buffer=0.2,
):
    lig_dists = torch.linalg.norm(
        lig_pos.index_select(-2, edge_index[1])
        - lig_pos.index_select(-2, edge_index[0]),
        dim=-1,
    )
    bond_checks = (lig_dists >= (1 - bond_buffer) * lower_bounds) * (
        lig_dists <= (1 + bond_buffer) * upper_bounds
    )
    angle_checks = (lig_dists >= (1 - angle_buffer) * lower_bounds) * (
        lig_dists <= (1 + angle_buffer) * upper_bounds
    )
    steric_checks = lig_dists >= (1 - steric_buffer) * lower_bounds
    bond_checks = bond_checks.index_select(-1, torch.argwhere(bond_mask).squeeze()).all(
        axis=-1
    )
    angle_checks = angle_checks.index_select(
        -1, torch.argwhere(angle_mask).squeeze()
    ).all(axis=-1)
    steric_checks = steric_checks.index_select(
        -1, torch.argwhere(~(bond_mask + angle_mask)).squeeze()
    ).all(axis=-1)
    return {
        "posebusters_bond_length": bond_checks.cpu().numpy(),
        "posebusters_bond_angle": angle_checks.cpu().numpy(),
        "posebusters_internal_steric_clash": steric_checks.cpu().numpy(),
    }


def compute_posebusters_interaction_metrics(
    lig_pos,
    atom_pos,
    lig_vdw_radii,
    atom_vdw_radii,
    overlap_radii_scale=1.0,
    volume_radii_scale=0.8,
    max_overlap=0.75,
    max_distance=5.0,
    max_volume_overlap=0.075,
):
    cross_dists = torch.linalg.norm(
        lig_pos.unsqueeze(-2) - atom_pos.unsqueeze(-3), dim=-1
    )
    vdw_overlaps = torch.clip(
        (lig_vdw_radii * overlap_radii_scale).view(
            *(1 for _ in lig_pos.shape[:-2]), -1, 1
        )
        + (atom_vdw_radii * overlap_radii_scale).view(
            *(1 for _ in atom_pos.shape[:-2]), 1, -1
        )
        - cross_dists,
        min=0,
    )
    # lig_bb_upper = torch.amax(lig_pos, dim=tuple(range(len(lig_pos.shape[:-1])))) + 2.5
    # lig_bb_lower = torch.amin(lig_pos, dim=tuple(range(len(lig_pos.shape[:-1])))) - 2.5
    # sampled_lig_points = torch.rand(size=(int(1e6), 3)) * (lig_bb_upper - lig_bb_lower) + lig_bb_lower
    # lig_point_fraction = (torch.linalg.norm(lig_pos.unsqueeze(-2) - sampled_lig_points, dim=-1) < (lig_vdw_radii * volume_radii_scale).unsqueeze(-1)).any(axis=-2).float().mean(axis=-1)
    # lig_volume = lig_point_fraction * torch.prod(lig_bb_upper - lig_bb_lower)
    # lig_overlap_idxs = torch.argwhere((vdw_overlaps > 0).sum(dim=(*(dim for dim in range(len(vdw_overlaps.shape[:-2]))),-1))).squeeze()
    # atom_overlap_idxs = torch.argwhere((vdw_overlaps > 0).sum(dim=(*(dim for dim in range(len(vdw_overlaps.shape[:-2]))),-2))).squeeze()
    # lig_overlap_pos = lig_pos.index_select(-2, lig_overlap_idxs)
    # atom_overlap_pos = atom_pos.index_select(-2, atom_overlap_idxs)
    # overlap_pos = torch.cat((lig_overlap_pos, atom_overlap_pos), dim=0)
    # overlap_bb_upper = torch.amax(overlap_pos, dim=tuple(range(len(overlap_pos.shape[:-1])))) + 2.5
    # overlap_bb_lower = torch.amin(overlap_pos, dim=tuple(range(len(overlap_pos.shape[:-1])))) - 2.5
    # sampled_overlap_points = torch.rand(size=(int(1e6), 3)) * (overlap_bb_upper - overlap_bb_lower) + overlap_bb_lower
    # overlap_point_fraction = (
    #     (torch.linalg.norm(lig_overlap_pos.unsqueeze(-2) - sampled_overlap_points, dim=-1) < (lig_vdw_radii[lig_overlap_idxs] * volume_radii_scale).unsqueeze(-1)).any(axis=-2) * \
    #     (torch.linalg.norm(atom_overlap_pos.unsqueeze(-2) - sampled_overlap_points, dim=-1) < (atom_vdw_radii[atom_overlap_idxs] * volume_radii_scale).unsqueeze(-1)).any(axis=-2)
    # ).float().mean(axis=-1)
    # overlap_volume = overlap_point_fraction * torch.prod(overlap_bb_upper - overlap_bb_lower)
    min_distance_checks = torch.amax(vdw_overlaps, dim=(-1, -2)) < max_overlap
    max_distance_checks = torch.amin(cross_dists, dim=(-1, -2)) < max_distance
    # max_volume_overlap_checks = overlap_volume / lig_volume < max_volume_overlap
    return {
        "posebusters_min_distance": min_distance_checks.cpu().numpy(),
        "posebusters_max_distance": max_distance_checks.cpu().numpy(),
        # 'posebusters_max_volume_overlap': max_volume_overlap_checks.cpu().numpy(),
    }


def construct_metric_dict(
    name,
    values,
    percentiles=None,
    upper_thresholds=None,
    lower_thresholds=None,
    rmsd_mask=None,
):
    metric_dict = {}
    for step_idx, step_values in enumerate(values):
        entry_name = name
        if step_idx == 0:
            entry_name = "initial_" + entry_name
        elif step_idx < values.shape[0] - 1:
            entry_name = f"step_{step_idx - 1}_" + entry_name
        else:
            entry_name = "final_" + entry_name
        if percentiles is not None:
            percentile_values = np.percentile(step_values, percentiles, axis=-1)
            for percentile, percentile_value in zip(percentiles, percentile_values):
                metric_dict[
                    f"{entry_name}_all_percentile_{percentile}"
                ] = percentile_value
            if rmsd_mask.sum() > 0:
                percentile_values = np.percentile(
                    step_values[rmsd_mask], percentiles, axis=-1
                )
                for percentile, percentile_value in zip(percentiles, percentile_values):
                    metric_dict[
                        f"{entry_name}_good_percentile_{percentile}"
                    ] = percentile_value
            if (~rmsd_mask).sum() > 0:
                percentile_values = np.percentile(
                    step_values[~rmsd_mask], percentiles, axis=-1
                )
                for percentile, percentile_value in zip(percentiles, percentile_values):
                    metric_dict[
                        f"{entry_name}_bad_percentile_{percentile}"
                    ] = percentile_value
        if upper_thresholds is not None:
            thresholded_fractions = np.mean(
                np.array(upper_thresholds).reshape(
                    -1, *(1 for dim in step_values.shape)
                )
                > step_values[None],
                axis=-1,
            )
            for threshold, thresholded_fraction in zip(
                upper_thresholds, thresholded_fractions
            ):
                metric_dict[
                    f"{entry_name}_all_fraction_below_{threshold}"
                ] = thresholded_fraction
            if rmsd_mask.sum() > 0:
                thresholded_fractions = np.mean(
                    np.array(upper_thresholds).reshape(
                        -1, *(1 for dim in step_values[rmsd_mask].shape)
                    )
                    > step_values[None, rmsd_mask],
                    axis=-1,
                )
                for threshold, thresholded_fraction in zip(
                    upper_thresholds, thresholded_fractions
                ):
                    metric_dict[
                        f"{entry_name}_good_fraction_below_{threshold}"
                    ] = thresholded_fraction
            if (~rmsd_mask).sum() > 0:
                thresholded_fractions = np.mean(
                    np.array(upper_thresholds).reshape(
                        -1, *(1 for dim in step_values[~rmsd_mask].shape)
                    )
                    > step_values[None, ~rmsd_mask],
                    axis=-1,
                )
                for threshold, thresholded_fraction in zip(
                    upper_thresholds, thresholded_fractions
                ):
                    metric_dict[
                        f"{entry_name}_bad_fraction_below_{threshold}"
                    ] = thresholded_fraction
        if lower_thresholds is not None:
            thresholded_fractions = np.mean(
                np.array(lower_thresholds).reshape(
                    -1, *(1 for dim in step_values.shape)
                )
                < step_values[None],
                axis=-1,
            )
            for threshold, thresholded_fraction in zip(
                lower_thresholds, thresholded_fractions
            ):
                metric_dict[
                    f"{entry_name}_all_fraction_above_{threshold}"
                ] = thresholded_fraction
            if rmsd_mask.sum() > 0:
                thresholded_fractions = np.mean(
                    np.array(lower_thresholds).reshape(
                        -1, *(1 for dim in step_values[rmsd_mask].shape)
                    )
                    < step_values[None, rmsd_mask],
                    axis=-1,
                )
                for threshold, thresholded_fraction in zip(
                    lower_thresholds, thresholded_fractions
                ):
                    metric_dict[
                        f"{entry_name}_good_fraction_above_{threshold}"
                    ] = thresholded_fraction
            if (~rmsd_mask).sum() > 0:
                thresholded_fractions = np.mean(
                    np.array(lower_thresholds).reshape(
                        -1, *(1 for dim in step_values[~rmsd_mask].shape)
                    )
                    < step_values[None, ~rmsd_mask],
                    axis=-1,
                )
                for threshold, thresholded_fraction in zip(
                    lower_thresholds, thresholded_fractions
                ):
                    metric_dict[
                        f"{entry_name}_bad_fraction_above_{threshold}"
                    ] = thresholded_fraction
    return metric_dict
