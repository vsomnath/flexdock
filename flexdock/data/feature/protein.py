import re

import numpy as np
import torch

from rdkit import Chem

from flexdock.data.constants import allowable_features, periodic_table
from flexdock.data.feature.helpers import safe_index, get_fragment_index

################################################################################
# Graph Generation
################################################################################


def filter_side_chain_atoms(atom):
    # ignores the O-H, OXT and NH2 group and drops the H atoms
    # re returns no match if we should keep the atom for the
    # side chain torsion graph
    return re.search("^(OXT)$|^C$|^O$|^N$|^H|^H$.|^H.$[1-9]", atom) is None


def construct_sidechain_edge_index(res):
    orderDict = {"A": "B", "B": "G", "G": "D", "D": "E", "E": "Z", "Z": "H", "H": ""}
    edge_index = []
    for i in range(len(res.atoms)):
        for j in range(i + 1, len(res.atoms)):
            atom_i, atom_j = res.atoms[i], res.atoms[j]
            if not (
                filter_side_chain_atoms(atom_i.name)
                and filter_side_chain_atoms(atom_j.name)
            ):
                continue
            elif (atom_i.name, atom_j.name) in {
                ("CE1", "NE2"),
                ("NE1", "CE2"),
                ("CD2", "CE3"),
                ("CZ3", "CH2"),
            }:
                edge_index.extend([[atom_i.idx, atom_j.idx], [atom_j.idx, atom_i.idx]])
            elif len(atom_i.name) == len(atom_j.name) == 3:
                if (
                    orderDict[atom_i.name[1]] == atom_j.name[1]
                    and atom_i.name[2] == atom_j.name[2]
                ):
                    edge_index.extend(
                        [[atom_i.idx, atom_j.idx], [atom_j.idx, atom_i.idx]]
                    )
            elif orderDict[atom_i.name[1]] == atom_j.name[1]:
                edge_index.extend([[atom_i.idx, atom_j.idx], [atom_j.idx, atom_i.idx]])
    return np.array(edge_index).T


def construct_backbone_edge_index(struct):
    edge_index = []
    for res_idx in range(len(struct.residues) - 1):
        res = struct.residues[res_idx]
        res_next = struct.residues[res_idx + 1]
        # Residue Backbone
        edge_index.extend(
            [
                [res.atoms[0].idx, res.atoms[1].idx],  # N-CA
                [res.atoms[1].idx, res.atoms[0].idx],  # CA-N
                [res.atoms[1].idx, res.atoms[2].idx],  # CA-C
                [res.atoms[2].idx, res.atoms[1].idx],  # C-CA
                [res.atoms[2].idx, res.atoms[3].idx],  # C-O
                [res.atoms[3].idx, res.atoms[2].idx],  # O-C
            ]
        )
        # Handle Carboxyl Terminus
        if res.atoms[-1].name == "OXT":
            edge_index.extend(
                [
                    [res.atoms[2].idx, res.atoms[-1].idx],  # N-CD
                    [res.atoms[-1].idx, res.atoms[2].idx],  # CD-N
                ]
            )
        # Handle Proline
        if res.name == "PRO":
            edge_index.extend(
                [
                    [res.atoms[0].idx, res.atoms[6].idx],  # N-CD
                    [res.atoms[6].idx, res.atoms[0].idx],  # CD-N
                ]
            )
        # Connect Sequential Residues in the Same Chain
        if res.chain == res_next.chain:
            edge_index.extend(
                [
                    [res.atoms[2].idx, res_next.atoms[0].idx],  # C-N
                    [res_next.atoms[0].idx, res.atoms[2].idx],  # N-C
                ]
            )

    # Handle Disulfide Bridges
    for bond in struct.bonds:
        if bond.atom1.name == bond.atom2.name == "S":
            print(bond)
            edge_index.extend(
                [[bond.atom1.idx, bond.atom2.idx], [bond.atom2.idx, bond.atom1.idx]]
            )
    return np.array(edge_index).T


def construct_protein_edge_index(struct):
    (
        edge_index,
        squeeze_mask,
        edge_mask,
        ring_sub_mask,
        ring_flip_mask,
        fragment_index,
        angle_2_index,
    ) = (np.empty((2, 0), dtype=np.int64), [], [], [], [], [], [])
    res_to_rotate_mask = []

    start = 0
    for res_idx, res in enumerate(struct.residues):
        if res.name == "GLY":
            continue
        res_edge_index = construct_sidechain_edge_index(res)
        if res.name == "PRO":
            squeeze_mask.append(np.zeros(res_edge_index.shape[1], dtype=bool))
            res_to_rotate_mask.extend(
                [[res_idx, start + i] for i in range(res_edge_index.shape[1])]
            )
            edge_mask.append(np.zeros(res_edge_index.shape[1], dtype=bool))
            ring_sub_mask.append(np.zeros(res_edge_index.shape[1], dtype=bool))
            ring_flip_mask.append(np.zeros(res_edge_index.shape[1], dtype=bool))
        else:
            (
                res_squeeze_mask,
                res_rotate_mask,
                res_ring_sub_mask,
                res_ring_flip_mask,
                res_fragment_index,
                res_angle_2_index,
            ) = get_fragment_index(
                res_edge_index, base_idx=res.atoms[1].idx, edge_ptr=edge_index.shape[1]
            )
            fragment_index.append(res_fragment_index)
            angle_2_index.append(res_angle_2_index)
            squeeze_mask.append(res_squeeze_mask)
            edge_mask.append(res_rotate_mask)
            res_to_rotate_mask.extend(
                [[res_idx, start + i] for i in range(res_rotate_mask.shape[0])]
            )
            ring_sub_mask.append(res_ring_sub_mask)
            ring_flip_mask.append(res_ring_flip_mask)

        edge_index = np.concatenate((edge_index, res_edge_index), axis=1)
        start += res_edge_index.shape[1]
        # edge_index.append(res_edge_index)

    # backbond_edge_index = construct_backbone_edge_index(struct)
    # start += backbond_edge_index.shape[1]
    # edge_index = np.concatenate((edge_index, backbond_edge_index), axis=1)
    # # edge_index.append(backbond_edge_index)
    # squeeze_mask.append(np.zeros(backbond_edge_index.shape[1], dtype=bool))
    # edge_mask.append(np.zeros(backbond_edge_index.shape[1], dtype=bool))
    # ring_sub_mask.append(np.zeros(backbond_edge_index.shape[1], dtype=bool))
    # ring_flip_mask.append(np.zeros(backbond_edge_index.shape[1], dtype=bool))

    # edge_index = np.concatenate(edge_index, axis=1)
    squeeze_mask = np.concatenate(squeeze_mask, axis=0)
    edge_mask = np.concatenate(edge_mask, axis=0)
    res_to_rotate_mask = np.array(res_to_rotate_mask)
    ring_sub_mask = np.concatenate(ring_sub_mask, axis=0)
    ring_flip_mask = np.concatenate(ring_flip_mask, axis=0)
    fragment_index = np.concatenate(fragment_index, axis=1)
    angle_2_index = np.concatenate(angle_2_index, axis=1)
    return (
        edge_index,
        squeeze_mask,
        edge_mask,
        ring_sub_mask,
        ring_flip_mask,
        fragment_index,
        angle_2_index,
        res_to_rotate_mask,
    )


def get_rec_graph(
    rec,
    rec_coords,
    c_alpha_coords,
    complex_graph,
    rec_radius,
    c_alpha_max_neighbors=None,
    all_atoms=False,
    atom_radius=5,
    atom_max_neighbors=None,
    remove_hs=False,
    lm_embeddings=None,
    knn_only_graph=False,
    fixed_knn_radius_graph=False,
):
    if all_atoms:
        return get_fullrec_graph(
            rec,
            rec_coords,
            c_alpha_coords,
            complex_graph,
            c_alpha_cutoff=rec_radius,
            c_alpha_max_neighbors=c_alpha_max_neighbors,
            atom_cutoff=atom_radius,
            atom_max_neighbors=atom_max_neighbors,
            remove_hs=remove_hs,
            lm_embeddings=lm_embeddings,
            knn_only_graph=knn_only_graph,
            fixed_knn_radius_graph=fixed_knn_radius_graph,
        )
    else:
        raise Exception("Not implemented yet")


def get_fullrec_graph(struct, complex_graph, lm_embeddings=None):
    # builds the receptor graph with both residues and atoms
    node_feat = rec_residue_featurizer(struct)
    atom_feat = rec_atom_featurizer(struct)

    complex_graph["receptor"].x = (
        torch.cat([node_feat, lm_embeddings], axis=1)
        if lm_embeddings is not None
        else node_feat
    )
    complex_graph["atom"].x = atom_feat
    complex_graph["atom"].vdw_radii = torch.FloatTensor(
        [periodic_table.GetRvdw(int(atom.atomic_number)) for atom in struct.atoms]
    )
    complex_graph["atom"].ca_mask = torch.tensor(
        [atom.name == "CA" for atom in struct.atoms]
    )
    complex_graph["atom"].c_mask = torch.tensor(
        [atom.name == "C" for atom in struct.atoms]
    )
    complex_graph["atom"].n_mask = torch.tensor(
        [atom.name == "N" for atom in struct.atoms]
    )
    complex_graph["atom", "atom_rec_contact", "receptor"].edge_index = torch.tensor(
        [[atom.idx, atom.residue.idx] for atom in struct.atoms], dtype=torch.int64
    ).T
    return


def get_protein_vdw_radii(complex_graph):
    complex_graph["atom"].vdw_radii = torch.FloatTensor(
        [
            periodic_table.GetRvdw(Chem.GetPeriodicTable(), atomic_number)
            for atomic_number in complex_graph["atom"].x[:, 1].numpy().tolist()
        ]
    )
    return complex_graph


# ==============================================================================
# Featurizing routines
# ==============================================================================


def rec_residue_featurizer(struct):
    feature_list = []
    # sr.compute(rec, level="R")
    for residue in struct.residues:
        """sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)"""
        feature_list.append(
            [
                safe_index(allowable_features["possible_amino_acids"], residue.name),
                # sasa, bfactor
            ]
        )
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def get_rec_atom_feat(atom=None, atom_name=None, element=None, get_misc_features=False):
    if get_misc_features:
        return [
            safe_index(allowable_features["possible_amino_acids"], "misc"),
            safe_index(allowable_features["possible_atomic_num_list"], "misc"),
            safe_index(allowable_features["possible_atom_type_2"], "misc"),
            safe_index(allowable_features["possible_atom_type_3"], "misc"),
        ]
    if atom_name is not None:
        atom_name = atom_name
    else:
        atom_name = atom.name
    atomic_num = atom.atomic_number

    atom_feat = [
        safe_index(allowable_features["possible_amino_acids"], atom.residue.name),
        safe_index(allowable_features["possible_atomic_num_list"], atomic_num),
        safe_index(allowable_features["possible_atom_type_2"], (atom_name + "*")[:2]),
        safe_index(allowable_features["possible_atom_type_3"], atom_name),
    ]
    return atom_feat


def rec_atom_featurizer(struct):
    atom_feats = []
    for i, atom in enumerate(struct.atoms):
        atom_feats.append(get_rec_atom_feat(atom))
    return torch.tensor(atom_feats, dtype=torch.float32)


def get_rec_misc_atom_feat(
    bio_atom=None, atom_name=None, element=None, get_misc_features=False
):
    if get_misc_features:
        return [
            safe_index(allowable_features["possible_amino_acids"], "misc"),
            safe_index(allowable_features["possible_atomic_num_list"], "misc"),
            safe_index(allowable_features["possible_atom_type_2"], "misc"),
            safe_index(allowable_features["possible_atom_type_3"], "misc"),
        ]
    if atom_name is not None:
        atom_name = atom_name
    else:
        atom_name = bio_atom.name
    if element is not None:
        element = element
    else:
        element = bio_atom.element
    if element == "CD":
        element = "C"
    assert not element == ""
    try:
        atomic_num = periodic_table.GetAtomicNumber(element.lower().capitalize())
    except Exception:
        atomic_num = -1

    atom_feat = [
        safe_index(
            allowable_features["possible_amino_acids"],
            bio_atom.get_parent().get_resname(),
        ),
        safe_index(allowable_features["possible_atomic_num_list"], atomic_num),
        safe_index(allowable_features["possible_atom_type_2"], (atom_name + "*")[:2]),
        safe_index(allowable_features["possible_atom_type_3"], atom_name),
    ]
    return atom_feat


################################################################################
# Binding pocket extraction
################################################################################


def get_nearby_residue_mask(
    atom_pos, lig_pos, atom_rec_index, cutoff=5.0, min_residues=0
):
    dists = torch.linalg.norm(atom_pos.unsqueeze(-2) - lig_pos.unsqueeze(-3), dim=-1)
    min_atom_dists = torch.min(dists, dim=-1).values
    min_rec_dists = torch.scatter_reduce(
        torch.full(
            (*min_atom_dists.shape[:-1], atom_rec_index.max() + 1),
            torch.inf,
            dtype=min_atom_dists.dtype,
            device=min_atom_dists.device,
        ),
        dim=-1,
        src=min_atom_dists,
        index=atom_rec_index.expand(min_atom_dists.shape),
        reduce="amin",
        include_self=False,
    )
    nearby_residue_mask = min_rec_dists < cutoff
    if nearby_residue_mask.sum() < min_residues:
        nearby_residue_mask = torch.zeros_like(min_rec_dists).bool()
        _, closest_residues = torch.topk(min_rec_dists, k=min_residues, largest=False)
        nearby_residue_mask.index_fill_(-1, closest_residues, True)
    return nearby_residue_mask


def get_nearby_atom_mask(atom_pos, lig_pos, atom_rec_index, cutoff=5.0, min_residues=0):
    nearby_residue_mask = get_nearby_residue_mask(
        atom_pos, lig_pos, atom_rec_index, cutoff=cutoff, min_residues=min_residues
    )
    nearby_atom_mask = torch.index_select(nearby_residue_mask, -1, atom_rec_index)
    return nearby_atom_mask


def get_binding_pocket_masks(
    atom_pos,
    ref_atom_pos,
    lig_pos,
    ca_mask,
    atom_rec_index,
    pocket_cutoff=5.0,
    pocket_buffer=20.0,
    pocket_min_size=0,
):
    nearby_residue_mask = get_nearby_residue_mask(
        ref_atom_pos,
        lig_pos,
        atom_rec_index,
        cutoff=pocket_cutoff,
        min_residues=pocket_min_size,
    )
    nearby_residue_idxs = torch.argwhere(nearby_residue_mask).squeeze()
    ca_atom_idxs = torch.argwhere(ca_mask).squeeze()
    ca_pos = torch.index_select(atom_pos, -2, ca_atom_idxs)
    pocket_center = torch.index_select(ca_pos, -2, nearby_residue_idxs).mean(dim=-2)
    pocket_res_mask = torch.linalg.norm(ca_pos - pocket_center, dim=-1) < pocket_buffer
    pocket_atom_mask = torch.index_select(pocket_res_mask, -1, atom_rec_index)
    return pocket_center, pocket_res_mask, pocket_atom_mask, nearby_residue_idxs
