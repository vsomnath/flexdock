import networkx as nx
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.utils import to_networkx

from flexdock.geometry.ops import axis_angle_to_matrix, matrix_to_axis_angle


################################################################################
# Helpers for computing rotatable bonds
################################################################################


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    # get all edges
    edges = pyg_data["ligand", "ligand"].edge_index.T.numpy()
    # itereate over edges, skip every second, because graph is still directed here
    # e.g. [[0,3] , [3,0]]  -> skip second edge and make graph undirected
    for i in range(0, edges.shape[0], 2):
        # assure that consecutive edges in list belong to same bond
        assert edges[i, 0] == edges[i + 1, 1]

        # transform to undirected graph and delete current edge
        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        # graph still connected ?
        if not nx.is_connected(G2):
            # if not, get largest connected component
            comp = list(sorted(nx.connected_components(G2), key=len)[0])
            # more than 1 vertex in component ?
            if len(comp) > 1:
                # first vertex from current edge in largest connected component ?
                # -> rotate all vertices of the subgraph which does not contain the vertex of
                # edge i from index 0
                if edges[i, 0] in comp:
                    # if yes, rotate the other part of the molecule
                    to_rotate.append([])
                    to_rotate.append(comp)
                else:
                    # if no, rotate around
                    to_rotate.append(comp)
                    to_rotate.append([])
                continue
        # graph still connected, so no rotatable bond here
        to_rotate.append([])
        to_rotate.append([])
    # True for all edges that connect 2 otherwise unconnected structures
    mask_edges = np.asarray(
        [0 if len(comp) == 0 else 1 for comp in to_rotate], dtype=bool
    )
    # initialize rotation mask with false for all edges in mask_edges
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # iterate over all edges in G (directed graph with duplicate edges )
    for i in range(len(G.edges())):
        # is it an edge that connectes 2 otherwise unconnected sub-structures?
        if mask_edges[i]:
            # write all vertices that should be rotated when rotating around edge i
            # into mask_rotate
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def get_fragment_index(edge_index, base_idx=None, edge_ptr=0):
    fragment_index, squeeze_mask, mask_rotate, ring_sub_mask, ring_flip_mask = (
        [],
        [],
        [],
        [],
        [],
    )

    G = nx.DiGraph()
    for edge_idx, edge in enumerate(edge_index.T):
        G.add_edge(*edge, idx=edge_idx)
    edge_to_edge_idx = nx.get_edge_attributes(G, "idx")
    in_ring = set(sum(list(nx.simple_cycles(G.to_undirected())), []))

    edge_rotation_graph = nx.DiGraph()
    for edge_idx in range(0, edge_index.shape[1], 2):
        assert edge_index[0, edge_idx] == edge_index[1, edge_idx + 1]
        assert edge_index[1, edge_idx] == edge_index[0, edge_idx + 1]
        G2 = G.to_undirected()
        G2.remove_edge(*edge_index[:, edge_idx])
        if not nx.is_connected(G2):
            fragments = nx.connected_components(G2)
            fragments = list(
                sorted(
                    fragments,
                    key=lambda fragment: int(edge_index[0, edge_idx] in fragment),
                )
            )
            if (base_idx is not None and base_idx in fragments[1]) or (
                base_idx is None and len(fragments[0]) <= len(fragments[1])
            ):
                edge_rotation_graph.add_node(edge_idx)
                fragment_index.extend(
                    [[edge_idx, atom_idx] for atom_idx in fragments[0]]
                )
                squeeze_mask.extend([True, False])
                if len(fragments[0]) > 1:
                    mask_rotate.extend([True, False])
                else:
                    mask_rotate.extend([False, False])
                if edge_index[0, edge_idx] in in_ring:
                    ring_sub_mask.extend([True, False])
                    ring_flip_mask.extend([False, False])
                elif edge_index[1, edge_idx] in in_ring:
                    ring_sub_mask.extend([True, False])
                    ring_flip_mask.extend([True, False])
                else:
                    ring_sub_mask.extend([False, False])
                    ring_flip_mask.extend([False, False])
            else:
                edge_rotation_graph.add_node(edge_idx + 1)
                fragment_index.extend(
                    [[edge_idx + 1, atom_idx] for atom_idx in fragments[1]]
                )
                squeeze_mask.extend([False, True])
                if len(fragments[1]) > 1:
                    mask_rotate.extend([False, True])
                else:
                    mask_rotate.extend([False, False])
                if edge_index[1, edge_idx] in in_ring:
                    ring_sub_mask.extend([False, True])
                    ring_flip_mask.extend([False, False])
                elif edge_index[1, edge_idx] in in_ring:
                    ring_sub_mask.extend([False, True])
                    ring_flip_mask.extend([False, True])
                else:
                    ring_sub_mask.extend([False, False])
                    ring_flip_mask.extend([False, False])
        else:
            squeeze_mask.extend([False, False])
            mask_rotate.extend([False, False])
            ring_sub_mask.extend([False, False])
            ring_flip_mask.extend([False, False])

    for edge_idx, atom_idx in fragment_index:
        rotated_edges = list(G.in_edges(atom_idx)) + list(G.out_edges(atom_idx))
        for rotated_edge in rotated_edges:
            rotated_edge_idx = edge_to_edge_idx[rotated_edge]
            if rotated_edge_idx != edge_idx and rotated_edge_idx in edge_rotation_graph:
                edge_rotation_graph.add_edge(edge_idx, rotated_edge_idx)
    edge_rotation_order = list(nx.topological_sort(edge_rotation_graph))[::-1]
    fragment_index.sort(key=lambda x: edge_rotation_order.index(x[0]))
    fragment_index.sort(key=lambda x: x[1])

    angle_2_index = []
    for atom_idx in G.nodes:
        if atom_idx not in in_ring and G.degree[atom_idx] == 4:
            angle_edge_idxs = [edge_to_edge_idx[edge] for edge in G.out_edges(atom_idx)]
            if squeeze_mask[angle_edge_idxs[0]]:
                angle_2_index.append(angle_edge_idxs)
            else:
                angle_2_index.append(angle_edge_idxs[::-1])

    squeeze_mask = np.array(squeeze_mask, dtype=bool)
    mask_rotate = np.array(mask_rotate, dtype=bool)
    ring_sub_mask = np.array(ring_sub_mask, dtype=bool)
    ring_flip_mask = np.array(ring_flip_mask, dtype=bool)
    fragment_index = (
        np.empty((2, 0), dtype=np.int64)
        if len(fragment_index) == 0
        else np.array(fragment_index, dtype=np.int64).T + np.array([[edge_ptr], [0]])
    )
    angle_2_index = (
        np.empty((2, 0), dtype=np.int64)
        if len(angle_2_index) == 0
        else np.array(angle_2_index, dtype=np.int64).T + edge_ptr
    )

    return (
        squeeze_mask,
        mask_rotate,
        ring_sub_mask,
        ring_flip_mask,
        fragment_index,
        angle_2_index,
    )


def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def align_sidechains_to_backbone_torch(
    apo_rec_pos, holo_rec_pos, ca_mask, c_mask, n_mask, atom_rec_index
):
    ca_n_vecs = apo_rec_pos[..., n_mask, :] - apo_rec_pos[..., ca_mask, :]
    ca_c_vecs = apo_rec_pos[..., c_mask, :] - apo_rec_pos[..., ca_mask, :]
    ca_n_vecs = ca_n_vecs / torch.linalg.norm(ca_n_vecs, dim=-1, keepdims=True)
    ca_c_vecs = ca_c_vecs / torch.linalg.norm(ca_c_vecs, dim=-1, keepdims=True)
    z_vecs = torch.cross(ca_n_vecs, ca_c_vecs, dim=-1)
    z_vecs = z_vecs / torch.linalg.norm(z_vecs, dim=-1, keepdims=True)

    ref_ca_n_vecs = holo_rec_pos[..., n_mask, :] - holo_rec_pos[..., ca_mask, :]
    ref_ca_c_vecs = holo_rec_pos[..., c_mask, :] - holo_rec_pos[..., ca_mask, :]
    ref_ca_n_vecs = ref_ca_n_vecs / torch.linalg.norm(
        ref_ca_n_vecs, dim=-1, keepdims=True
    )
    ref_ca_c_vecs = ref_ca_c_vecs / torch.linalg.norm(
        ref_ca_c_vecs, dim=-1, keepdims=True
    )
    ref_z_vecs = torch.cross(ref_ca_n_vecs, ref_ca_c_vecs, dim=-1)
    ref_z_vecs = ref_z_vecs / torch.linalg.norm(ref_z_vecs, dim=-1, keepdims=True)

    axis_z = torch.cross(z_vecs, ref_z_vecs, dim=-1)
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
    axis_x = torch.cross(ca_n_vecs, ref_ca_n_vecs, dim=-1)
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

    apo_rec_pos = (
        (
            apo_rec_pos - apo_rec_pos[..., ca_mask, :].index_select(-2, atom_rec_index)
        ).unsqueeze(-2)
        @ rot_mat.index_select(-3, atom_rec_index).swapaxes(-1, -2)
        + holo_rec_pos[ca_mask].index_select(-2, atom_rec_index).unsqueeze(-2)
    ).squeeze()
    return apo_rec_pos, rot_vec


def align_sidechains_to_backbone_numpy(apo_atoms, holo_atoms, lens_receptors):
    x = np.concatenate(
        [np.zeros(l, dtype="int") + i for i, l in enumerate(lens_receptors)]
    )
    y = np.concatenate([np.arange(l) for l in lens_receptors])
    apo_atom_grid = np.zeros(
        (len(lens_receptors), max(lens_receptors), 3)
    )  # (n_residues, max_atoms, 3)
    apo_atom_grid[x, y] = apo_atoms
    holo_atom_grid = np.zeros(
        (len(lens_receptors), max(lens_receptors), 3)
    )  # (n_residues, max_atoms, 3)
    holo_atom_grid[x, y] = holo_atoms

    # import pdb; pdb.set_trace()
    t_holo = holo_atom_grid[:, 1:2]  # calpha atom of holo
    t_apo = apo_atom_grid[:, 1:2]  # calpha atom of apo
    # move calpha atom of holo and apo to zero
    holo_atom_grid = holo_atom_grid - t_holo
    apo_atom_grid = apo_atom_grid - t_apo

    # import pdb; pdb.set_trace()
    # Aligning the N-Calpha-C planes
    z_holo = np.cross(holo_atom_grid[:, 0:1], holo_atom_grid[:, 2:3])
    z_holo = z_holo / np.linalg.norm(z_holo, axis=-1, keepdims=True)
    z_apo = np.cross(apo_atom_grid[:, 0:1], apo_atom_grid[:, 2:3])
    z_apo = z_apo / np.linalg.norm(z_apo, axis=-1, keepdims=True)
    axis_z = np.cross(z_apo, z_holo)
    axis_z = axis_z / np.linalg.norm(axis_z, axis=-1, keepdims=True)
    angle_z = np.arccos(np.sum(z_apo * z_holo, axis=-1, keepdims=True))
    rot_z = Rotation.from_rotvec((axis_z * angle_z)[:, 0])

    # import pdb; pdb.set_trace()
    # Aligning the N-Calpha vectors
    x_apo_rotated = rot_z.apply(apo_atom_grid[:, 0])[:, None, :]
    x_apo_rotated = x_apo_rotated / np.linalg.norm(
        x_apo_rotated, axis=-1, keepdims=True
    )
    x_holo = holo_atom_grid[:, 0:1] + 0
    x_holo = x_holo / np.linalg.norm(x_holo, axis=-1, keepdims=True)
    axis_x = np.cross(x_apo_rotated, x_holo)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    x_cos = np.sum(x_apo_rotated * x_holo, axis=-1, keepdims=True)
    angle_x = np.where(np.isclose(x_cos, 1), 0, np.arccos(x_cos))
    rot_x = Rotation.from_rotvec((axis_x * angle_x)[:, 0])
    rot = rot_x * rot_z
    # import pdb; pdb.set_trace()
    rot_mat = rot.as_matrix()
    rot_vec = rot.as_rotvec()
    apo_atom_grid = np.matmul(apo_atom_grid, np.transpose(rot_mat, (0, 2, 1)))
    # import pdb; pdb.set_trace()
    # substituting in the backbone coordinates, TODO not sure about this
    # apo_atom_grid[:, :3] = holo_atom_grid[:, :3]

    apo_atom_grid = apo_atom_grid + t_holo
    apo_atoms = apo_atom_grid[x, y]

    bb = np.zeros((apo_atom_grid.shape[0], apo_atom_grid.shape[1]), dtype="bool")
    bb[:, 1] = True
    calpha = bb[x, y]
    calpha_coords = apo_atom_grid[:, 1]
    # import pdb; pdb.set_trace()
    return apo_atoms, calpha, calpha_coords, rot_vec


def get_calpha_mask(lens_receptors):
    x = np.concatenate(
        [np.zeros(l, dtype="int") + i for i, l in enumerate(lens_receptors)]
    )
    y = np.concatenate([np.arange(l) for l in lens_receptors])

    bb = np.zeros((len(lens_receptors), max(lens_receptors)), dtype="bool")
    bb[:, 1] = True
    calpha = bb[x, y]
    return calpha


def to_atom_grid_torch(atoms, lens_receptors):
    x = torch.cat([torch.zeros(l).long() + i for i, l in enumerate(lens_receptors)]).to(
        atoms.device
    )
    y = torch.cat([torch.arange(l) for l in lens_receptors]).to(atoms.device)
    atom_grid = (
        torch.zeros((len(lens_receptors), max(lens_receptors), 3)).to(atoms.device)
        + 1000000
    )
    atom_grid[x, y] = atoms
    return atom_grid, x, y


def rotate_backbone_numpy(atoms, t_vec, rot_vec, lens_receptors):
    x = np.concatenate(
        [np.zeros(l, dtype="int") + i for i, l in enumerate(lens_receptors)]
    )
    y = np.concatenate([np.arange(l) for l in lens_receptors])
    atom_grid = np.zeros((len(lens_receptors), max(lens_receptors), 3))
    atom_grid[x, y] = atoms

    t_before = atom_grid[:, 1:2]
    atom_grid = atom_grid - t_before

    rot = Rotation.from_rotvec(rot_vec)
    rot_mat = rot.as_matrix()
    atom_grid = np.matmul(atom_grid, np.transpose(rot_mat, (0, 2, 1)))

    atom_grid = atom_grid + t_vec[:, None, :] + t_before
    atoms = atom_grid[x, y]
    return atoms


def rotate_backbone_torch(
    atoms, t_vec, rot_mat, lens_receptors, total_rot=None, detach: bool = False
):
    x = torch.cat([torch.zeros(l).long() + i for i, l in enumerate(lens_receptors)]).to(
        atoms.device
    )
    y = torch.cat([torch.arange(l) for l in lens_receptors]).to(atoms.device)
    atom_grid = torch.zeros((len(lens_receptors), max(lens_receptors), 3)).to(
        atoms.device
    )
    atom_grid[x, y] = atoms

    t_before = atom_grid[:, 1:2]
    atom_grid = atom_grid - t_before
    atom_grid = torch.bmm(
        atom_grid, torch.transpose(rot_mat.detach() if detach else rot_mat, 2, 1)
    )  # detach is for stop gradient that is also in AlphaFold

    atom_grid = atom_grid + t_vec[:, None, :] + t_before
    atoms = atom_grid[x, y]

    if total_rot is not None:
        rot_mat = torch.bmm(rot_mat, total_rot)
    return atoms, rot_mat


def construct_frames(atom_pos: torch.Tensor, lens_receptors: torch.Tensor):
    """
    Frame construction based on the algorithm adopted in AlphaFold-2.

    These frames are only used in combination with a IPA layer to predict
    updates to rotations and translations

    """
    x = torch.cat(
        [atom_pos.new_zeros(l).long() + i for i, l in enumerate(lens_receptors)]
    )
    y = torch.cat([torch.arange(l, device=atom_pos.device) for l in lens_receptors])

    atom_grid = atom_pos.new_zeros((len(lens_receptors), max(lens_receptors), 3))
    atom_grid[x, y] = atom_pos

    # [n_residues, 1, 3]
    v1 = atom_grid[:, 2:3] - atom_grid[:, 1:2]
    v2 = atom_grid[:, 0:1] - atom_grid[:, 1:2]
    e1 = v1 / torch.linalg.vector_norm(v1, dim=-1, keepdims=True)

    # [n_residues, 1, 3]
    u2 = v2 - e1 * (e1 * v2).sum(dim=-1, keepdims=True)
    e2 = u2 / torch.linalg.vector_norm(u2, dim=-1, keepdims=True)

    # [n_residues, 1, 3]
    e3 = torch.linalg.cross(e1, e2, dim=-1)

    # [n_residues, 3, 3]
    rot_frames = torch.cat([e1, e2, e3], dim=1)
    # [n_residues, 3]
    tr_frames = atom_grid[:, 1:2].squeeze(1)

    return rot_frames, tr_frames


def filter_flexible_residues(complex_graph, atoms_filter):
    old_subcomponentsMerged = complex_graph["flexResidues"].subcomponents
    old_subcomponentsMapping = complex_graph["flexResidues"].subcomponentsMapping
    old_edge_idx_merged = complex_graph["flexResidues"].edge_idx
    old_residueNBondsMapping = complex_graph["flexResidues"].residueNBondsMapping
    old_pdbIds = complex_graph["flexResidues"].pdbIds
    old_conf_rot = complex_graph.sc_conformer_match_rotations

    subcomponentsMerged = []
    subcomponentsMapping = []
    edge_idx_merged = []
    residueNBondsMapping = []
    pdbIds = []
    conf_rot = []

    component_idx = 0
    bond_idx = 0

    for i, pdbId in enumerate(old_pdbIds):
        if atoms_filter[old_edge_idx_merged[bond_idx, 1]]:
            pdbIds.append(pdbId)
            conf_rot.append(old_conf_rot[i])
            n_bonds = old_residueNBondsMapping[i]
            residueNBondsMapping.append(n_bonds)
            for j in range(n_bonds):
                edge_idx_merged.append(
                    [old_edge_idx_merged[bond_idx][0], old_edge_idx_merged[bond_idx][1]]
                )
                old_comp0, old_comp1 = (
                    old_subcomponentsMapping[bond_idx][0].item(),
                    old_subcomponentsMapping[bond_idx][1].item(),
                )
                subcomponentsMerged.append(old_subcomponentsMerged[old_comp0:old_comp1])
                subcomponentsMapping.append(
                    [component_idx, component_idx + old_comp1 - old_comp0]
                )
                component_idx += old_comp1 - old_comp0
                bond_idx += 1
        else:
            for j in range(old_residueNBondsMapping[i]):
                bond_idx += 1

    subcomponentsMerged = torch.cat(subcomponentsMerged, dim=0)
    subcomponentsMapping = torch.tensor(
        np.asarray(subcomponentsMapping), dtype=torch.long
    )
    edge_idx_merged = torch.tensor(np.asarray(edge_idx_merged), dtype=torch.long)
    residueNBondsMapping = torch.tensor(
        np.asarray(residueNBondsMapping), dtype=torch.long
    )

    complex_graph["flexResidues"].subcomponents = subcomponentsMerged
    complex_graph["flexResidues"].subcomponentsMapping = subcomponentsMapping
    complex_graph["flexResidues"].edge_idx = edge_idx_merged
    complex_graph["flexResidues"].residueNBondsMapping = residueNBondsMapping
    complex_graph["flexResidues"].pdbIds = pdbIds
    complex_graph.sc_conformer_match_rotations = conf_rot
    complex_graph["flexResidues"].num_nodes = complex_graph[
        "flexResidues"
    ].edge_idx.shape[0]

    return
