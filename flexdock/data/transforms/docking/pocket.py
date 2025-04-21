import torch
from torch_geometric.transforms import BaseTransform

from flexdock.data.feature.protein import get_binding_pocket_masks
from flexdock.geometry.ops import rigid_transform_kabsch


class PocketTransform(BaseTransform):
    def __init__(
        self,
        pocket_reduction: bool = False,
        pocket_radius: float = 5.0,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        holo_pocket_center: bool = False,
        use_origpos_pocket: bool = False,
        fast_updates: bool = False,
        all_atoms: bool = False,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
    ):
        self.pocket_reduction = pocket_reduction

        self.pocket_radius = pocket_radius
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size

        self.holo_pocket_center = holo_pocket_center
        self.use_origpos_pocket = use_origpos_pocket
        self.fast_updates = fast_updates

        self.all_atoms = all_atoms
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains

    def __call__(self, data):
        if not self.pocket_reduction:
            center = torch.mean(data["receptor"].pos, dim=0)
            data = self.center_complex(data, pocket_center=center)
            return data
        pocket_info = self.compute_pocket(data)
        pocket_selection_fn = (
            self.select_pocket if self.fast_updates else self.select_pocket_old
        )
        data = pocket_selection_fn(data, pocket_info)
        return data

    def compute_pocket(self, data):
        apo_rec_pos = data["atom"].orig_apo_pos
        holo_rec_pos = data["atom"].orig_holo_pos
        (
            pocket_center,
            res_pocket_mask,
            atom_pocket_mask,
            nearby_residues,
        ) = get_binding_pocket_masks(
            apo_rec_pos,
            holo_rec_pos if holo_rec_pos is not None else apo_rec_pos,
            data["ligand"].pos,  # TODO
            data["atom"].ca_mask,
            data["atom", "atom_rec_contact", "receptor"].edge_index[1],
            pocket_cutoff=5.0,
            pocket_min_size=1,
            pocket_buffer=20,
        )
        return pocket_center, res_pocket_mask, atom_pocket_mask, nearby_residues

    def select_pocket_old(self, data, pocket_info):
        pass

    def select_pocket(self, data, pocket_info):
        pocket_center, res_pocket_mask, atom_pocket_mask, nearby_residues = pocket_info
        amber_pocket_mask = ":" + ",".join(
            [
                str(idx + 1)
                for idx in torch.argwhere(res_pocket_mask).squeeze().numpy().tolist()
            ]
        )
        data.pocket_mask = amber_pocket_mask

        data["receptor"].nearby_residues = nearby_residues

        # Update atom numbering
        atom_numbering_old = torch.arange(data["atom"].pos.size(0))
        atom_numbering_old = atom_numbering_old[atom_pocket_mask]
        atom_numbering_new = torch.arange(atom_pocket_mask.sum())
        atom_numbering_dict = dict(
            zip(atom_numbering_old.numpy(), atom_numbering_new.numpy())
        )

        residue_numbering_old = torch.arange(data["receptor"].x.size(0))
        residue_numbering_old = residue_numbering_old[res_pocket_mask]
        residue_numbering_new = torch.arange(res_pocket_mask.sum())
        residue_numbering_dict = dict(
            zip(residue_numbering_old.numpy(), residue_numbering_new.numpy())
        )

        # Gather pocket residues and attributes
        data["receptor"].x = data["receptor"].x[res_pocket_mask]
        data["receptor"].pos = data["receptor"].pos[res_pocket_mask]
        data["receptor"].lens_receptors = data["receptor"].lens_receptors[
            res_pocket_mask
        ]
        data["receptor"].rot_vec = data["receptor"].rot_vec[res_pocket_mask]

        # Gather pocket atom attributes
        data["atom"].x = data["atom"].x[atom_pocket_mask]
        data["atom"].vdw_radii = data["atom"].vdw_radii[atom_pocket_mask]

        data["atom"].pos = data["atom"].pos[atom_pocket_mask]
        data["atom"].orig_apo_pos = data["atom"].orig_apo_pos[atom_pocket_mask]
        data["atom"].orig_aligned_apo_pos = data["atom"].orig_aligned_apo_pos[
            atom_pocket_mask
        ]
        data["atom"].orig_holo_pos = data["atom"].orig_holo_pos[atom_pocket_mask]
        data["atom"].pos_sc_matched = data["atom"].pos_sc_matched[atom_pocket_mask]

        data["atom"].ca_mask = data["atom"].ca_mask[atom_pocket_mask]
        data["atom"].c_mask = data["atom"].c_mask[atom_pocket_mask]
        data["atom"].n_mask = data["atom"].n_mask[atom_pocket_mask]

        # Gather edges between atoms in pocket
        atom_edge_index = data["atom", "atom_bond", "atom"].edge_index
        edges_in_pocket = (
            atom_pocket_mask[atom_edge_index[0]] & atom_pocket_mask[atom_edge_index[1]]
        )

        # Create new edge numbering (used in fragment_index)
        edges_order_old = torch.arange(atom_edge_index.size(1))
        edges_order_old = edges_order_old[edges_in_pocket]
        edges_order_new = torch.arange(edges_order_old.size(0))
        edge_numbering_dict = dict(
            zip(edges_order_old.numpy(), edges_order_new.numpy())
        )

        # Which edge rotates which atoms in topological sorted order
        atom_fragment_index = data["atom_bond", "atom"].atom_fragment_index
        fragment_old_edge_order, fragment_old_atom_idx = atom_fragment_index
        fragment_edge_pocket_mask = edges_in_pocket[fragment_old_edge_order]
        fragment_atom_pocket_mask = atom_pocket_mask[fragment_old_atom_idx]

        # Gather edges in pocket and renumber them
        fragment_edge_pocket = fragment_old_edge_order[fragment_edge_pocket_mask]
        fragment_edge_pocket.apply_(lambda x: edge_numbering_dict[x])

        # Gather atoms in pocket and renumber them
        fragment_atom_idx_pocket = fragment_old_atom_idx[fragment_atom_pocket_mask]
        fragment_atom_idx_pocket.apply_(lambda x: atom_numbering_dict[x])

        # Update to new fragment index
        atom_fragment_index_pocket = torch.stack(
            [fragment_edge_pocket, fragment_atom_idx_pocket], dim=0
        )
        data["atom_bond", "atom"].atom_fragment_index = atom_fragment_index_pocket

        # Update receptor edge index
        atom_idx, res_idx = data["atom", "atom_rec_contact", "receptor"].edge_index
        atoms_in_pocket = atom_idx[atom_pocket_mask]
        atom_res_idx_pocket = res_idx[atom_pocket_mask]

        atoms_in_pocket.apply_(lambda x: atom_numbering_dict[x])
        atom_res_idx_pocket.apply_(lambda x: residue_numbering_dict[x])
        data["atom", "atom_rec_contact", "receptor"].edge_index = torch.stack(
            [atoms_in_pocket, atom_res_idx_pocket], dim=0
        )

        # Update edge index and edge mask
        data["atom", "atom_bond", "atom"].edge_index = atom_edge_index[
            :, edges_in_pocket
        ]
        data["atom", "atom_bond", "atom"].edge_index.apply_(
            lambda x: atom_numbering_dict[x]
        )
        data["atom", "atom_bond", "atom"].edge_mask = data[
            "atom", "atom_bond", "atom"
        ].edge_mask[edges_in_pocket]
        data["atom", "atom_bond", "atom"].sc_conformer_match_rotations = data[
            "atom", "atom_bond", "atom"
        ].sc_conformer_match_rotations[edges_in_pocket]
        data["atom", "atom_bond", "atom"].squeeze_mask = data[
            "atom", "atom_bond", "atom"
        ].squeeze_mask[edges_in_pocket]
        data["atom", "atom_bond", "atom"].ring_sub_mask = data[
            "atom", "atom_bond", "atom"
        ].ring_sub_mask[edges_in_pocket]
        data["atom", "atom_bond", "atom"].ring_flip_mask = data[
            "atom", "atom_bond", "atom"
        ].ring_flip_mask[edges_in_pocket]

        res_ids_rotatable = data["atom", "atom_bond", "atom"].res_to_rotate[:, 0]
        res_ids_rotatable_pocket = res_ids_rotatable[res_pocket_mask[res_ids_rotatable]]
        res_ids_rotatable_pocket.apply_(lambda x: residue_numbering_dict[x])

        data["atom", "atom_bond", "atom"].res_to_rotate = torch.stack(
            [res_ids_rotatable_pocket, torch.arange(len(res_ids_rotatable_pocket))],
            dim=1,
        )

        # Realign the proteins
        ca_apo = data["atom"].orig_aligned_apo_pos[data["atom"].ca_mask]
        ca_holo = data["atom"].orig_holo_pos[data["atom"].ca_mask]

        R, tr = rigid_transform_kabsch(ca_apo, ca_holo)
        data["atom"].orig_aligned_apo_pos = (
            data["atom"].orig_aligned_apo_pos @ R.T
        ) + tr.unsqueeze(-2)
        data["receptor"].rot_vec = data["receptor"].rot_vec @ R.T  # TODO: Check this
        # pocket_center = data['atom'].orig_aligned_apo_pos[data['atom'].ca_mask].mean(dim=0, keepdims=True)

        # if pocket_center.ndim == 1:
        #    pocket_center = pocket_center[None, :]
        pocket_center = torch.zeros((1, 3), dtype=torch.float32)
        data = self.center_complex(data, pocket_center)
        return data

    def center_complex(self, data, pocket_center):
        data["receptor"].pos -= pocket_center

        if self.all_atoms:
            data["atom"].pos -= pocket_center

        if self.flexible_backbone or self.flexible_sidechains:
            data["atom"].orig_apo_pos -= pocket_center
            data["receptor"].orig_apo_pos = data["atom"].orig_apo_pos[
                data["atom"].ca_mask
            ]

            data["atom"].orig_holo_pos -= pocket_center
            data["receptor"].orig_holo_pos = data["atom"].orig_holo_pos[
                data["atom"].ca_mask
            ]

            data["atom"].orig_aligned_apo_pos -= pocket_center
            data["receptor"].orig_aligned_apo_pos = data["atom"].orig_aligned_apo_pos[
                data["atom"].ca_mask
            ]

        data["ligand"].pos -= pocket_center
        data.original_center = pocket_center
        return data


class UnbalancedTransform(BaseTransform):
    def __init__(self, match_max_rmsd: float = None, fast_updates: bool = False):
        self.match_max_rmsd = match_max_rmsd
        self.fast_updates = fast_updates

    def __call__(self, data):
        if self.match_max_rmsd is not None:
            if self.fast_updates:
                aligned_apo_pos = data["atom"].orig_aligned_apo_pos[
                    data["atom"].ca_mask
                ]
                holo_pos = data["atom"].orig_holo_pos[data["atom"].ca_mask]
            else:
                aligned_apo_pos = data["atom"].orig_aligned_apo_pos[data["atom"].calpha]
                holo_pos = data["atom"].orig_holo_pos[data["atom"].calpha]

            rmsd = torch.sqrt(
                torch.mean(torch.sum((holo_pos - aligned_apo_pos) ** 2, axis=1))
            ).item()
            if rmsd > self.match_max_rmsd:
                data.loss_weight = 0.0
                return data

        data.loss_weight = 1.0
        return data
