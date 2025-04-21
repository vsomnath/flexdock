from torch_geometric.transforms import BaseTransform
import torch

from flexdock.data.conformers.modify import modify_conformer
from flexdock.geometry.ops import (
    axis_angle_to_matrix,
)
from flexdock.geometry.manifolds import so3


class ProteinNoiseTransform(BaseTransform):
    def __init__(
        self,
        bb_tr_sigma=None,
        bb_rot_sigma=None,
        sidechain_bond_sigma=None,
        sidechain_angle_sigma=None,
        sidechain_torsion_sigma=None,
        sidechain_fragment_sigma=None,
    ):
        self.bb_tr_sigma = bb_tr_sigma
        self.bb_rot_sigma = bb_rot_sigma
        self.sidechain_bond_sigma = sidechain_bond_sigma
        self.sidechain_angle_sigma = sidechain_angle_sigma
        self.sidechain_torsion_sigma = sidechain_torsion_sigma
        self.sidechain_fragment_sigma = sidechain_fragment_sigma
        self.rigid_sidechains = all(
            sigma is None
            for sigma in (
                sidechain_bond_sigma,
                sidechain_angle_sigma,
                sidechain_torsion_sigma,
                sidechain_fragment_sigma,
            )
        )

    def __call__(self, data):
        n_conf = data["atom"].flexdock_pos.shape[0]
        n_res = data["receptor"].num_nodes
        n_sidechain_bonds = data["atom", "atom_bond", "atom"].squeeze_mask.sum()
        n_sidechain_angles = (
            0
            if len(data["atom_bond", "atom_angle", "atom_bond"].angle_2_index.shape)
            == 1
            else data["atom_bond", "atom_angle", "atom_bond"].angle_2_index.shape[1]
        )
        n_sidechain_torsions = (
            data["atom", "atom_bond", "atom"].rotate_mask.sum()
            if self.sidechain_fragment_sigma is None
            else (
                data["atom", "atom_bond", "atom"].rotate_mask
                * ~data["atom", "atom_bond", "atom"].ring_sub_mask
            ).sum()
        )
        n_sidechain_fragments = data["atom", "atom_bond", "atom"].ring_sub_mask.sum()

        pos = data["atom"].flexdock_pos
        if self.bb_tr_sigma is not None:
            bb_tr_updates = torch.normal(
                mean=0, std=self.bb_tr_sigma, size=(n_conf, n_res, 3)
            )
            pos += bb_tr_updates[:, data["atom", "receptor"].edge_index[1]]

        if self.bb_rot_sigma is not None:
            bb_rot_updates = torch.tensor(
                so3.sample_vec(eps=self.bb_rot_sigma, shape=(n_conf, n_res, 3)),
                dtype=torch.float32,
                device=pos.device,
            )
            bb_rot_mats = axis_angle_to_matrix(bb_rot_updates)[
                :, data["atom", "receptor"].edge_index[1]
            ]
            pivots = pos[:, data["atom"].ca_mask][
                :, data["atom", "receptor"].edge_index[1]
            ]
            pos = ((pos - pivots).unsqueeze(-2) @ bb_rot_mats.swapaxes(-1, -2)).squeeze(
                -2
            ) + pivots

        if not self.rigid_sidechains:
            if self.sidechain_bond_sigma is not None and n_sidechain_bonds > 0:
                sidechain_bond_length_updates = torch.normal(
                    mean=0,
                    std=self.sidechain_bond_sigma,
                    size=(n_conf, n_sidechain_bonds),
                    device=pos.device,
                )
            else:
                sidechain_bond_length_updates = None

            if self.sidechain_angle_sigma is not None and n_sidechain_angles > 0:
                sidechain_bond_angle_updates = torch.normal(
                    mean=0,
                    std=self.sidechain_angle_sigma,
                    size=(n_conf, n_sidechain_angles),
                    device=pos.device,
                )
            else:
                sidechain_bond_angle_updates = None

            if self.sidechain_torsion_sigma is not None and n_sidechain_torsions > 0:
                sidechain_torsion_angle_updates = torch.normal(
                    mean=0,
                    std=self.sidechain_torsion_sigma,
                    size=(n_conf, n_sidechain_torsions),
                    device=pos.device,
                )
            else:
                sidechain_torsion_angle_updates = None

            if self.sidechain_fragment_sigma is not None and n_sidechain_fragments > 0:
                sidechain_fragment_updates = torch.tensor(
                    so3.sample_vec(
                        eps=self.sidechain_fragment_sigma,
                        shape=(n_conf, n_sidechain_fragments, 3),
                    ),
                    dtype=torch.float32,
                    device=pos.device,
                )
            else:
                sidechain_fragment_updates = None

            pos = modify_conformer(
                pos,
                data["atom", "atom_bond", "atom"].edge_index,
                data["atom", "atom_bond", "atom"].squeeze_mask,
                (
                    data["atom", "atom_bond", "atom"].rotate_mask
                    if sidechain_fragment_updates is None
                    else data["atom", "atom_bond", "atom"].rotate_mask
                    * ~data["atom", "atom_bond", "atom"].ring_sub_mask
                ),
                data["atom", "atom_bond", "atom"].ring_sub_mask,
                data["atom", "atom_bond", "atom"].ring_flip_mask,
                data["atom_bond", "atom_angle", "atom_bond"].angle_2_index,
                data["atom_bond", "atom"].fragment_index,
                bond_length_updates=sidechain_bond_length_updates,
                bond_angle_updates=sidechain_bond_angle_updates,
                torsion_angle_updates=sidechain_torsion_angle_updates,
                ring_sub_updates=sidechain_fragment_updates,
            )

        data["atom"].flexdock_pos = pos
        data["receptor"].flexdock_pos = data["atom"].flexdock_pos[
            :, data["atom"].ca_mask
        ]

        return data
