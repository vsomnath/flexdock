from torch_geometric.transforms import BaseTransform
import torch
from torch_scatter import scatter_mean

from flexdock.data.conformers.modify import modify_conformer
from flexdock.geometry.ops import (
    rigid_transform_kabsch,
    rigid_transform_kabsch_batch,
    axis_angle_to_matrix,
)
from flexdock.geometry.manifolds import so3


class LigandNoiseTransform(BaseTransform):
    def __init__(
        self,
        tr_sigma=None,
        rot_sigma=None,
        bond_sigma=None,
        angle_sigma=None,
        torsion_sigma=None,
        fragment_sigma=None,
    ):
        self.tr_sigma = tr_sigma
        self.rot_sigma = rot_sigma
        self.bond_sigma = bond_sigma
        self.angle_sigma = angle_sigma
        self.torsion_sigma = torsion_sigma
        self.fragment_sigma = fragment_sigma
        self.rigid = all(
            sigma is None
            for sigma in (bond_sigma, angle_sigma, torsion_sigma, fragment_sigma)
        )

    def __call__(self, data):
        n_conf = data["ligand"].flexdock_pos.shape[0]
        n_bonds = data["ligand", "lig_bond", "ligand"].squeeze_mask.sum()
        n_angles = (
            0
            if len(data["lig_bond", "lig_angle", "lig_bond"].angle_2_index.shape) == 1
            else data["lig_bond", "lig_angle", "lig_bond"].angle_2_index.shape[1]
        )
        n_torsions = (
            data["ligand", "lig_bond", "ligand"].rotate_mask.sum()
            if self.fragment_sigma is None
            else (
                data["ligand", "lig_bond", "ligand"].rotate_mask
                * ~data["ligand", "lig_bond", "ligand"].ring_sub_mask
            ).sum()
        )
        n_fragments = data["ligand", "lig_bond", "ligand"].ring_sub_mask.sum()

        pos = data["ligand"].flexdock_pos

        if self.tr_sigma is not None:
            tr_updates = torch.normal(mean=0, std=self.tr_sigma, size=(n_conf, 3))
            pos += tr_updates.unsqueeze(-2)

        if self.rot_sigma is not None:
            rot_updates = torch.FloatTensor(
                so3.sample_vec(eps=self.rot_sigma, shape=(n_conf, 3))
            )
            rot_mats = axis_angle_to_matrix(rot_updates)
            centroids = pos.mean(dim=-2, keepdims=True)
            pos = (pos - centroids) @ rot_mats.swapaxes(-1, -2) + centroids

        if not self.rigid:
            if self.bond_sigma is not None and n_bonds > 0:
                bond_length_updates = torch.normal(
                    mean=0, std=self.bond_sigma, size=(n_bonds,), device=pos.device
                )
            else:
                bond_length_updates = None

            if self.angle_sigma is not None and n_angles > 0:
                bond_angle_updates = torch.normal(
                    mean=0, std=self.angle_sigma, size=(n_angles,), device=pos.device
                )
            else:
                bond_angle_updates = None

            if self.torsion_sigma is not None and n_torsions > 0:
                torsion_angle_updates = torch.normal(
                    mean=0,
                    std=self.torsion_sigma,
                    size=(n_torsions,),
                    device=pos.device,
                )
            else:
                torsion_angle_updates = None

            if self.fragment_sigma is not None and n_fragments > 0:
                fragment_updates = torch.tensor(
                    so3.sample_vec(eps=self.fragment_sigma, shape=(n_fragments, 3)),
                    dtype=torch.float32,
                    device=pos.device,
                )
            else:
                fragment_updates = None

            flexible_pos = modify_conformer(
                pos,
                data["ligand", "lig_bond", "ligand"].edge_index,
                data["ligand", "lig_bond", "ligand"].squeeze_mask,
                (
                    data["ligand", "lig_bond", "ligand"].rotate_mask
                    if fragment_updates is None
                    else data["ligand", "lig_bond", "ligand"].rotate_mask
                    * ~data["ligand", "lig_bond", "ligand"].ring_sub_mask
                ),
                data["ligand", "lig_bond", "ligand"].ring_sub_mask,
                data["ligand", "lig_bond", "ligand"].ring_flip_mask,
                data["lig_bond", "lig_angle", "lig_bond"].angle_2_index,
                data["lig_bond", "ligand"].fragment_index,
                bond_length_updates=bond_length_updates,
                bond_angle_updates=bond_angle_updates,
                torsion_angle_updates=torsion_angle_updates,
                ring_sub_updates=fragment_updates,
            )

            R, tr = rigid_transform_kabsch(flexible_pos, pos)
            pos = flexible_pos @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)

        data["ligand"].flexdock_pos = pos

        return data


class BatchLigandNoiseTransform(BaseTransform):
    def __init__(
        self,
        tr_sigma=None,
        rot_sigma=None,
        bond_sigma=None,
        angle_sigma=None,
        torsion_sigma=None,
        fragment_sigma=None,
    ):
        self.tr_sigma = tr_sigma
        self.rot_sigma = rot_sigma
        self.bond_sigma = bond_sigma
        self.angle_sigma = angle_sigma
        self.torsion_sigma = torsion_sigma
        self.fragment_sigma = fragment_sigma
        self.rigid = all(
            sigma is None
            for sigma in (bond_sigma, angle_sigma, torsion_sigma, fragment_sigma)
        )

    def __call__(self, data):
        n_conf = torch.amax(data["ligand"].batch) + 1
        n_bonds = data["ligand", "lig_bond", "ligand"].squeeze_mask.sum()
        n_angles = (
            0
            if len(data["lig_bond", "lig_angle", "lig_bond"].angle_2_index.shape) == 1
            else data["lig_bond", "lig_angle", "lig_bond"].angle_2_index.shape[1]
        )
        n_torsions = data["ligand", "lig_bond", "ligand"].rotate_mask.sum()
        n_fragments = data["ligand", "lig_bond", "ligand"].ring_sub_mask.sum()

        pos = data["ligand"].pos

        if self.tr_sigma is not None:
            tr_updates = torch.normal(
                mean=0, std=self.tr_sigma, size=(n_conf, 3), device=pos.device
            )
            pos += tr_updates[data["ligand"].batch]

        if self.rot_sigma is not None:
            rot_updates = torch.tensor(
                so3.sample_vec(eps=self.rot_sigma, shape=(n_conf, 3)),
                dtype=torch.float32,
                device=pos.device,
            )
            rot_mats = axis_angle_to_matrix(rot_updates)
            centroids = scatter_mean(pos, data["ligand"].batch, dim=0)
            pos = (
                (pos - centroids[data["ligand"].batch]).unsqueeze(-2)
                @ rot_mats[data["ligand"].batch].swapaxes(-1, -2)
                + centroids[data["ligand"].batch].unsqueeze(-2)
            ).squeeze()

        if not self.rigid:
            if self.bond_sigma is not None and n_bonds > 0:
                bond_length_updates = torch.normal(
                    mean=0, std=self.bond_sigma, size=(n_bonds,), device=pos.device
                )
            else:
                bond_length_updates = None

            if self.angle_sigma is not None and n_angles > 0:
                bond_angle_updates = torch.normal(
                    mean=0, std=self.angle_sigma, size=(n_angles,), device=pos.device
                )
            else:
                bond_angle_updates = None

            if self.torsion_sigma is not None and n_torsions > 0:
                torsion_angle_updates = torch.normal(
                    mean=0,
                    std=self.torsion_sigma,
                    size=(n_torsions,),
                    device=pos.device,
                )
            else:
                torsion_angle_updates = None

            if self.fragment_sigma is not None and n_fragments > 0:
                fragment_updates = torch.tensor(
                    so3.sample_vec(eps=self.fragment_sigma, shape=(n_fragments, 3)),
                    dtype=torch.float32,
                    device=pos.device,
                )
            else:
                fragment_updates = None

            flexible_pos = modify_conformer(
                pos,
                data["ligand", "lig_bond", "ligand"].edge_index,
                data["ligand", "lig_bond", "ligand"].squeeze_mask,
                (
                    data["ligand", "lig_bond", "ligand"].rotate_mask
                    if fragment_updates is None
                    else data["ligand", "lig_bond", "ligand"].rotate_mask
                    * ~data["ligand", "lig_bond", "ligand"].ring_sub_mask
                ),
                data["ligand", "lig_bond", "ligand"].ring_sub_mask,
                data["ligand", "lig_bond", "ligand"].ring_flip_mask,
                data["lig_bond", "lig_angle", "lig_bond"].angle_2_index,
                data["lig_bond", "ligand"].fragment_index,
                bond_length_updates=bond_length_updates,
                bond_angle_updates=bond_angle_updates,
                torsion_angle_updates=torsion_angle_updates,
                ring_sub_updates=fragment_updates,
            )

            R, tr = rigid_transform_kabsch_batch(
                flexible_pos, pos, data["ligand"].batch
            )
            pos = (
                flexible_pos.unsqueeze(-2) @ R[data["ligand"].batch].swapaxes(-1, -2)
                + tr[data["ligand"].batch].unsqueeze(-2)
            ).squeeze()

        data["ligand"].pos = pos

        return data
