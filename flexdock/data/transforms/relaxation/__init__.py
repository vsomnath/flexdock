import random

import numpy as np
from torch_geometric.transforms import BaseTransform, Compose
import torch

from flexdock.data.transforms.relaxation.molecule import (
    LigandNoiseTransform,
)
from flexdock.data.transforms.relaxation.protein import ProteinNoiseTransform
from flexdock.data.transforms.relaxation.components import (
    SubsetTransform,
    EuclideanNoiseTransform,
)
from flexdock.geometry.ops import rigid_transform_kabsch


class SampleTransform(BaseTransform):
    def __init__(self, rmsd_cutoff=None, sampling_epsilon=None, sampling_kappa=None):
        # assert rmsd_cutoff is not None or sampling_kappa is not None
        self.rmsd_cutoff = rmsd_cutoff
        self.sampling_epsilon = sampling_epsilon
        self.sampling_kappa = sampling_kappa

    def __call__(self, data):
        if data is None:
            return None
        lig_pos = data["ligand"].flexdock_pos
        atom_pos = data["atom"].flexdock_pos

        try:
            R, tr = rigid_transform_kabsch(
                atom_pos[:, data["atom"].nearby_atom_mask],
                data["atom"].orig_holo_pos[data["atom"].nearby_atom_mask],
            )
        except:
            return None
        lig_pos = lig_pos @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
        atom_pos = atom_pos @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)

        lig_rmsd = torch.sqrt(
            torch.mean(
                torch.sum((lig_pos - data["ligand"].orig_pos) ** 2, dim=-1), dim=-1
            )
        )
        atom_rmsd = torch.sqrt(
            torch.mean(
                torch.sum((atom_pos - data["atom"].orig_holo_pos) ** 2, dim=-1), dim=-1
            )
        )
        rmsd = (lig_rmsd + atom_rmsd) / 2

        if self.rmsd_cutoff is not None:
            if self.sampling_epsilon is None or random.random() > self.sampling_epsilon:
                weights = (rmsd < self.rmsd_cutoff).double()
            else:
                weights = (rmsd > self.rmsd_cutoff).double()
        else:
            weights = torch.exp(-1 * (rmsd**2) / self.sampling_kappa).double()
            weights[weights.isnan()] = 0.0

        if weights.sum() == 0:
            return None
        probs = weights / weights.sum()
        conf_idx = np.random.choice(probs.shape[0], p=probs)
        data["ligand"].flexdock_pos = lig_pos[conf_idx]
        data["atom"].flexdock_pos = atom_pos[conf_idx]
        return data


class FlowTransform(BaseTransform):
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, data):
        if data is None:
            return data
        data.complex_t = float(np.random.beta(self.alpha, self.beta))
        data["ligand"].pos = data["ligand"].flexdock_pos + data.complex_t * (
            data["ligand"].orig_pos - data["ligand"].flexdock_pos
        )
        data["atom"].pos = data["atom"].flexdock_pos + data.complex_t * (
            data["atom"].orig_holo_pos - data["atom"].flexdock_pos
        )

        data["ligand"].u = data["ligand"].orig_pos - data["ligand"].flexdock_pos
        data["atom"].u = data["atom"].orig_holo_pos - data["atom"].flexdock_pos

        data["receptor"].pos = data["atom"].pos[data["atom"].ca_mask]

        atom_center = torch.mean(
            data["atom"].pos[data["atom"].nearby_atom_mask], axis=0, keepdims=True
        )
        data.original_center = atom_center
        for store in ["ligand", "atom", "receptor"]:
            data[store].node_t = torch.ones(data[store].num_nodes) * data.complex_t
            data[store].pos -= atom_center
        return data


def construct_transform(
    n_conformers,
    tr_sigma,
    rot_sigma,
    lig_bond_sigma,
    lig_angle_sigma,
    lig_torsion_sigma,
    lig_fragment_sigma,
    bb_tr_sigma,
    bb_rot_sigma,
    sidechain_bond_sigma,
    sidechain_angle_sigma,
    sidechain_torsion_sigma,
    sidechain_fragment_sigma,
    lig_sigma,
    atom_sigma,
    nearby_atom_cutoff,
    rmsd_cutoff,
    sampling_kappa,
    sampling_epsilon,
    sampling_alpha,
    sampling_beta,
):
    transform_sequence = []
    if n_conformers is not None:
        transform_sequence.append(SubsetTransform(n=n_conformers))
    if any(
        sigma is not None
        for sigma in (
            tr_sigma,
            rot_sigma,
            lig_bond_sigma,
            lig_angle_sigma,
            lig_torsion_sigma,
            lig_fragment_sigma,
        )
    ):
        transform_sequence.append(
            LigandNoiseTransform(
                tr_sigma=tr_sigma,
                rot_sigma=rot_sigma,
                bond_sigma=lig_bond_sigma,
                angle_sigma=lig_angle_sigma,
                torsion_sigma=lig_torsion_sigma,
                fragment_sigma=lig_fragment_sigma,
            )
        )
    if any(
        sigma is not None
        for sigma in (
            bb_tr_sigma,
            bb_rot_sigma,
            sidechain_bond_sigma,
            sidechain_angle_sigma,
            sidechain_torsion_sigma,
            sidechain_fragment_sigma,
        )
    ):
        transform_sequence.append(
            ProteinNoiseTransform(
                bb_tr_sigma=bb_tr_sigma,
                bb_rot_sigma=bb_rot_sigma,
                sidechain_bond_sigma=sidechain_bond_sigma,
                sidechain_angle_sigma=sidechain_angle_sigma,
                sidechain_torsion_sigma=sidechain_torsion_sigma,
                sidechain_fragment_sigma=sidechain_fragment_sigma,
            )
        )
    if lig_sigma is not None or atom_sigma is not None:
        transform_sequence.append(
            EuclideanNoiseTransform(lig_sigma=lig_sigma, atom_sigma=atom_sigma)
        )
    transform_sequence.extend(
        [
            SampleTransform(
                rmsd_cutoff=rmsd_cutoff,
                sampling_epsilon=sampling_epsilon,
                sampling_kappa=sampling_kappa,
            ),
            FlowTransform(alpha=sampling_alpha, beta=sampling_beta),
        ]
    )
    transform = Compose(transform_sequence)
    return transform
