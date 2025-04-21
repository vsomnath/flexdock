import numpy as np
import torch

from flexdock.geometry.manifolds import so3, torus
from flexdock.data.conformers.modify import modify_conformer


class LigandTransform:
    def __init__(
        self,
        no_torsion: bool = False,
        fast_updates: bool = False,
    ):
        self.no_torsion = no_torsion
        self.fast_updates = fast_updates

    def __call__(self, data, t_dict, sigma_dict):
        return self.apply_diffusion_transform(data, t_dict, sigma_dict)

    def apply_diffusion_transform(self, data, t_dict, sigma_dict):
        tr_sigma, rot_sigma = sigma_dict["tr_sigma"], sigma_dict["rot_sigma"]
        if not self.no_torsion:
            tor_sigma = sigma_dict["tor_sigma"]

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3))
        rot_update = so3.sample_vec(eps=rot_sigma)

        if not self.no_torsion:
            torsion_updates = np.random.normal(
                loc=0.0, scale=tor_sigma, size=data["ligand"].edge_mask.sum()
            )

        # Now modified to move between fast (on GPU) vs CPU <-> GPU
        modify_conformer(
            data,
            tr_update,
            torch.from_numpy(rot_update).float(),
            torsion_updates,
            fast=self.fast_updates,
        )

        data.tr_score = -tr_update / tr_sigma**2
        data.rot_score = (
            torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma))
            .float()
            .unsqueeze(0)
        )
        data.tor_score = (
            None
            if self.no_torsion
            else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        )
        data.tor_sigma_edge = (
            None
            if self.no_torsion
            else np.ones(data["ligand"].edge_mask.sum()) * tor_sigma
        )

        return data
