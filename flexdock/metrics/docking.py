from typing import Union, List

import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from flexdock.geometry.ops import rigid_transform_kabsch_numpy
from flexdock.data.conformers import protein
from flexdock.data.conformers import molecule


def pli_lddt_score(
    rec_coords_predicted: torch.Tensor,
    lig_coords_predicted: torch.Tensor,
    rec_coords_true: torch.Tensor,
    lig_coords_true: torch.Tensor,
    pli_distance_threshold: float = 6.0,
    lddt_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> torch.Tensor:
    """Computes the local distance difference test (LDDT) score on the protein ligand interaction interface.

    All pairs of atoms for the true receptor and ligand that are within a distance threshold of eachother are
    considered for scoring.

    For each threshold provided in lddt_thresholds, if the predicted distance of the atom pairs are within the threshold,
    this contributes an equal amount to the score, up to 100 if all thresholds are passed.

    The final score is the average of the scores across atom pairs that are within the distance threshold.

    Args:
        rec_coords_predicted (torch.Tensor): predicted receptor atom coordinates
        lig_coords_predicted (torch.Tensor): predicted ligand atom coordinates
        rec_coords_true (torch.Tensor): true receptor atom coordinates
        lig_coords_true (torch.Tensor): true ligand atom coordinates
        pli_distance_threshold (float, optional): Distance for an atom pair to be consider in the protein ligand interface. Defaults to 6.0.
        lddt_thresholds (List[float], optional): Distance test thresholds for each atom pair. Defaults to [0.5, 1.0, 2.0, 4.0].

    Returns:
        torch.Tensor: pli-LDDT score for each prediction.
    """

    # Compute pairwise distances
    dmat_predicted = torch.cdist(rec_coords_predicted, lig_coords_predicted)
    dmat_true = torch.cdist(rec_coords_true, lig_coords_true)

    # Compute mask over distances
    dists_to_score = (dmat_true < pli_distance_threshold).float()
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # raise an error if no rec-lig distances are below the threshold
    if torch.sum(dists_to_score) == 0:
        raise ValueError(
            "No atoms are below the distance threshold, Check raw inputs!"
            f" Minimum distance between receptor and ligand atoms: {dmat_true.min()}"
        )

    # Number of thresholds passed / num thresholds
    score = torch.mean(
        torch.stack([(dist_l1 < threshold).float() for threshold in lddt_thresholds]),
        dim=0,
    )

    # Normalize over the appropriate axes.
    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
    return score


def convert_to_tensor(
    x: Union[float, torch.Tensor, np.ndarray, List[float]], device: torch.device
):
    """Converts x to a torch.Tensor, float type and moves it to the device"""
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    elif isinstance(x, List):
        return torch.from_numpy(np.array(x)).float().to(device)
    else:
        raise ValueError(f"Invalid type for x: {type(x)}, Implement the conversion")


class CustomMeanMetric(Metric):
    """CustomMeanMetric
    Computes nan-mean of the values since nans could be present.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def _custom_format(
        self, values: Union[float, torch.Tensor, np.ndarray, List[float]]
    ) -> torch.Tensor:
        converted_values = convert_to_tensor(values, self.device)
        return converted_values.to(self.device)

    def update(
        self, values: Union[float, torch.Tensor, np.ndarray, List[float]]
    ) -> None:
        values = self._custom_format(values)
        self.values.append(values)

    def compute(self) -> torch.Tensor:
        # Concatenate all the values
        values = dim_zero_cat(self.values)

        # Compute the nanmean
        return torch.nanmean(values)


def align_proteins(
    true_atom_pos, pred_atom_pos, ca_mask, nearby_atom_mask, mode="nearby_atoms"
):
    if mode == "nearby_atoms":
        assert nearby_atom_mask is not None
        R, t, _rmsd = rigid_transform_kabsch_numpy(
            true_atom_pos[nearby_atom_mask], pred_atom_pos[nearby_atom_mask]
        )

    elif mode == "calpha":
        assert ca_mask is not None, "calpha cannot be None when mode=calpha"
        R, t, _rmsd = rigid_transform_kabsch_numpy(
            true_atom_pos[ca_mask], pred_atom_pos[ca_mask]
        )

    elif mode == "all_atoms":
        R, t, _rmsd = rigid_transform_kabsch_numpy(true_atom_pos, pred_atom_pos)

    elif mode == "noalign":
        R = np.eye(3)
        t = np.zeros((1, 3))
        _rmsd = 0.0

    return R, t, _rmsd


def compute_ligand_rmsd(true_ligand_pos, pred_ligand_pos, name, mol=None):
    if mol is not None:
        try:
            print(f"Trying to compute symmetry RMSD for {name}")
            rmsd = molecule.get_symmetry_rmsd(mol, true_ligand_pos, [pred_ligand_pos])[
                0
            ]
        except Exception as e:
            print(f"Complex={name}: Using non corrected RMSD because of the error {e}")
            rmsd = np.sqrt(
                ((pred_ligand_pos - true_ligand_pos) ** 2).sum(axis=1).mean(axis=0)
            )
    else:
        rmsd = np.sqrt(
            ((pred_ligand_pos - true_ligand_pos) ** 2).sum(axis=1).mean(axis=0)
        )

    return rmsd


def compute_metrics(
    complex_id,
    pred_atom_pos_list,
    pred_lig_pos_list,
    true_atom_pos,
    true_lig_pos,
    filterHs,
    ca_mask,
    nearby_atom_mask,
    orig_mol=None,
    align_proteins_by: str = "nearby_atoms",
):
    rmsds, centroid_distances, bb_rmsds, aa_rmsds = [], [], [], []
    rmsds_before_alignment = []

    for pred_atom_pos, pred_lig_pos in zip(pred_atom_pos_list, pred_lig_pos_list):
        rmsd_before_alignment = compute_ligand_rmsd(
            true_lig_pos, pred_lig_pos[filterHs], name=complex_id, mol=orig_mol
        )
        rmsds_before_alignment.append(rmsd_before_alignment)

        try:
            R, t, _rmsd = align_proteins(
                true_atom_pos,
                pred_atom_pos,
                ca_mask=ca_mask,
                nearby_atom_mask=nearby_atom_mask,
                mode=align_proteins_by,
            )
        #    atom_p = (R @ atom_p.T).T + t
        #    ligand_p = (R @ ligand_p.T).T + t
        except Exception:
            print(f"{complex_id}: Alignment failed, using no alignment")
            R = np.eye(3)
            t = np.zeros((1, 3))

        pred_atom_pos = (R @ pred_atom_pos.T).T + t
        pred_lig_pos = (R @ pred_lig_pos.T).T + t

        rmsd = compute_ligand_rmsd(
            true_lig_pos, pred_lig_pos[filterHs], name=complex_id, mol=orig_mol
        )
        rmsds.append(rmsd)

        centroid_distance = np.linalg.norm(
            pred_lig_pos.mean(axis=0) - true_lig_pos.mean(axis=0)
        )
        centroid_distances.append(centroid_distance)

        calpha_pred_atoms = pred_atom_pos[ca_mask]
        calpha_holo_atoms = true_atom_pos[ca_mask]
        calpha_rmsd = np.sqrt(
            ((calpha_pred_atoms - calpha_holo_atoms) ** 2).sum(axis=1).mean(axis=0)
        )
        bb_rmsds.append(calpha_rmsd)

        aa_rmsd = protein.scRMSD(nearby_atom_mask, pred_atom_pos, true_atom_pos)
        aa_rmsds.append(aa_rmsd)

    metrics = {
        "aa_rmsds": aa_rmsds,
        "bb_rmsds": bb_rmsds,
        "rmsds": rmsds,
        "centroid_distances": centroid_distances,
        "rmsds_before_alignment": rmsds_before_alignment,
    }

    return metrics
