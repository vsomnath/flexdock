import copy
from typing import List, Callable

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from scipy.spatial.transform import Rotation

from flexdock.data.conformers.modify import (
    modify_conformer,
    modify_sidechains_old,
    modify_conformer_torsion_angles,
)
from flexdock.data.feature.helpers import (
    rotate_backbone_numpy,
    to_atom_grid_torch,
)
from flexdock.geometry.manifolds import so3
from flexdock.sampling.docking.diffusion import (
    set_time,
    get_t_schedule,
    bridge_transform_t,
)
from scipy.spatial.transform import Rotation as R


def get_schedules(
    inference_steps: int,
    bb_tr_bridge_alpha,
    bb_rot_bridge_alpha,
    sc_tor_bridge_alpha,
    sidechain_tor_bridge,
    inf_sched_alpha=1,
    inf_sched_beta=1,
    sigma_schedule="expbeta",
):
    t_schedule = get_t_schedule(
        sigma_schedule=sigma_schedule,
        inference_steps=inference_steps,
        inf_sched_alpha=inf_sched_alpha,
        inf_sched_beta=inf_sched_beta,
    )

    bb_tr_schedule = bridge_transform_t(t_schedule, bb_tr_bridge_alpha)
    bb_rot_schedule = bridge_transform_t(t_schedule, bb_rot_bridge_alpha)
    sc_tor_schedule = (
        bridge_transform_t(t_schedule, sc_tor_bridge_alpha)
        if sidechain_tor_bridge
        else t_schedule
    )

    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    schedules = {
        "t": t_schedule,
        "tr": tr_schedule,
        "rot": rot_schedule,
        "tor": tor_schedule,
        "sc_tor": sc_tor_schedule,
        "bb_tr": bb_tr_schedule,
        "bb_rot": bb_rot_schedule,
    }

    return schedules


def randomize_position(
    data_list: List[HeteroData],
    no_torsion: bool,
    no_random: bool,
    tr_sigma_max: float,
    flexible_sidechains: bool = False,
    flexible_backbone: bool = False,
    sidechain_tor_bridge: bool = False,
    use_bb_orientation_feats: bool = False,
    prior=None,
):
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi, size=complex_graph["ligand"].edge_mask.sum()
            )

            edge_index = complex_graph["ligand", "ligand"].edge_index.T
            edge_index_masked = edge_index[complex_graph["ligand"].edge_mask]

            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                complex_graph["ligand"].pos,
                edge_index_masked,
                complex_graph["ligand"].mask_rotate[0],
                torsion_updates,
            )

    if flexible_sidechains and not sidechain_tor_bridge:
        for complex_graph in data_list:
            sidechain_torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi, size=len(complex_graph["flexResidues"].edge_idx)
            )
            complex_graph["atom"].pos = modify_sidechains_old(
                complex_graph, complex_graph["atom"].pos, sidechain_torsion_updates
            )
            # Don't delete the part below.
            # complex_graph["atom"].orig_aligned_apo_pos = modify_sidechains_old(complex_graph, complex_graph["atom"].orig_aligned_apo_pos, sidechain_torsion_updates)
            # complex_graph["atom"].pos = complex_graph["atom"].orig_aligned_apo_pos

    elif flexible_sidechains and sidechain_tor_bridge:
        for complex_graph in data_list:
            sidechain_torsion_updates = np.concatenate(
                complex_graph.sc_conformer_match_rotations[0]
            )
            complex_graph["atom"].pos = modify_sidechains_old(
                complex_graph, complex_graph["atom"].pos, -sidechain_torsion_updates
            )

    if flexible_backbone:
        for complex_graph in data_list:
            complex_graph["atom"].pos = complex_graph[
                "atom"
            ].orig_aligned_apo_pos.float()
            complex_graph["receptor"].pos = complex_graph["atom"].pos[
                complex_graph["atom"].calpha
            ]

            # Add Gaussian or Harmonic noise to perturb structures slightly
            if prior is not None:
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors
                )
                calpha_delta_random = prior.sample_for_inference(complex_graph)
                complex_graph["receptor"].pos = (
                    complex_graph["receptor"].pos + calpha_delta_random
                )
                atom_grid = atom_grid + calpha_delta_random.unsqueeze(1)
                complex_graph["atom"].pos = atom_grid[x, y]

            if use_bb_orientation_feats:
                # compute orientation features
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors
                )
                complex_graph["receptor"].bb_orientation = torch.cat(
                    [
                        atom_grid[:, 0] - atom_grid[:, 1],
                        atom_grid[:, 2] - atom_grid[:, 1],
                    ],
                    dim=1,
                )

    for complex_graph in data_list:
        # set the center of the molecule to the center of the pocket atoms
        center_pocket = (
            complex_graph["atom"]
            .orig_aligned_apo_pos[complex_graph["atom"].nearby_atoms]
            .mean(dim=0)
            if hasattr(complex_graph["atom"], "nearby_atoms")
            else complex_graph["atom"].orig_aligned_apo_pos.mean(dim=0)
        )

        # randomize position
        molecule_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph["ligand"].pos = (
            complex_graph["ligand"].pos - molecule_center
        ) @ random_rotation.T + center_pocket.float()
        # base_rmsd = np.sqrt(np.sum(\
        # (complex_graph['ligand'].pos.cpu().numpy() - \
        # orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean()
        # )

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph["ligand"].pos += tr_update


def randomize_position_inf(
    data_list: List[HeteroData],
    no_torsion: bool,
    no_random: bool,
    tr_sigma_max: float,
    flexible_sidechains: bool = False,
    flexible_backbone: bool = False,
    sidechain_tor_bridge: bool = False,
    use_bb_orientation_feats: bool = False,
    prior=None,
    initial_noise_std_proportion: float = 1.0,
):
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi, size=complex_graph["ligand"].edge_mask.sum()
            )

            edge_index = complex_graph["ligand", "ligand"].edge_index.T
            edge_index_masked = edge_index[complex_graph["ligand"].edge_mask]

            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                complex_graph["ligand"].pos,
                edge_index_masked,
                complex_graph["ligand"].mask_rotate[0],
                torsion_updates,
            )

    if flexible_backbone:
        for complex_graph in data_list:
            complex_graph["atom"].pos = complex_graph[
                "atom"
            ].orig_aligned_apo_pos.float()
            complex_graph["receptor"].pos = complex_graph["atom"].pos[
                complex_graph["atom"].calpha
            ]

            # Add Gaussian or Harmonic noise to perturb structures slightly
            if prior is not None:
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors
                )
                calpha_delta_random = prior.sample_for_inference(complex_graph)
                complex_graph["receptor"].pos = (
                    complex_graph["receptor"].pos + calpha_delta_random
                )
                atom_grid = atom_grid + calpha_delta_random.unsqueeze(1)
                complex_graph["atom"].pos = atom_grid[x, y]

            if use_bb_orientation_feats:
                # compute orientation features
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors
                )
                complex_graph["receptor"].bb_orientation = torch.cat(
                    [
                        atom_grid[:, 0] - atom_grid[:, 1],
                        atom_grid[:, 2] - atom_grid[:, 1],
                    ],
                    dim=1,
                )
    else:
        if flexible_sidechains and not sidechain_tor_bridge:
            for complex_graph in data_list:
                sidechain_torsion_updates = np.random.uniform(
                    low=-np.pi,
                    high=np.pi,
                    size=len(complex_graph["flexResidues"].edge_idx),
                )
                complex_graph["atom"].pos = modify_sidechains_old(
                    complex_graph, complex_graph["atom"].pos, sidechain_torsion_updates
                )
                # Don't delete the part below.
                # complex_graph["atom"].orig_aligned_apo_pos = modify_sidechains_old(complex_graph, complex_graph["atom"].orig_aligned_apo_pos, sidechain_torsion_updates)
                # complex_graph["atom"].pos = complex_graph["atom"].orig_aligned_apo_pos

        elif flexible_sidechains and sidechain_tor_bridge:
            for complex_graph in data_list:
                sidechain_torsion_updates = np.concatenate(
                    complex_graph.sc_conformer_match_rotations[0]
                )
                complex_graph["atom"].pos = modify_sidechains_old(
                    complex_graph, complex_graph["atom"].pos, -sidechain_torsion_updates
                )

    for complex_graph in data_list:
        # set the center of the molecule to the center of the pocket atoms
        center_pocket = (
            complex_graph["atom"]
            .orig_aligned_apo_pos[complex_graph["atom"].nearby_atoms]
            .mean(dim=0)
            if hasattr(complex_graph["atom"], "nearby_atoms")
            else complex_graph["atom"].orig_aligned_apo_pos.mean(dim=0)
        )

        # randomize position
        molecule_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph["ligand"].pos = (
            complex_graph["ligand"].pos - molecule_center
        ) @ random_rotation.T + center_pocket.float()
        # base_rmsd = np.sqrt(np.sum(\
        # (complex_graph['ligand'].pos.cpu().numpy() - \
        # orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean()
        # )

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(
                mean=0, std=tr_sigma_max * initial_noise_std_proportion, size=(1, 3)
            )
            complex_graph["ligand"].pos += tr_update


def sampling(
    data_list: List[HeteroData],
    model: torch.nn.Module,
    inference_steps: int,
    schedules,
    sidechain_tor_bridge,
    device: str,
    t_to_sigma: Callable,
    model_args,
    no_random: bool = False,
    ode: bool = False,
    visualization_list=None,
    sidechain_visualization_list=None,
    confidence_model=None,
    filtering_data_list=None,
    filtering_model_args=None,
    batch_size: int = 32,
    no_final_step_noise: bool = False,
    return_full_trajectory: bool = False,
    debug_backbone: bool = False,
    debug_sidechain: bool = False,
    use_bb_orientation_feats: bool = False,
    diff_temp_sampling: tuple = None,
    diff_temp_psi: tuple = None,
    diff_temp_sigma_data: tuple = None,
    flow_temp_scale_0: tuple = None,
    flow_temp_scale_1: tuple = None,
):
    if model_args.flexible_sidechains:
        # If in the whole batch there are no flexible residues, we have to delete the
        # sidechain information, so that the loader does not break
        no_sidechains_in_batch = (
            sum([len(c["flexResidues"].subcomponents) for c in data_list]) == 0
        )
        if no_sidechains_in_batch:
            data_list = copy.deepcopy(data_list)
            for c in data_list:
                del c["flexResidues"]

    N = len(data_list)
    trajectory = []
    sidechain_trajectory = []

    tr_schedule = schedules["tr"]
    rot_schedule = schedules["rot"]
    tor_schedule = schedules["tor"]
    bb_tr_schedule = schedules["bb_tr"]
    bb_rot_schedule = schedules["bb_rot"]
    sc_tor_schedule = schedules["sc_tor"]
    t_schedule = schedules["t"]

    for t_idx in range(inference_steps):
        t_tr = tr_schedule[t_idx]
        t_rot = rot_schedule[t_idx]
        t_tor = tor_schedule[t_idx]
        t_sidechain_tor = sc_tor_schedule[t_idx]
        t_bb_tr = bb_tr_schedule[t_idx]
        t_bb_rot = bb_rot_schedule[t_idx]

        if model_args.lig_transform_type == "flow":
            dt_tr = (
                tr_schedule[t_idx + 1] - tr_schedule[t_idx]
                if t_idx < inference_steps - 1
                else 1 - tr_schedule[t_idx]
            )
            dt_rot = (
                rot_schedule[t_idx + 1] - rot_schedule[t_idx]
                if t_idx < inference_steps - 1
                else 1 - rot_schedule[t_idx]
            )
            dt_tor = (
                tor_schedule[t_idx + 1] - tor_schedule[t_idx]
                if t_idx < inference_steps - 1
                else 1 - tor_schedule[t_idx]
            )

        elif model_args.lig_transform_type == "diffusion":
            dt_tr = (
                tr_schedule[t_idx] - tr_schedule[t_idx + 1]
                if t_idx < inference_steps - 1
                else tr_schedule[t_idx]
            )
            dt_rot = (
                rot_schedule[t_idx] - rot_schedule[t_idx + 1]
                if t_idx < inference_steps - 1
                else rot_schedule[t_idx]
            )
            dt_tor = (
                tor_schedule[t_idx] - tor_schedule[t_idx + 1]
                if t_idx < inference_steps - 1
                else tor_schedule[t_idx]
            )

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        if return_full_trajectory:
            lig_pos = np.asarray(
                [
                    complex_graph["ligand"].pos.cpu().numpy()
                    for complex_graph in data_list
                ]
            )
            trajectory.append(lig_pos)

            if no_sidechains_in_batch:
                sidechain_trajectory.append(np.asarray([]))
            else:
                pos_to_add = []

                for complex_graph in data_list:
                    atom_pos = complex_graph["atom"].pos
                    flex_residues = complex_graph["flexResidues"]
                    res_subcomponents = flex_residues.subcomponents.unique()
                    pos_to_add.append(atom_pos[res_subcomponents].cpu().numpy())
                sidechain_trajectory.append(np.asarray(pos_to_add))

        tr_score_list = []
        rot_score_list = []
        tor_score_list = []
        sidechain_tor_score_list = []
        bb_tr_drift_list = []
        bb_rot_drift_list = []

        t_dict = {
            "tr": t_tr,
            "rot": t_rot,
            "tor": t_tor,
            "sc_tor": t_sidechain_tor,
            "bb_tr": t_bb_tr,
            "bb_rot": t_bb_rot,
        }

        sigma_dict = t_to_sigma(t_dict)
        tr_sigma = sigma_dict["tr_sigma"]
        rot_sigma = sigma_dict["rot_sigma"]
        tor_sigma = sigma_dict["tor_sigma"]
        sidechain_tor_sigma = sigma_dict["sc_tor_sigma"]
        bb_tr_sigma = sigma_dict["bb_tr_sigma"]
        bb_rot_sigma = sigma_dict["bb_rot_sigma"]

        include_miscellaneous_atoms = (
            hasattr(model_args, "include_miscellaneous_atoms")
            and model_args.include_miscellaneous_atoms
        )
        all_atoms = "all_atoms" in model_args and model_args.all_atoms

        for complex_graph_batch in loader:
            complex_graph_batch = complex_graph_batch.to(device)
            b = complex_graph_batch.num_graphs

            set_time(
                complex_graph_batch,
                t_schedule[t_idx] if t_schedule is not None else None,
                t_tr,
                t_rot,
                t_tor,
                t_sidechain_tor,
                t_bb_tr,
                t_bb_rot,
                b,
                all_atoms=all_atoms,
                device=device,
                include_miscellaneous_atoms=include_miscellaneous_atoms,
            )

            with torch.no_grad():
                if (
                    getattr(model_args, "precision", None) is not None
                    and model_args.precision == "bf16-mixed"
                ):
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.autocast(device_type=device_type, enabled=True):
                        outputs = model(complex_graph_batch)

                    tr_score = outputs["tr_pred"].float()
                    rot_score = outputs["rot_pred"].float()
                    tor_score = outputs["tor_pred"].float()
                    bb_tr_drift = outputs["bb_tr_pred"].float()
                    bb_rot_drift = outputs["bb_rot_pred"].float()
                    sidechain_tor_score = outputs["sc_tor_pred"].float()

                else:
                    outputs = model(complex_graph_batch)

                    tr_score = outputs["tr_pred"]
                    rot_score = outputs["rot_pred"]
                    tor_score = outputs["tor_pred"]
                    bb_tr_drift = outputs["bb_tr_pred"]
                    bb_rot_drift = outputs["bb_rot_pred"]
                    sidechain_tor_score = outputs["sc_tor_pred"]

            if len(bb_tr_drift.shape) == 3:
                bb_tr_drift = bb_tr_drift[:, -1]
                bb_rot_drift = bb_rot_drift[:, -1]

            tr_score_list.append(tr_score.cpu())
            rot_score_list.append(rot_score.cpu())
            tor_score_list.append(tor_score.cpu())

            if debug_backbone:
                print("debug backbone")
                calpha_mask = complex_graph_batch["atom"].calpha
                calpha_apo = complex_graph_batch["atom"].orig_aligned_apo_pos[
                    calpha_mask
                ]
                calpha_holo = complex_graph_batch["atom"].orig_holo_pos[calpha_mask]
                bb_tr_t = bb_tr_schedule[t_idx]

                calpha_atoms_t = calpha_apo * (1 - bb_tr_t) + calpha_holo * bb_tr_t

                bb_tr_drift = (calpha_holo - calpha_atoms_t) / (1 - bb_tr_t)
                bb_rot_drift = complex_graph_batch["receptor"].rot_vec

            bb_tr_drift_list.append(bb_tr_drift.cpu())
            bb_rot_drift_list.append(bb_rot_drift.cpu())

            if debug_sidechain and sidechain_tor_bridge:
                print("debug sidechain")
                rot = [
                    c2
                    for l in complex_graph_batch.sc_conformer_match_rotations
                    for c in l
                    for c2 in c
                ]
                sidechain_drift = torch.from_numpy(np.concatenate(rot)).float()
                sidechain_tor_score_list.append(sidechain_drift)
            else:
                sidechain_tor_score_list.append(sidechain_tor_score.cpu())

        tr_score = torch.cat(tr_score_list, dim=0)
        rot_score = torch.cat(rot_score_list, dim=0)
        tor_score = torch.cat(tor_score_list, dim=0)
        sidechain_tor_score = torch.cat(sidechain_tor_score_list, dim=0)
        bb_tr_drift = torch.cat(bb_tr_drift_list, dim=0)
        bb_rot_drift = torch.cat(bb_rot_drift_list, dim=0)

        if model_args.lig_transform_type == "flow":
            tr_perturb = tr_score.cpu() * dt_tr
            rot_perturb = rot_score.cpu() * dt_rot

            if not model_args.no_torsion:
                tor_perturb = (tor_score.cpu() * dt_tor).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // N
            else:
                tor_perturb = None

        elif model_args.lig_transform_type == "diffusion":
            tr_g = tr_sigma * torch.sqrt(
                torch.tensor(
                    2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)
                )
            )
            rot_g = rot_sigma * torch.sqrt(
                torch.tensor(
                    2 * np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)
                )
            )

            if ode:
                tr_perturb = (0.5 * tr_g**2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g**2).cpu()
            else:
                if no_random or (no_final_step_noise and t_idx == inference_steps - 1):
                    tr_z, rot_z = torch.zeros((N, 3)), torch.zeros((N, 3))
                else:
                    tr_z = torch.normal(mean=0, std=1, size=(N, 3))
                    rot_z = torch.normal(mean=0, std=1, size=(N, 3))

                tr_perturb = (
                    tr_g**2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z
                ).cpu()
                rot_perturb = (
                    rot_score.cpu() * dt_rot * rot_g**2
                    + rot_g * np.sqrt(dt_rot) * rot_z
                ).cpu()

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(
                    torch.tensor(
                        2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)
                    )
                )
                if ode:
                    tor_perturb = (0.5 * tor_g**2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    if no_random or (
                        no_final_step_noise and t_idx == inference_steps - 1
                    ):
                        tor_z = torch.zeros(tor_score.shape)
                    else:
                        tor_z = torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (
                        tor_g**2 * dt_tor * tor_score.cpu()
                        + tor_g * np.sqrt(dt_tor) * tor_z
                    ).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // N
            else:
                tor_perturb = None

            # diffusion low temperature sampling
            if diff_temp_sampling is not None:
                assert len(diff_temp_sampling) == 3
                assert len(diff_temp_psi) == 3
                assert len(diff_temp_sigma_data) == 3

                if diff_temp_sampling[0] != 1.0:
                    tr_sigma_data = np.exp(
                        diff_temp_sigma_data[0] * np.log(model_args.tr_sigma_max)
                        + (1 - diff_temp_sigma_data[0])
                        * np.log(model_args.tr_sigma_min)
                    )
                    lambda_tr = (tr_sigma_data + tr_sigma) / (
                        tr_sigma_data + tr_sigma / diff_temp_sampling[0]
                    )
                    tr_perturb = (
                        tr_g**2
                        * dt_tr
                        * (lambda_tr + diff_temp_sampling[0] * diff_temp_psi[0] / 2)
                        * tr_score
                        + tr_g * np.sqrt(dt_tr * (1 + diff_temp_psi[0])) * tr_z
                    )

                if diff_temp_sampling[1] != 1.0:
                    rot_sigma_data = np.exp(
                        diff_temp_sigma_data[1] * np.log(model_args.rot_sigma_max)
                        + (1 - diff_temp_sigma_data[1])
                        * np.log(model_args.rot_sigma_min)
                    )
                    lambda_rot = (rot_sigma_data + rot_sigma) / (
                        rot_sigma_data + rot_sigma / diff_temp_sampling[1]
                    )
                    rot_perturb = (
                        rot_g**2
                        * dt_rot
                        * (lambda_rot + diff_temp_sampling[1] * diff_temp_psi[1] / 2)
                        * rot_score
                        + rot_g * np.sqrt(dt_rot * (1 + diff_temp_psi[1])) * rot_z
                    )

                if diff_temp_sampling[2] != 1.0:
                    tor_sigma_data = np.exp(
                        diff_temp_sigma_data[2] * np.log(model_args.tor_sigma_max)
                        + (1 - diff_temp_sigma_data[2])
                        * np.log(model_args.tor_sigma_min)
                    )
                    lambda_tor = (tor_sigma_data + tor_sigma) / (
                        tor_sigma_data + tor_sigma / diff_temp_sampling[2]
                    )
                    tor_perturb = (
                        tor_g**2
                        * dt_tor
                        * (lambda_tor + diff_temp_sampling[2] * diff_temp_psi[2] / 2)
                        * tor_score
                        + tor_g * np.sqrt(dt_tor * (1 + diff_temp_psi[2])) * tor_z
                    )
                    if not isinstance(tor_perturb, np.ndarray):
                        tor_perturb = tor_perturb.cpu().numpy()

        # flow low temperature sampling
        if flow_temp_scale_0 is not None:
            assert len(flow_temp_scale_0) == 3
            assert len(flow_temp_scale_1) == 3

            bb_tr_drift = bb_tr_drift * (
                bb_tr_schedule[t_idx] * flow_temp_scale_0[0]
                + (1 - bb_tr_schedule[t_idx]) * flow_temp_scale_1[0]
            )
            bb_rot_drift = bb_rot_drift * (
                bb_rot_schedule[t_idx] * flow_temp_scale_0[1]
                + (1 - bb_rot_schedule[t_idx]) * flow_temp_scale_1[1]
            )
            sidechain_tor_score = sidechain_tor_score * (
                sc_tor_schedule[t_idx] * flow_temp_scale_0[2]
                + (1 - sc_tor_schedule[t_idx]) * flow_temp_scale_1[2]
            )

        if model_args.flexible_sidechains:
            if sidechain_tor_bridge:
                dt_sidechain_tor = (
                    sc_tor_schedule[t_idx + 1] - sc_tor_schedule[t_idx]
                    if t_idx < inference_steps - 1
                    else 1 - sc_tor_schedule[t_idx]
                )

                if ode:
                    sidechain_tor_perturb = (
                        dt_sidechain_tor * sidechain_tor_score.cpu()
                    ).numpy()
                else:
                    if no_random or (
                        no_final_step_noise and t_idx == inference_steps - 1
                    ):
                        sidechain_tor_z = torch.zeros(sidechain_tor_score.shape)
                    else:
                        sidechain_tor_z = torch.normal(
                            mean=0, std=1, size=sidechain_tor_score.shape
                        )
                    sidechain_tor_perturb = (
                        dt_sidechain_tor * sidechain_tor_score.cpu()
                        + np.sqrt(dt_sidechain_tor)
                        * sidechain_tor_sigma
                        * sidechain_tor_z
                    ).numpy()

            else:
                dt_sidechain_tor = (
                    sc_tor_schedule[t_idx] - sc_tor_schedule[t_idx + 1]
                    if t_idx < inference_steps - 1
                    else sc_tor_schedule[t_idx]
                )

                sidechain_tor_g = sidechain_tor_sigma * torch.sqrt(
                    torch.tensor(
                        2
                        * np.log(
                            model_args.sidechain_tor_sigma_max
                            / model_args.sidechain_tor_sigma_min
                        )
                    )
                )
                if ode:
                    sidechain_tor_perturb = (
                        0.5
                        * sidechain_tor_g**2
                        * dt_sidechain_tor
                        * sidechain_tor_score.cpu()
                    ).numpy()
                else:
                    if no_random or (
                        no_final_step_noise and t_idx == inference_steps - 1
                    ):
                        sidechain_tor_z = torch.zeros(sidechain_tor_score.shape)
                    else:
                        sidechain_tor_z = torch.normal(
                            mean=0, std=1, size=sidechain_tor_score.shape
                        )
                    sidechain_tor_perturb = (
                        sidechain_tor_g**2
                        * dt_sidechain_tor
                        * sidechain_tor_score.cpu()
                        + sidechain_tor_g * np.sqrt(dt_sidechain_tor) * sidechain_tor_z
                    ).numpy()

            sidechain_torsions_per_molecule = sidechain_tor_perturb.shape[0] // N
        else:
            sidechain_tor_perturb = None

        # Apply noise
        if model_args.flexible_sidechains:
            for i, complex_graph in enumerate(data_list):
                idx_start = i * sidechain_torsions_per_molecule
                idx_end = (i + 1) * sidechain_torsions_per_molecule

                complex_graph["atom"].pos = modify_sidechains_old(
                    complex_graph,
                    complex_graph["atom"].pos,
                    sidechain_tor_perturb[idx_start:idx_end],
                )

        # Perturb the backbone
        if model_args.flexible_backbone:
            dt_bb_tr = (
                bb_tr_schedule[t_idx + 1] - bb_tr_schedule[t_idx]
                if t_idx < inference_steps - 1
                else 1 - bb_tr_schedule[t_idx]
            )
            dt_bb_rot = (
                bb_rot_schedule[t_idx + 1] - bb_rot_schedule[t_idx]
                if t_idx < inference_steps - 1
                else 1 - bb_rot_schedule[t_idx]
            )
            calpha_atoms_per_molecule = bb_tr_drift.shape[0] // N

            if ode:
                bb_tr_perturb = (bb_tr_drift * dt_bb_tr).numpy()
                bb_rot_perturb = (bb_rot_drift * dt_bb_rot).numpy()
            else:
                if no_random or (no_final_step_noise and t_idx == inference_steps - 1):
                    bb_tr_z = torch.zeros(bb_tr_drift.shape)
                    bb_rot_z = torch.zeros(bb_rot_drift.shape)
                else:
                    bb_tr_z = torch.normal(mean=0, std=1, size=bb_tr_drift.shape)
                    bb_rot_z = torch.normal(mean=0, std=1, size=bb_rot_drift.shape)

                bb_tr_perturb = (
                    bb_tr_drift * dt_bb_tr + bb_tr_z * np.sqrt(dt_bb_tr) * bb_tr_sigma
                ).numpy()
                bb_rot_perturb = (
                    bb_rot_drift * dt_bb_rot
                    + bb_rot_z * np.sqrt(dt_bb_rot) * bb_rot_sigma
                ).numpy()

            for i, complex_graph in enumerate(data_list):
                idx_start = i * calpha_atoms_per_molecule
                idx_end = (i + 1) * calpha_atoms_per_molecule

                if hasattr(model_args, "ipa_model") and model_args.ipa_model:
                    rot_frames_t = complex_graph["receptor"].rot_frames
                    tr_frames_t = complex_graph["receptor"].tr_frames

                    rot_frames_t = so3.exp_map_at_point(
                        tangent_vec=bb_rot_perturb,
                        base_point=Rotation.from_matrix(
                            rot_frames_t.numpy()
                        ).as_rotvec(),
                    )
                    rot_frames_t = Rotation.from_rotvec(rot_frames_t).as_matrix()
                    tr_frames_t = tr_frames_t.numpy() + bb_tr_perturb

                    complex_graph["receptor"].rot_frames = torch.from_numpy(
                        rot_frames_t
                    ).float()
                    complex_graph["receptor"].tr_frames = torch.from_numpy(
                        tr_frames_t
                    ).float()

                new_pos = rotate_backbone_numpy(
                    atoms=complex_graph["atom"].pos.numpy(),
                    t_vec=bb_tr_perturb[idx_start:idx_end],
                    rot_vec=bb_rot_perturb[idx_start:idx_end],
                    lens_receptors=complex_graph["receptor"].lens_receptors.numpy(),
                )

                calpha_mask = complex_graph["atom"].calpha
                complex_graph["atom"].pos = torch.from_numpy(new_pos).float()
                complex_graph["receptor"].pos = complex_graph["atom"].pos[calpha_mask]

                # if model_args.ipa_model:
                #     rot_frames, tr_frames = construct_frames(
                #         complex_graph['atom'].pos,
                #         complex_graph['receptor'].lens_receptors
                #     )
                #     complex_graph['receptor'].rot_frames = rot_frames
                #     complex_graph['receptor'].tr_frames = tr_frames

                if use_bb_orientation_feats:
                    # compute orientation features
                    atom_grid, x, y = to_atom_grid_torch(
                        complex_graph["atom"].pos,
                        complex_graph["receptor"].lens_receptors,
                    )
                    complex_graph["receptor"].bb_orientation = torch.cat(
                        [
                            atom_grid[:, 0] - atom_grid[:, 1],
                            atom_grid[:, 2] - atom_grid[:, 1],
                        ],
                        dim=1,
                    )

        data_list = [
            modify_conformer(
                complex_graph,
                tr_perturb[i : i + 1],
                rot_perturb[i : i + 1].squeeze(0),
                tor_perturb[i * torsions_per_molecule : (i + 1) * torsions_per_molecule]
                if not model_args.no_torsion
                else None,
            )
            for i, complex_graph in enumerate(data_list)
        ]

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                filterHs = torch.not_equal(data_list[idx]["ligand"].x[:, 0], 0)
                ligand_pos = data_list[idx]["ligand"].pos
                orig_center = data_list[idx].original_center[0]
                visualization.add(
                    (ligand_pos[filterHs] + orig_center).detach().cpu(),
                    part=1,
                    order=t_idx + 2,
                )

        if sidechain_visualization_list is not None:
            for idx, visualization in enumerate(sidechain_visualization_list):
                # append current sidechain conformation
                visualization.append(
                    data_list[idx][0]["atom"].pos + data_list[idx][0]["original_center"]
                )

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            filtering_loader = iter(
                DataLoader(filtering_data_list, batch_size=batch_size)
            )
            confidence = []

            include_miscellaneous_atoms = (
                hasattr(filtering_model_args, "include_miscellaneous_atoms")
                and filtering_model_args.include_miscellaneous_atoms
            )

            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if filtering_data_list is not None:
                    filtering_complex_graph_batch = next(filtering_loader).to(device)
                    filtering_complex_graph_batch["ligand"].pos = complex_graph_batch[
                        "ligand"
                    ].pos
                    filtering_complex_graph_batch["atom"].pos = complex_graph_batch[
                        "atom"
                    ].pos
                    filtering_complex_graph_batch["receptor"].pos = complex_graph_batch[
                        "receptor"
                    ].pos

                    if sidechain_tor_bridge:
                        t_sc_tor = 1
                    else:
                        t_sc_tor = 0

                    t_bb_tr, t_bb_rot = 1, 1

                    set_time(
                        filtering_complex_graph_batch,
                        t=0,
                        t_tr=0,
                        t_rot=0,
                        t_tor=0,
                        t_sidechain_tor=t_sc_tor,
                        t_bb_tr=t_bb_tr,
                        t_bb_rot=t_bb_rot,
                        batchsize=N,
                        all_atoms=filtering_model_args.all_atoms,
                        device=device,
                        include_miscellaneous_atoms=include_miscellaneous_atoms,
                    )
                    pred = confidence_model(filtering_complex_graph_batch)
                    confidence.append(pred["filtering_pred"])
                else:
                    if sidechain_tor_bridge:
                        t_sc_tor = 1
                    else:
                        t_sc_tor = 0

                    t_bb_tr, t_bb_rot = 1, 1

                    set_time(
                        complex_graph_batch,
                        t=0,
                        t_tr=0,
                        t_rot=0,
                        t_tor=0,
                        t_sidechain_tor=t_sc_tor,
                        t_bb_tr=t_bb_tr,
                        t_bb_rot=t_bb_rot,
                        batchsize=N,
                        all_atoms=filtering_model_args.all_atoms,
                        device=device,
                        include_miscellaneous_atoms=include_miscellaneous_atoms,
                    )
                    pred = confidence_model(filtering_complex_graph_batch)
                    confidence.append(pred["filtering_pred"])

            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
    if return_full_trajectory:
        return data_list, confidence, trajectory, sidechain_trajectory
    return data_list, confidence
