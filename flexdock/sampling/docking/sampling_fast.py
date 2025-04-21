import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R

from flexdock.data.pipeline import ComplexData
from flexdock.data.conformers.modify import (
    modify_conformer_fast_batch,
    modify_sidechains_old,
    modify_conformer_torsion_angles,
)
from flexdock.data.feature.helpers import rotate_backbone_torch, to_atom_grid_torch
from flexdock.geometry.ops import axis_angle_to_matrix
from flexdock.sampling.docking.diffusion import set_time


def randomize_position_inf(
    data_list: list[HeteroData],
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

            edge_index = complex_graph["ligand", "lig_bond", "ligand"].edge_index

            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                pos=complex_graph["ligand"].pos,
                edge_index=edge_index,
                mask_rotate=complex_graph["ligand"].edge_mask,
                torsion_updates=torch.tensor(
                    torsion_updates, device=edge_index.device
                ).float(),
                fragment_index=complex_graph["ligand"].lig_fragment_index,
            )

    if flexible_backbone:
        for complex_graph in data_list:
            complex_graph["atom"].pos = complex_graph[
                "atom"
            ].orig_aligned_apo_pos.float()
            complex_graph["receptor"].pos = complex_graph["atom"].pos[
                complex_graph["atom"].ca_mask
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

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(
                mean=0, std=tr_sigma_max * initial_noise_std_proportion, size=(1, 3)
            )
            complex_graph["ligand"].pos += tr_update


def sampling(
    data_list: list[ComplexData],
    model: torch.nn.Module,
    inference_steps: int,
    schedules,
    sidechain_tor_bridge,
    device: str,
    t_to_sigma: callable,
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

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    include_miscellaneous_atoms = False
    all_atoms = hasattr(model_args, "all_atoms") and model_args.all_atoms

    final_data_list = []

    # Run diffusion and flow
    for complex_graph_batch in loader:
        complex_graph_batch = complex_graph_batch.to(device)
        b = complex_graph_batch.num_graphs

        for t_idx in range(inference_steps):
            inputs = complex_graph_batch.clone()

            t_tr = tr_schedule[t_idx]
            t_rot = rot_schedule[t_idx]
            t_tor = tor_schedule[t_idx]
            t_sidechain_tor = sc_tor_schedule[t_idx]
            t_bb_tr = bb_tr_schedule[t_idx]
            t_bb_rot = bb_rot_schedule[t_idx]

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

            set_time(
                inputs,
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

            with torch.no_grad():
                if (
                    getattr(model_args, "precision", None) is not None
                    and model_args.precision == "bf16-mixed"
                ):
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.autocast(device_type=device_type, enabled=True):
                        outputs = model(inputs, fast_updates=True)

                    tr_score = outputs["tr_pred"].float()
                    rot_score = outputs["rot_pred"].float()
                    tor_score = outputs["tor_pred"].float()
                    bb_tr_drift = outputs["bb_tr_pred"].float()
                    bb_rot_drift = outputs["bb_rot_pred"].float()
                    sidechain_tor_score = outputs["sc_tor_pred"].float()

                else:
                    outputs = model(inputs, fast_updates=True)

                    tr_score = outputs["tr_pred"]
                    rot_score = outputs["rot_pred"]
                    tor_score = outputs["tor_pred"]
                    bb_tr_drift = outputs["bb_tr_pred"]
                    bb_rot_drift = outputs["bb_rot_pred"]
                    sidechain_tor_score = outputs["sc_tor_pred"]

            tr_g = tr_sigma * torch.sqrt(
                torch.tensor(
                    2
                    * np.log(
                        model_args.sigma.tr_sigma_max / model_args.sigma.tr_sigma_min
                    ),
                    device=tr_score.device,
                )
            )

            rot_g = rot_sigma * torch.sqrt(
                torch.tensor(
                    2
                    * np.log(
                        model_args.sigma.rot_sigma_max / model_args.sigma.rot_sigma_min
                    ),
                    device=rot_score.device,
                )
            )

            if ode:
                tr_perturb = 0.5 * tr_g**2 * dt_tr * tr_score
                rot_perturb = 0.5 * rot_score * dt_rot * rot_g**2
            else:
                if no_random or (no_final_step_noise and t_idx == inference_steps - 1):
                    tr_z, rot_z = tr_score.new_zeros((b, 3)), rot_score.new_zeros(
                        (b, 3)
                    )
                else:
                    tr_z = torch.normal(
                        mean=0, std=1, size=(b, 3), device=tr_score.device
                    )
                    rot_z = torch.normal(
                        mean=0, std=1, size=(b, 3), device=rot_score.device
                    )

                tr_perturb = tr_g**2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z
                rot_perturb = (
                    rot_score * dt_rot * rot_g**2 + rot_g * np.sqrt(dt_rot) * rot_z
                )

                if not model_args.no_torsion:
                    tor_g = tor_sigma * torch.sqrt(
                        torch.tensor(
                            2
                            * np.log(
                                model_args.sigma.tor_sigma_max
                                / model_args.sigma.tor_sigma_min
                            ),
                            device=tor_score.device,
                        )
                    )
                    if ode:
                        tor_perturb = 0.5 * tor_g**2 * dt_tor * tor_score
                    else:
                        if no_random or (
                            no_final_step_noise and t_idx == inference_steps - 1
                        ):
                            tor_z = tor_perturb.new_zeros(tor_score.shape)
                        else:
                            tor_z = torch.normal(
                                mean=0,
                                std=1,
                                size=tor_score.shape,
                                device=tor_score.device,
                            )
                        tor_perturb = (
                            tor_g**2 * dt_tor * tor_score
                            + tor_g * np.sqrt(dt_tor) * tor_z
                        )
                else:
                    tor_perturb = None

                # diffusion low temperature sampling
                if diff_temp_sampling is not None:
                    assert len(diff_temp_sampling) == 3
                    assert len(diff_temp_psi) == 3
                    assert len(diff_temp_sigma_data) == 3

                    if diff_temp_sampling[0] != 1.0:
                        tr_sigma_data = np.exp(
                            diff_temp_sigma_data[0]
                            * np.log(model_args.sigma.tr_sigma_max)
                            + (1 - diff_temp_sigma_data[0])
                            * np.log(model_args.sigma.tr_sigma_min)
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
                            diff_temp_sigma_data[1]
                            * np.log(model_args.sigma.rot_sigma_max)
                            + (1 - diff_temp_sigma_data[1])
                            * np.log(model_args.sigma.rot_sigma_min)
                        )
                        lambda_rot = (rot_sigma_data + rot_sigma) / (
                            rot_sigma_data + rot_sigma / diff_temp_sampling[1]
                        )
                        rot_perturb = (
                            rot_g**2
                            * dt_rot
                            * (
                                lambda_rot
                                + diff_temp_sampling[1] * diff_temp_psi[1] / 2
                            )
                            * rot_score
                            + rot_g * np.sqrt(dt_rot * (1 + diff_temp_psi[1])) * rot_z
                        )

                    if diff_temp_sampling[2] != 1.0:
                        tor_sigma_data = np.exp(
                            diff_temp_sigma_data[2]
                            * np.log(model_args.sigma.tor_sigma_max)
                            + (1 - diff_temp_sigma_data[2])
                            * np.log(model_args.sigma.tor_sigma_min)
                        )
                        lambda_tor = (tor_sigma_data + tor_sigma) / (
                            tor_sigma_data + tor_sigma / diff_temp_sampling[2]
                        )
                        tor_perturb = (
                            tor_g**2
                            * dt_tor
                            * (
                                lambda_tor
                                + diff_temp_sampling[2] * diff_temp_psi[2] / 2
                            )
                            * tor_score
                            + tor_g * np.sqrt(dt_tor * (1 + diff_temp_psi[2])) * tor_z
                        )

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
                        sidechain_tor_perturb = dt_sidechain_tor * sidechain_tor_score
                    else:
                        if no_random or (
                            no_final_step_noise and t_idx == inference_steps - 1
                        ):
                            sidechain_tor_z = sidechain_tor_score.new_zeros(
                                sidechain_tor_score.shape
                            )
                        else:
                            sidechain_tor_z = torch.normal(
                                mean=0,
                                std=1,
                                size=sidechain_tor_score.shape,
                                device=sidechain_tor_score.device,
                            )
                        sidechain_tor_perturb = (
                            dt_sidechain_tor * sidechain_tor_score
                            + np.sqrt(dt_sidechain_tor)
                            * sidechain_tor_sigma
                            * sidechain_tor_z
                        )

                else:
                    raise ValueError(
                        "We only support sidechain_tor_bridge=True for sidechain updates"
                    )

                complex_graph_batch["atom"].pos = modify_conformer_torsion_angles(
                    pos=complex_graph_batch["atom"].pos,
                    edge_index=complex_graph_batch[
                        "atom", "atom_bond", "atom"
                    ].edge_index,
                    mask_rotate=complex_graph_batch[
                        "atom", "atom_bond", "atom"
                    ].edge_mask,
                    fragment_index=complex_graph_batch[
                        "atom_bond", "atom"
                    ].atom_fragment_index,
                    torsion_updates=sidechain_tor_perturb,
                    sidechains=True,
                )

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

                if ode:
                    bb_tr_perturb = bb_tr_drift * dt_bb_tr
                    bb_rot_perturb = bb_rot_drift * dt_bb_rot
                else:
                    if no_random or (
                        no_final_step_noise and t_idx == inference_steps - 1
                    ):
                        bb_tr_z = bb_tr_drift.new_zeros(bb_tr_drift.shape)
                        bb_rot_z = bb_rot_drift.new_zeros(bb_rot_drift.shape)
                    else:
                        bb_tr_z = torch.normal(
                            mean=0,
                            std=1,
                            size=bb_tr_drift.shape,
                            device=bb_tr_drift.device,
                        )
                        bb_rot_z = torch.normal(
                            mean=0,
                            std=1,
                            size=bb_rot_drift.shape,
                            device=bb_rot_drift.device,
                        )

                    bb_tr_perturb = (
                        bb_tr_drift * dt_bb_tr
                        + bb_tr_z * np.sqrt(dt_bb_tr) * bb_tr_sigma
                    )
                    bb_rot_perturb = (
                        bb_rot_drift * dt_bb_rot
                        + bb_rot_z * np.sqrt(dt_bb_rot) * bb_rot_sigma
                    )

                new_pos, _ = rotate_backbone_torch(
                    atoms=complex_graph_batch["atom"].pos,
                    t_vec=bb_tr_perturb,
                    rot_mat=axis_angle_to_matrix(bb_rot_perturb),
                    lens_receptors=complex_graph_batch["receptor"].lens_receptors,
                    total_rot=None,
                    detach=False,
                )

                calpha_mask = complex_graph_batch["atom"].ca_mask
                complex_graph_batch["atom"].pos = new_pos
                complex_graph_batch["receptor"].pos = complex_graph_batch["atom"].pos[
                    calpha_mask
                ]

                if use_bb_orientation_feats:
                    # compute orientation features
                    atom_grid, x, y = to_atom_grid_torch(
                        complex_graph_batch["atom"].pos,
                        complex_graph_batch["receptor"].lens_receptors,
                    )
                    complex_graph_batch["receptor"].bb_orientation = torch.cat(
                        [
                            atom_grid[:, 0] - atom_grid[:, 1],
                            atom_grid[:, 2] - atom_grid[:, 1],
                        ],
                        dim=1,
                    )

            complex_graph_batch = modify_conformer_fast_batch(
                data=complex_graph_batch,
                tr_update=tr_perturb,
                rot_update=rot_perturb,
                torsion_updates=tor_perturb,
            )

        # This is the perturbed graph we should be returning, but need to detach it
        final_data_list.extend(Batch.to_data_list(complex_graph_batch))

    # Run confidence model
    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(final_data_list, batch_size=batch_size)
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
                    pred = confidence_model(
                        filtering_complex_graph_batch, fast_updates=True
                    )
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
                    pred = confidence_model(
                        filtering_complex_graph_batch, fast_updates=True
                    )
                    confidence.append(pred["filtering_pred"])

            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    if return_full_trajectory:
        return final_data_list, confidence, trajectory, sidechain_trajectory
    return final_data_list, confidence
