from collections import defaultdict

import numpy as np
import torch

from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

from flexdock.geometry.ops import rigid_transform_kabsch

try:
    from relaxflow.utils.forces import (
        get_torsion_force,
        get_bond_force,
        get_angle_force,
    )
except Exception:
    pass


def set_time(batch, t):
    batch.complex_t = torch.full(
        (batch.num_graphs,), t, device=batch["conf_idx"].device
    )
    batch["ligand"].node_t = torch.full(
        (batch["ligand"].num_nodes,), t, device=batch["ligand"].pos.device
    )
    batch["receptor"].node_t = torch.full(
        (batch["receptor"].num_nodes,), t, device=batch["receptor"].pos.device
    )
    batch["atom"].node_t = torch.full(
        (batch["atom"].num_nodes,), t, device=batch["atom"].pos.device
    )


def center_complex(data):
    atom_center = scatter_mean(
        data["atom"].pos[data["atom"].nearby_atom_mask],
        data["atom"].batch[data["atom"].nearby_atom_mask],
        dim=0,
    )
    # atom_center = scatter_mean(data['atom'].pos, data['atom'].batch, dim=0)
    data["ligand"].pos -= atom_center[data["ligand"].batch]
    data["atom"].pos -= atom_center[data["atom"].batch]
    data["receptor"].pos = data["atom"].pos[data["atom"].ca_mask]
    return data


def sigmoid(t):
    return 1 / (1 + np.e ** (-t))


def sigmoid_schedule(t, k):
    s = lambda t: sigmoid(k * t)
    return (s(t) - s(0)) / (s(1) - s(0))


def exponential_schedule(t, a):
    return (np.exp(a * t) - 1) / (np.exp(a) - 1)


@torch.no_grad()
def sampling_on_batch(
    model,
    batch,
    inference_steps,
    x_zero_pred=False,
    save_traj=True,
    schedule_type="uniform",
    schedule_param=1.0,
):
    batch = center_complex(batch)
    if save_traj:
        lig_traj = batch["ligand"].pos.clone().unsqueeze(1)
        atom_traj = batch["atom"].pos.clone().unsqueeze(1)

    t_schedule = np.linspace(0, 1, inference_steps + 1)
    if schedule_type == "sigmoid":
        t_schedule = sigmoid_schedule(t_schedule, schedule_param)
    if schedule_type == "exponential":
        t_schedule = exponential_schedule(t_schedule, schedule_param)

    for t_idx in range(t_schedule.shape[0] - 1):
        t = t_schedule[t_idx]
        dt = t_schedule[t_idx + 1] - t_schedule[t_idx]
        set_time(batch, t)

        lig_pred, atom_pred = model(batch)

        if x_zero_pred:
            lig_update = (lig_pred - batch["ligand"].pos) * dt / (1 - t)
            atom_update = (atom_pred - batch["atom"].pos) * dt / (1 - t)
        else:
            lig_update = lig_pred * dt
            atom_update = atom_pred * dt
        batch["ligand"].pos += lig_update
        batch["atom"].pos += atom_update
        batch["receptor"].pos = batch["atom"].pos[batch["atom"].ca_mask]
        batch = center_complex(batch)

        if save_traj:
            lig_traj = torch.cat(
                (lig_traj, batch["ligand"].pos.clone().unsqueeze(1)), dim=1
            )
            atom_traj = torch.cat(
                (atom_traj, batch["atom"].pos.clone().unsqueeze(1)), dim=1
            )

    if save_traj:
        return lig_traj, atom_traj
    else:
        return batch["ligand"].pos, batch["atom"].pos


def sampling_on_confs(
    model,
    conf_loader,
    inference_steps,
    x_zero_pred=False,
    save_traj=False,
    device=None,
    schedule_type="uniform",
    schedule_param=1.0,
):
    lig_pred, atom_pred = [], []
    for batch in conf_loader:
        lig_pred_batch, atom_pred_batch = sampling_on_batch(
            model,
            batch.to(device),
            inference_steps,
            x_zero_pred=x_zero_pred,
            save_traj=save_traj,
            schedule_type=schedule_type,
            schedule_param=schedule_param,
        )  # N_Batch_Atoms x 3
        lig_pred.append(to_dense_batch(lig_pred_batch, batch["ligand"].batch)[0])
        atom_pred.append(to_dense_batch(atom_pred_batch, batch["atom"].batch)[0])
    lig_pred = torch.cat(lig_pred, dim=0)
    atom_pred = torch.cat(atom_pred, dim=0)
    if save_traj:
        lig_pred = lig_pred.permute(2, 0, 1, 3)
        atom_pred = atom_pred.permute(2, 0, 1, 3)
    return lig_pred, atom_pred


def compute_min_rmsds(graph, lig_pred, atom_pred):
    # R, tr = rigid_transform_kabsch_pairs(
    #     atom_pred.index_select(-2, torch.argwhere(graph['atom'].ca_mask).squeeze()),
    #     graph['atom'].tgt_pos[graph['atom'].ca_mask].swapaxes(0,1)
    # )
    R, tr = rigid_transform_kabsch(atom_pred, graph["atom"].tgt_pos)
    aligned_lig_pred = lig_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
    aligned_atom_pred = atom_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
    lig_rmsds = torch.sqrt(
        torch.mean(
            torch.sum((aligned_lig_pred - graph["ligand"].tgt_pos) ** 2, axis=-1),
            axis=-1,
        )
    )
    atom_rmsds = torch.sqrt(
        torch.mean(
            torch.sum((aligned_atom_pred - graph["atom"].tgt_pos) ** 2, axis=-1),
            axis=-1,
        )
    )
    # closest_conf_idxs = torch.argmin(lig_rmsds + atom_rmsds, axis=-1)
    # lig_min_rmsds = torch.gather(lig_rmsds, dim=-1, index=closest_conf_idxs.unsqueeze(-1)).squeeze(-1)
    # atom_min_rmsds = torch.gather(atom_rmsds, dim=-1, index=closest_conf_idxs.unsqueeze(-1)).squeeze(-1)

    return {"lig_rmsds": lig_rmsds, "atom_rmsds": atom_rmsds}


def compute_energy_ratios(graph, lig_pred, atom_pred):
    pred_lig_bond_length, pred_lig_bond_energy, pred_lig_bond_force = get_bond_force(
        lig_pred,
        graph["ligand"].bond_index,
        graph["ligand"].bond_k,
        graph["ligand"].bond_r_0,
    )
    pred_atom_bond_length, pred_atom_bond_energy, pred_atom_bond_force = get_bond_force(
        atom_pred,
        graph["atom"].bond_index,
        graph["atom"].bond_k,
        graph["atom"].bond_r_0,
    )
    pred_lig_angle, pred_lig_angle_energy, pred_lig_angle_force = get_angle_force(
        lig_pred,
        graph["ligand"].angle_index,
        graph["ligand"].angle_k,
        graph["ligand"].angle_theta_0,
    )
    pred_atom_angle, pred_atom_angle_energy, pred_atom_angle_force = get_angle_force(
        atom_pred,
        graph["atom"].angle_index,
        graph["atom"].angle_k,
        graph["atom"].angle_theta_0,
    )
    (
        pred_lig_torsion,
        pred_lig_torsion_energy,
        pred_lig_torsion_force,
    ) = get_torsion_force(
        lig_pred,
        graph["ligand"].torsion_index,
        graph["ligand"].torsion_k,
        graph["ligand"].torsion_n,
        graph["ligand"].torsion_phi_0,
    )
    (
        pred_atom_torsion,
        pred_atom_torsion_energy,
        pred_atom_torsion_force,
    ) = get_torsion_force(
        atom_pred,
        graph["atom"].torsion_index,
        graph["atom"].torsion_k,
        graph["atom"].torsion_n,
        graph["atom"].torsion_phi_0,
    )
    energy_ratio_dict = {}
    for key in ["src", "tgt"]:
        if key == "src":
            ref_lig_pos = graph["ligand"].src_pos.swapaxes(0, 1)
            ref_atom_pos = graph["atom"].src_pos.swapaxes(0, 1)
        else:
            ref_lig_pos = graph["ligand"].tgt_pos
            ref_atom_pos = graph["atom"].tgt_pos
        ref_lig_bond_length, ref_lig_bond_energy, ref_lig_bond_force = get_bond_force(
            ref_lig_pos,
            graph["ligand"].bond_index,
            graph["ligand"].bond_k,
            graph["ligand"].bond_r_0,
        )
        (
            ref_atom_bond_length,
            ref_atom_bond_energy,
            ref_atom_bond_force,
        ) = get_bond_force(
            ref_atom_pos,
            graph["atom"].bond_index,
            graph["atom"].bond_k,
            graph["atom"].bond_r_0,
        )
        ref_lig_angle, ref_lig_angle_energy, ref_lig_angle_force = get_angle_force(
            ref_lig_pos,
            graph["ligand"].angle_index,
            graph["ligand"].angle_k,
            graph["ligand"].angle_theta_0,
        )
        ref_atom_angle, ref_atom_angle_energy, ref_atom_angle_force = get_angle_force(
            ref_atom_pos,
            graph["atom"].angle_index,
            graph["atom"].angle_k,
            graph["atom"].angle_theta_0,
        )
        (
            ref_lig_torsion,
            ref_lig_torsion_energy,
            ref_lig_torsion_force,
        ) = get_torsion_force(
            ref_lig_pos,
            graph["ligand"].torsion_index,
            graph["ligand"].torsion_k,
            graph["ligand"].torsion_n,
            graph["ligand"].torsion_phi_0,
        )
        (
            ref_atom_torsion,
            ref_atom_torsion_energy,
            ref_atom_torsion_force,
        ) = get_torsion_force(
            ref_atom_pos,
            graph["atom"].torsion_index,
            graph["atom"].torsion_k,
            graph["atom"].torsion_n,
            graph["atom"].torsion_phi_0,
        )
        lig_bond_energy_ratio = pred_lig_bond_energy / ref_lig_bond_energy
        atom_bond_energy_ratio = pred_atom_bond_energy / ref_atom_bond_energy
        lig_angle_energy_ratio = pred_lig_angle_energy / ref_lig_angle_energy
        atom_angle_energy_ratio = pred_atom_angle_energy / ref_atom_angle_energy
        lig_torsion_energy_ratio = pred_lig_torsion_energy / ref_lig_torsion_energy
        atom_torsion_energy_ratio = pred_atom_torsion_energy / ref_atom_torsion_energy
        energy_ratio_dict.update(
            {
                f"{key}_lig_bond_energy_ratio": lig_bond_energy_ratio,
                f"{key}_atom_bond_energy_ratio": atom_bond_energy_ratio,
                f"{key}_lig_angle_energy_ratio": lig_angle_energy_ratio,
                f"{key}_atom_angle_energy_ratio": atom_angle_energy_ratio,
                f"{key}_lig_torsion_energy_ratio": lig_torsion_energy_ratio,
                f"{key}_atom_torsion_energy_ratio": atom_torsion_energy_ratio,
            }
        )
    return energy_ratio_dict


def aggregate_metric_dicts(metric_dicts):
    aggregated_metrict_dict = defaultdict(list)
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            aggregated_metrict_dict[key].append(value)
    aggregated_metrict_dict = {
        "avg_" + key: sum(values) / len(values)
        for key, values in aggregated_metrict_dict.items()
    }
    return aggregated_metrict_dict
