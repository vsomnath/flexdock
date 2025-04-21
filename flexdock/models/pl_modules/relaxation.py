from collections import defaultdict
import copy
import logging
import math
import contextlib
from typing import Any

import numpy as np
import lightning.pytorch as pl
import lightning.pytorch.utilities as pl_utils
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_sum
import wandb

from flexdock.geometry.ops import rigid_transform_kabsch, rigid_transform_kabsch_batch
from flexdock.sampling.relaxation.sampling import sampling_on_confs
from flexdock.models.networks import get_model
from flexdock.models.optim.ema import ExponentialMovingAverage

from flexdock.metrics.relaxation import (
    compute_posebusters_geometry_metrics,
    compute_posebusters_interaction_metrics,
    compute_ligand_alignment_metrics,
    compute_protein_alignment_metrics,
    compute_lddt_pli,
    construct_metric_dict,
    get_bond_lengths,
)

from flexdock.data.modules.training.relaxation import IterableConfDataset


logging.basicConfig(filename="logging.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([log_device[key] for log_device in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            # if "valinf" in key:
            #     values = np.array(log[key]).squeeze()
            #     if len(values):
            #         out[key] = (values.sum()) / len(values)
            # else:
            out[key] = np.mean(log[key])
        except Exception:
            pass
    return out


class RelaxFlowModule(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.model = get_model(
            args=args,
            t_to_sigma=None,
            relaxation=True,
            device=self.device,  # PyL doesn't need a device argument
        )
        self.ema = None

        self._log = defaultdict(list)

    def relaxflow_log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy().tolist()

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        log = self._log
        log[self.stage + "/" + key].extend(data)

    # Done
    def training_step(self, batch, batch_idx, stage: str = "train"):
        self.stage = stage

        pred = self.general_step_with_oom(batch, batch_idx)

        if pred is None:
            return None
        lig_pred, atom_pred = pred

        loss, loss_breakdown = self.loss(lig_pred, atom_pred, batch)

        for key, value in loss_breakdown.items():
            self.relaxflow_log(key, value)

        return loss

    # Done
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.stage = "val"

        # Validation dataloader
        if dataloader_idx == 0:
            with torch.no_grad():
                pred = self.general_step_with_oom(batch, batch_idx)
                if pred is None:
                    return None
                lig_pred, atom_pred = pred
            _, loss_breakdown = self.loss(lig_pred, atom_pred, batch)
            for key, value in loss_breakdown.items():
                self.relaxflow_log(key, value)

        # Inference data loader
        elif dataloader_idx == 1:
            val_inference_freq = self.args.val_inference_freq
            if (
                val_inference_freq is not None
                and (self.trainer.current_epoch + 1) % val_inference_freq == 0
            ):
                self.stage = "valinf"
                graph = batch[0]
                for inference_steps in self.args.inference_steps:
                    traj = self.relaxflow_inference(
                        graph, batch_idx, int(inference_steps)
                    )
                    if traj is None:
                        return None
                    lig_traj, atom_traj = traj
                    metrics = self._compute_inference_metrics(
                        lig_traj, atom_traj, graph, int(inference_steps)
                    )

                    for key, value in metrics.items():
                        self.relaxflow_log(f"steps_{inference_steps}/{key}", value)

    def relaxflow_inference(self, graph, graph_idx, inference_steps):
        graph["ligand"].pos = graph["ligand"].flexdock_pos
        graph["atom"].pos = graph["atom"].flexdock_pos
        graph["receptor"].pos = graph["atom"].pos[:, graph["atom"].ca_mask]
        graph.n_samples = graph["ligand"].pos.shape[0]
        graph.conf_idx = torch.arange(graph.n_samples)

        conf_dataset = IterableConfDataset(graph)
        conf_loader = DataLoader(conf_dataset, batch_size=self.args.batch_size)
        return sampling_on_confs(
            self.model,
            conf_loader,
            inference_steps,
            x_zero_pred=self.args.x_zero_pred,
            device=self.device,
            save_traj=True,
        )

    def _compute_inference_metrics(self, lig_traj, atom_traj, graph, inference_steps):
        conf_metrics, metrics = {}, {}

        lig_x_zero_pred = lig_traj[:-1] + (
            lig_traj[1:] - lig_traj[:-1]
        ) * inference_steps * (
            1
            - torch.from_numpy(np.linspace(0, 1, inference_steps + 1))[
                :-1, None, None, None
            ]
        ).float().to(
            lig_traj.device
        )
        atom_x_zero_pred = atom_traj[:-1] + (
            atom_traj[1:] - atom_traj[:-1]
        ) * inference_steps * (
            1
            - torch.from_numpy(np.linspace(0, 1, inference_steps + 1))[
                :-1, None, None, None
            ]
        ).float().to(
            atom_traj.device
        )

        lig_x_zero_pred = torch.cat((lig_traj[0, None], lig_x_zero_pred), dim=0)
        atom_x_zero_pred = torch.cat((atom_traj[0, None], atom_x_zero_pred), dim=0)

        R, tr = rigid_transform_kabsch(
            atom_x_zero_pred[:, :, graph["atom"].nearby_atom_mask],
            graph["atom"].orig_holo_pos[graph["atom"].nearby_atom_mask],
        )
        lig_x_zero_pred = lig_x_zero_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
        atom_x_zero_pred = atom_x_zero_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)

        filter_hs = graph["ligand"].x[:, 0] == 0
        if filter_hs.sum() > 0:
            print(graph.name)

        conf_metrics = {}
        conf_metrics.update(
            compute_ligand_alignment_metrics(
                lig_x_zero_pred,
                graph["ligand"].orig_pos,
                graph.mol,
                return_aligned_pos=False,
            )
        )
        conf_metrics.update(
            compute_protein_alignment_metrics(
                atom_x_zero_pred,
                graph["atom"].orig_holo_pos,
                graph["atom"].nearby_atom_mask,
                graph["atom"].ca_mask,
                graph["atom"].c_mask,
                graph["atom"].n_mask,
            )
        )
        conf_metrics.update(
            compute_lddt_pli(
                lig_x_zero_pred,
                graph["ligand"].orig_pos,
                atom_x_zero_pred,
                graph["atom"].orig_holo_pos,
                graph["atom", "receptor"].edge_index[1],
                inclusion_radius=6.0,
            )
        )

        posebusters_metrics = {}
        posebusters_metrics.update(
            compute_posebusters_geometry_metrics(
                lig_x_zero_pred,
                graph["ligand", "lig_edge", "ligand"].posebusters_edge_index,
                graph["ligand", "lig_edge", "ligand"].lower_bound,
                graph["ligand", "lig_edge", "ligand"].upper_bound,
                graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask,
                graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask,
            )
        )
        posebusters_metrics.update(
            compute_posebusters_interaction_metrics(
                lig_x_zero_pred,
                atom_x_zero_pred,
                graph["ligand"].vdw_radii,
                graph["atom"].vdw_radii,
            )
        )

        flexdock_lig_rmsds = torch.mean(
            torch.sum((lig_x_zero_pred[0] - graph["ligand"].orig_pos) ** 2, axis=-1),
            axis=-1,
        )
        flexdock_atom_rmsds = torch.mean(
            torch.sum(
                (atom_x_zero_pred[0] - graph["atom"].orig_holo_pos) ** 2, axis=-1
            ),
            axis=-1,
        )
        flexdock_rmsds = (flexdock_lig_rmsds + flexdock_atom_rmsds) / 2
        rmsd_mask = flexdock_rmsds < 2.0

        metrics = {}
        for name, vals in conf_metrics.items():
            if "loss" in name:
                upper_thresholds = None
                lower_thresholds = None
            elif "rmsd" in name:
                upper_thresholds = [5.0, 2.0, 1.0, 0.5]
                lower_thresholds = None
            elif name == "lig_tr_mags":
                upper_thresholds = [5.0, 2.0, 1.0, 0.5]
                lower_thresholds = None
            elif name == "lig_rot_mags":
                upper_thresholds = [1.571, 0.785, 0.393, 0.196]
                lower_thresholds = None
            elif "length" in name or "q" in name:
                upper_thresholds = [0.25, 0.2, 0.1, 0.05]
                lower_thresholds = None
            elif "angle" in name or "phi" in name or "theta" in name:
                upper_thresholds = [1.571, 0.785, 0.393, 0.196]
                lower_thresholds = None
            elif name == "lddt_pli":
                upper_thresholds = None
                lower_thresholds = [0.9, 0.69]
            metrics.update(
                construct_metric_dict(
                    name=name,
                    values=vals,
                    percentiles=[25, 50, 75],
                    upper_thresholds=upper_thresholds,
                    lower_thresholds=lower_thresholds,
                    rmsd_mask=rmsd_mask.cpu().numpy(),
                )
            )

        for name, values in posebusters_metrics.items():
            for step_idx, step_values in enumerate(values):
                entry_name = name
                if step_idx == 0:
                    entry_name = "initial_" + entry_name
                elif step_idx < values.shape[0] - 1:
                    entry_name = f"step_{step_idx - 1}_" + entry_name
                else:
                    entry_name = "final_" + entry_name
                metrics[entry_name] = step_values.mean()
                if rmsd_mask.sum() > 0:
                    metrics[entry_name + "_good"] = (
                        step_values * rmsd_mask.cpu().numpy()
                    ).sum() / rmsd_mask.cpu().numpy().sum()
                if (~rmsd_mask).sum() > 0:
                    metrics[entry_name + "_bad"] = (
                        step_values * (~rmsd_mask).cpu().numpy()
                    ).sum() / (~rmsd_mask.cpu().numpy()).sum()
        return metrics

    # Done
    def loss(self, lig_pred, atom_pred, batch):
        if self.args.x_zero_pred:
            lig_x_zero_pred = lig_pred
            atom_x_zero_pred = atom_pred
            lig_vec_pred = (
                lig_pred
                - batch["ligand"].pos
                - batch.original_center[batch["ligand"].batch]
            ) / (1 - batch["ligand"].node_t.unsqueeze(-1))
            atom_vec_pred = (
                atom_pred
                - batch["atom"].pos
                - batch.original_center[batch["atom"].batch]
            ) / (1 - batch["atom"].node_t.unsqueeze(-1))
        else:
            lig_x_zero_pred = batch["ligand"].pos + lig_pred * (
                1 - batch["ligand"].node_t.unsqueeze(-1)
            )
            atom_x_zero_pred = batch["atom"].pos + atom_pred * (
                1 - batch["atom"].node_t.unsqueeze(-1)
            )
            lig_vec_pred = lig_pred
            atom_vec_pred = atom_pred

        if self.args.align_pred:
            R, tr = rigid_transform_kabsch_batch(
                atom_x_zero_pred.detach()[batch["atom"].nearby_atom_mask],
                batch["atom"].orig_holo_pos[batch["atom"].nearby_atom_mask],
                batch["atom"].batch[batch["atom"].nearby_atom_mask],
            )
            lig_x_zero_pred = (
                lig_x_zero_pred.unsqueeze(-2)
                @ R[batch["ligand"].batch].swapaxes(-1, -2)
                + tr[batch["ligand"].batch].unsqueeze(-2)
            ).squeeze()
            atom_x_zero_pred = (
                atom_x_zero_pred.unsqueeze(-2) @ R[batch["atom"].batch].swapaxes(-1, -2)
                + tr[batch["atom"].batch].unsqueeze(-2)
            ).squeeze()

        if self.args.x_zero_loss:
            lig_batch_loss = scatter_mean(
                ((lig_x_zero_pred - batch["ligand"].orig_pos) ** 2).mean(axis=1),
                batch["ligand"].batch,
                0,
            )
            atom_batch_loss = scatter_mean(
                ((atom_x_zero_pred - batch["atom"].orig_holo_pos) ** 2).mean(axis=1),
                batch["atom"].batch,
                0,
            )
        else:
            lig_batch_loss = scatter_mean(
                ((lig_vec_pred - batch["ligand"].u) ** 2).mean(axis=1),
                batch["ligand"].batch,
                0,
            )
            atom_batch_loss = scatter_mean(
                ((atom_vec_pred - batch["atom"].u) ** 2).mean(axis=1),
                batch["atom"].batch,
                0,
            )

        batch_loss = (
            lig_batch_loss * self.args.ligand_loss_weight
            + atom_batch_loss * self.args.atom_loss_weight
        )
        batch_loss_norm = self.args.ligand_loss_weight + self.args.atom_loss_weight
        lig_loss = lig_batch_loss.mean()
        atom_loss = atom_batch_loss.mean()

        loss_breakdown = {
            "lig_loss": lig_loss.detach().cpu(),
            "atom_loss": atom_loss.detach().cpu(),
        }

        if self.args.posebusters_loss:
            lig_dists = torch.linalg.norm(
                lig_x_zero_pred[
                    batch["ligand", "lig_edge", "ligand"].posebusters_edge_index[1]
                ]
                - lig_x_zero_pred[
                    batch["ligand", "lig_edge", "ligand"].posebusters_edge_index[0]
                ],
                dim=-1,
            )
            batch_bond_loss = scatter_sum(
                torch.clip(
                    batch["ligand", "lig_edge", "ligand"].lower_bound
                    * (1.0 - self.args.bond_loss_buffer)
                    - lig_dists,
                    min=0,
                )[batch["ligand", "lig_edge", "ligand"].posebusters_bond_mask]
                + torch.clip(
                    lig_dists
                    - batch["ligand", "lig_edge", "ligand"].upper_bound
                    * (1.0 + self.args.bond_loss_buffer),
                    min=0,
                )[batch["ligand", "lig_edge", "ligand"].posebusters_bond_mask],
                batch["ligand"].batch[
                    batch["ligand", "lig_edge", "ligand"].posebusters_edge_index[
                        0, batch["ligand", "lig_edge", "ligand"].posebusters_bond_mask
                    ]
                ],
            )
            batch_angle_loss = scatter_sum(
                torch.clip(
                    batch["ligand", "lig_edge", "ligand"].lower_bound
                    * (1.0 - self.args.bond_loss_buffer)
                    - lig_dists,
                    min=0,
                )[batch["ligand", "lig_edge", "ligand"].posebusters_angle_mask]
                + torch.clip(
                    lig_dists
                    - batch["ligand", "lig_edge", "ligand"].upper_bound
                    * (1.0 + self.args.bond_loss_buffer),
                    min=0,
                )[batch["ligand", "lig_edge", "ligand"].posebusters_angle_mask],
                batch["ligand"].batch[
                    batch["ligand", "lig_edge", "ligand"].posebusters_edge_index[
                        0, batch["ligand", "lig_edge", "ligand"].posebusters_angle_mask
                    ]
                ],
            )
            batch_steric_loss = scatter_sum(
                torch.clip(
                    batch["ligand", "lig_edge", "ligand"].lower_bound
                    * (1.0 - self.args.steric_loss_buffer)
                    - lig_dists,
                    min=0,
                )[
                    ~(
                        batch["ligand", "lig_edge", "ligand"].posebusters_bond_mask
                        + batch["ligand", "lig_edge", "ligand"].posebusters_angle_mask
                    )
                ],
                batch["ligand"].batch[
                    batch["ligand", "lig_edge", "ligand"].posebusters_edge_index[
                        0,
                        ~(
                            batch["ligand", "lig_edge", "ligand"].posebusters_bond_mask
                            + batch[
                                "ligand", "lig_edge", "ligand"
                            ].posebusters_angle_mask
                        ),
                    ]
                ],
            )
            if self.args.posebusters_loss_cutoff is not None:
                posebusters_loss_weight = (
                    self.args.posebusters_loss_weight
                    * (batch.complex_t >= self.args.posebusters_loss_cutoff).float()
                )
            elif self.args.posebusters_loss_alpha is not None:
                posebusters_loss_weight = (
                    self.args.posebusters_loss_weight
                    * (
                        torch.exp(batch.complex_t * self.args.posebusters_loss_alpha)
                        - 1
                    )
                    / (math.exp(self.args.posebusters_loss_alpha) - 1)
                )
            else:
                posebusters_loss_weight = 1.0

            batch_bond_loss *= posebusters_loss_weight
            batch_angle_loss *= posebusters_loss_weight
            batch_steric_loss *= posebusters_loss_weight
            batch_loss += batch_bond_loss * posebusters_loss_weight
            batch_loss += batch_angle_loss * posebusters_loss_weight
            batch_loss += batch_steric_loss * posebusters_loss_weight
            batch_loss_norm += 3 * posebusters_loss_weight
            bond_loss = batch_bond_loss.mean()
            angle_loss = batch_angle_loss.mean()
            steric_loss = batch_steric_loss.mean()
            loss_breakdown["bond_loss"] = bond_loss.detach().cpu()
            loss_breakdown["angle_loss"] = angle_loss.detach().cpu()
            loss_breakdown["steric_loss"] = steric_loss.detach().cpu()

        if self.args.crystal_loss:
            bond_lengths = get_bond_lengths(
                lig_x_zero_pred, batch["ligand", "ligand"].edge_index[:, ::2]
            )
            ref_bond_lengths = get_bond_lengths(
                batch["ligand"].orig_pos, batch["ligand", "ligand"].edge_index[:, ::2]
            )
            bond_batch_loss = scatter_mean(
                torch.abs(bond_lengths - ref_bond_lengths),
                batch["ligand"].batch[batch["ligand", "ligand"].edge_index[0, ::2]],
                0,
            )
            if self.args.crystal_loss_cutoff is not None:
                crystal_loss_weight = (
                    self.args.crystal_loss_weight
                    * (batch.complex_t >= self.args.crystal_loss_cutoff).float()
                )
            elif self.args.crystal_loss_alpha is not None:
                crystal_loss_weight = (
                    self.args.crystal_loss_weight
                    * (torch.exp(batch.complex_t * self.args.crystal_loss_alpha) - 1)
                    / (math.exp(self.args.crystal_loss_alpha) - 1)
                )
                batch_bond_loss *= crystal_loss_weight
            else:
                crystal_loss_weight = 1.0

            batch_loss += bond_batch_loss * crystal_loss_weight
            batch_loss_norm += crystal_loss_weight
            bond_loss = bond_batch_loss.mean()
            loss_breakdown["bond_loss"] = bond_loss.detach().cpu()

        if self.args.overlap_loss:
            lig_dense_pos, lig_dense_mask = to_dense_batch(
                lig_x_zero_pred, batch["ligand"].batch
            )
            atom_dense_pos, atom_dense_mask = to_dense_batch(
                atom_x_zero_pred, batch["atom"].batch
            )
            lig_dense_radii, _ = to_dense_batch(
                batch["ligand"].vdw_radii, batch["ligand"].batch
            )
            atom_dense_radii, _ = to_dense_batch(
                batch["atom"].vdw_radii, batch["atom"].batch
            )
            dists = torch.linalg.norm(
                lig_dense_pos.unsqueeze(2) - atom_dense_pos.unsqueeze(1), dim=-1
            )
            vdw_overlaps = torch.clip(
                lig_dense_radii.unsqueeze(2) + atom_dense_radii.unsqueeze(1) - dists,
                min=0,
            )
            pair_mask = lig_dense_mask.unsqueeze(2) * atom_dense_mask.unsqueeze(1)
            vdw_overlaps[~pair_mask] = torch.nan
            overlap_batch_loss = torch.nansum(
                torch.clip(vdw_overlaps - self.args.overlap_loss_buffer, min=0),
                axis=(-1, -2),
            )
            if self.args.overlap_loss_cutoff is not None:
                overlap_loss_weight = (
                    self.args.overlap_loss_weight
                    * (batch.complex_t >= self.args.overlap_loss_cutoff).float()
                )
            elif self.args.overlap_loss_alpha is not None:
                overlap_loss_weight = (
                    self.args.overlap_loss_weight
                    * (torch.exp(batch.complex_t * self.args.overlap_loss_alpha) - 1)
                    / (math.exp(self.args.overlap_loss_alpha) - 1)
                )
            else:
                overlap_loss_weight = 1.0

            batch_loss += overlap_batch_loss * overlap_loss_weight
            batch_loss_norm += overlap_loss_weight
            overlap_loss = overlap_batch_loss.mean()
            loss_breakdown["overlap_loss"] = overlap_loss.detach().cpu()

        loss = batch_loss.mean()  # (batch_loss / batch_loss_norm).mean()
        loss_breakdown["loss"] = (loss.detach().cpu(),)

        return loss, loss_breakdown

    # Done
    def general_step_with_oom(self, batch, batch_idx):
        if batch is None:
            return

        # Runs model step but with a OOM check and cleanup
        try:
            return self.model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("| WARNING: ran out of memory, skipping batch")
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

            elif "Input mismatch" in str(e):
                logging.error("| WARNING: weird torch_cluster error, skipping batch")
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        self.ema.update(self.model.parameters())

    def on_validation_start(self):
        self.ema.store(self.model.parameters())
        if self.args.use_ema:
            pl_utils.rank_zero_info(
                "Copying EMA parameters into model before validation"
            )
            self.ema.copy_to(self.model.parameters())

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        if self.args.use_ema:
            checkpoint["ema_weights"] = copy.deepcopy(self.model.state_dict())
        checkpoint["ema"] = self.ema.state_dict()

    def on_train_start(self):
        if self.ema is None:
            pl_utils.rank_zero_info("Initializing EMA")
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(), decay=self.args.ema_rate
            )

    def on_train_epoch_start(self) -> None:
        if self.args.use_ema:
            pl_utils.rank_zero_info(
                "Restoring model parameters from EMA before epoch training"
            )
            self.ema.restore(self.model.parameters())

        if (self.trainer.current_epoch + 1) % 5 == 0 and self.trainer.is_global_zero:
            pl_utils.rank_zero_info("Run name: ", self.args.run_name)

    def on_train_epoch_end(self):
        log = {
            key: self._log[key] for key in self._log if "train" in key or "val" in key
        }
        log_inference_values = (
            self.args.val_inference_freq is not None
            and (self.trainer.current_epoch + 1) % self.args.val_inference_freq == 0
        )

        if log_inference_values:
            log.update({key: self._log[key] for key in self._log if "valinf" in key})
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        mean_log.update(
            {"epoch": self.trainer.current_epoch, "step": self.trainer.global_step}
        )

        # We log the mean values to all trainers in DDP for appropriate checkpointing checks
        # Even though checkpoints are only saved by rank-0 trainers, the metrics are monitored on all
        tensor_metrics = {
            key: torch.tensor(metric).float() for key, metric in mean_log.items()
        }
        self.log_dict(tensor_metrics, batch_size=1)

        if self.trainer.is_global_zero:
            if self.args.wandb:
                wandb.log(mean_log)

            print("Epoch completed. Printing metrics...")
            print_str = f"Epoch {self.trainer.current_epoch}:"
            print_str += f" Training loss: {mean_log['train/loss']:.4f}"

            for key in ["lig", "atom"]:
                key_to_check = f"train/{key}_loss"
                if key_to_check in mean_log:
                    print_str += f" {key} {mean_log[key_to_check]:.4f}"

            print(print_str, flush=True)

            print_str = f"Epoch {self.trainer.current_epoch}:"
            print_str += f" Validation loss: {mean_log['val/loss']:.4f}"

            for key in ["lig", "atom", "bond"]:
                key_to_check = f"val/{key}_loss"
                if key_to_check in mean_log:
                    print_str += f" {key} {mean_log[key_to_check]:.4f}"

            print(print_str, flush=True)

        # Print newline
        print(flush=True)

        for key in list(log.keys()):
            del self._log[key]

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "| WARNING: ran OOM error, skipping batch. Exception:", str(e)
                )
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def on_before_optimizer_step(self, optimizer):
        if "check_unused_params" in self.args:
            if self.args.check_unused_params:
                for name, p in self.model.named_parameters():
                    if p.grad is None:
                        logging.info(f"gradients were None for {name}")

        if self.args.check_nan_grads:
            had_nan_grads = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    had_nan_grads = True
                    logging.info(f"gradients were nan for {name}")
            if had_nan_grads and self.args.except_on_nan_grads:
                raise Exception(
                    "There were nan gradients and except_on_nan_grads was set to True"
                )

    def configure_optimizers(self):
        optimizer_cls = (
            torch.optim.AdamW if self.args.adamw == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.args.lr),
            weight_decay=self.args.w_decay,
        )

        scheduler = None

        if self.args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.args.inference_earlystop_goal
                if self.args.val_inference_freq is not None
                else "min",
                factor=0.7,
                patience=self.args.scheduler_patience,
                min_lr=float(self.args.lr) / 100,
            )
        else:
            pl_utils.rank_zero_info("No scheduler")
            scheduler = None

        optim_dict = {"optimizer": optimizer}

        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.args.inference_earlystop_metric
                if self.args.val_inference_freq is not None
                else "val/loss",
                "interval": "epoch",
                "frequency": self.args.val_inference_freq or 1,
            }

        return optim_dict


def load_pretrained_model(model_args, ckpt_file, freeze: bool = False):
    model = RelaxFlowModule(args=model_args)
    loaded_weights = torch.load(ckpt_file, map_location="cpu")
    state_dict = loaded_weights["state_dict"]

    keys = list(state_dict.keys())
    for key in keys:
        if "batch_norm" in key:
            new_key = key.replace("batch_norm", "norm")
            state_dict[new_key] = state_dict.pop(key)

    strat_context = contextlib.nullcontext()
    with strat_context:
        model.load_state_dict(state_dict, strict=True)
        if freeze:
            model.freeze()
    return model
