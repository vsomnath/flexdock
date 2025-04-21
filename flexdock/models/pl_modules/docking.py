import copy
from typing import Any
from functools import partial
import contextlib

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
import torch
import timeit
import logging

from flexdock.data.conformers.protein import scRMSD

# from flexdock.data.transforms.docking.bb_priors import construct_bb_prior
from flexdock.geometry.ops import rigid_transform_kabsch_numpy

from flexdock.models.networks import get_model
from flexdock.models.loss.docking import FlexDockLoss
from flexdock.models.optim.lr_schedulers import AlphaFoldLRScheduler
from flexdock.models.optim.ema import ExponentialMovingAverage
from flexdock.metrics.docking import pli_lddt_score, CustomMeanMetric

from flexdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl
from flexdock.sampling.docking import sampling, sampling_fast


class FlexDockModule(pl.LightningModule):
    def __init__(
        self, model_cfg, sigma_cfg, training_cfg, sampler_cfg, loss_cfg=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.sigma_cfg = sigma_cfg
        self.sampler_cfg = sampler_cfg
        self.training_cfg = training_cfg

        self.t_to_sigma = t_to_sigma = partial(t_to_sigma_compl, args=sigma_cfg)
        self.model = get_model(
            args=model_cfg,
            t_to_sigma=t_to_sigma,
            confidence_mode=False,
            device=self.device,  # PyL doesn't need a device argument
        )
        if loss_cfg is not None:
            self.loss = FlexDockLoss(args=loss_cfg, t_to_sigma=t_to_sigma)
        # Move this into sampler
        # self.bb_prior = construct_bb_prior(args)
        self.bb_prior = None
        self.ema = None
        self.setup_metrics()

    def setup_metrics(self):
        """Setup torchmetrics"""
        self.metrics_dict = torch.nn.ModuleDict()
        for metric in [
            "rmsds_lt1",
            "rmsds_lt2",
            "rmsds_lt5",
            "bb_rmsds_lt1",
            "bb_rmsds_lt2",
            "bb_rmsds_lt05",
            "aa_rmsds_lt1",
            "aa_rmsds_lt2",
            "aa_rmsds_lt05",
            "pli_lddt",
        ]:
            # nan vals possible for certain metrics
            self.metrics_dict[metric] = CustomMeanMetric()

    def training_step(self, batch, batch_idx):
        predictions = self.general_step_with_oom(batch, batch_idx)
        loss, loss_breakdown = self.loss(predictions, batch, apply_mean=True)

        for key, value in loss_breakdown.items():
            skip_logging_condn = "tor" in key and torch.isnan(value)
            value = 0.0 if skip_logging_condn else value
            batch_size = 0 if skip_logging_condn else batch.num_graphs
            self.log(
                f"train_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.stage = "val"

        # Validation dataloader
        if dataloader_idx == 0:
            with torch.no_grad():
                predictions = self.general_step_with_oom(batch, batch_idx)

            _, loss_breakdown = self.loss(predictions, batch, apply_mean=True)
            for key, value in loss_breakdown.items():
                skip_logging_condn = "tor" in key and torch.isnan(value)
                value = 0.0 if skip_logging_condn else value
                batch_size = 0 if skip_logging_condn else batch.num_graphs
                self.log(
                    f"val_{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )

        # Inference data loader
        elif dataloader_idx == 1:
            val_inference_freq = self.training_cfg.val_inference_freq
            if (
                val_inference_freq is not None
                and (self.trainer.current_epoch + 1) % val_inference_freq == 0
            ):
                self.stage = "valinf"
                # Need to specify to use apo positions if not using flexible sidechain or flexible backbone.
                # During training this is done via ProteinTransform but not applied during inference
                if (
                    not self.model_cfg.flexible_backbone
                    and not self.model_cfg.flexible_sidechains
                ):
                    batch["atom"].pos = batch["atom"].orig_aligned_apo_pos
                    batch["receptor"].pos = batch["atom"].pos[batch["atom"].ca_mask]

                # This is needed for bf16-mixed, since torch.linalg.svd is not supported in bf16
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    inf_predictions = self.run_inference(batch, batch_idx)
                    self._compute_inference_metrics(
                        batch=batch, predictions=inf_predictions, batch_idx=batch_idx
                    )

    def general_step_with_oom(self, batch, batch_idx):
        # Runs model step but with a OOM check and cleanup
        try:
            return self.model(batch, fast_updates=True)

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

    @torch.no_grad()
    def run_inference(self, batch, batch_idx):
        schedules = sampling.get_schedules(
            inference_steps=self.sampler_cfg.inference_steps,
            bb_tr_bridge_alpha=self.sampler_cfg.bb_tr_bridge_alpha,
            bb_rot_bridge_alpha=self.sampler_cfg.bb_rot_bridge_alpha,
            sidechain_tor_bridge=self.sampler_cfg.sidechain_tor_bridge,
            sc_tor_bridge_alpha=self.sampler_cfg.sc_tor_bridge_alpha,
            inf_sched_alpha=1,
            inf_sched_beta=1,
            sigma_schedule="expbeta",
        )
        data_list = [copy.deepcopy(batch.to("cpu"))]

        randomize_fn = sampling_fast.randomize_position_inf
        sampling_fn = sampling_fast.sampling

        randomize_fn(
            data_list=data_list,
            no_torsion=self.sampler_cfg.no_torsion,
            no_random=False,
            tr_sigma_max=self.sigma_cfg.tr_sigma_max,
            flexible_sidechains=self.model_cfg.flexible_sidechains,
            flexible_backbone=self.model_cfg.flexible_backbone,
            sidechain_tor_bridge=self.model_cfg.sidechain_tor_bridge,
            use_bb_orientation_feats=self.model_cfg.use_bb_orientation_feats,
            prior=self.bb_prior,
        )

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list is None:
            try:
                predictions_list, confidences = sampling_fn(
                    data_list=data_list,
                    model=self.model,
                    inference_steps=self.sampler_cfg.inference_steps,
                    schedules=schedules,
                    sidechain_tor_bridge=self.model_cfg.sidechain_tor_bridge,
                    device=self.device,
                    t_to_sigma=self.t_to_sigma,
                    model_args=self.sampler_cfg,
                    no_final_step_noise=True,
                    debug_backbone=False,
                    debug_sidechain=False,
                    use_bb_orientation_feats=self.model_cfg.use_bb_orientation_feats,
                )
            except Exception as e:
                if "failed to converge" in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logging.warning(
                            "| WARNING: SVD failed to converge 5 times - skipping the complex"
                        )
                        break
                    logging.warning(
                        "| WARNING: SVD failed to converge - trying again with a new sample"
                    )
                elif "out of bounds" in str(e):
                    failed_convergence_counter += 1
                    logging.warning(f"Failing: DEBUG | WARNING: {str(e)}")
                    if failed_convergence_counter > 5:
                        logging.info(
                            "DEBUG: `index out of bound` for more than 5 times - skipping the complex"
                        )
                        break
                else:
                    failed_convergence_counter += 1
                    logging.warning(f"Failing: DEBUG | WARNING: {str(e)}")
                    if failed_convergence_counter > 5:
                        logging.warning(
                            f"DEBUG: Unexpected error `{e}` for more than 5 times - skipping the complex"
                        )
                        break

            if failed_convergence_counter > 5:
                pass

        return predictions_list

    def _compute_inference_metrics(self, batch, predictions, batch_idx=None):
        if predictions is None:
            logging.info(
                f"Skipping inference metrics for batch_idx={batch_idx} since predictions=None"
            )
            return

        if self.model_cfg.no_torsion:
            orig_center = batch.original_center.cpu().numpy()
            centered_lig_pos = batch["ligand"].pos.cpu().numpy()
            batch["ligand"].orig_pos = centered_lig_pos + orig_center

        filterHs = torch.not_equal(predictions[0]["ligand"].x[:, 0], 0).cpu().numpy()

        if isinstance(batch["ligand"].orig_pos, list):
            batch["ligand"].orig_pos = batch["ligand"].orig_pos[0]
        if isinstance(batch["atom"].orig_holo_pos, list):
            batch["atom"].orig_holo_pos = batch["atom"].orig_holo_pos[0]

        orig_pos = batch["ligand"].orig_pos
        if isinstance(orig_pos, torch.Tensor):
            orig_pos = orig_pos.cpu().numpy()

        ligand_pos = []
        atom_pos = []
        ligand_pos_before = []
        orig_atom_pos = batch["atom"].orig_holo_pos.numpy()

        try:
            nearby_atoms = batch["atom"].nearby_atoms.cpu().numpy()
        except Exception:
            nearby_atoms = np.full(len(batch["atom"].pos), True)

        for complex_graph in predictions:
            atom_p = complex_graph["atom"].pos.cpu().numpy()
            try:
                R, t, _ = rigid_transform_kabsch_numpy(
                    orig_atom_pos[nearby_atoms], atom_p[nearby_atoms]
                )

                ligand_p = complex_graph["ligand"].pos.cpu().numpy()[filterHs]
                ligand_pos_before.append(ligand_p)

                atom_p = (R @ atom_p.T).T + t
                ligand_p = (R @ ligand_p.T).T + t
                atom_pos.append(atom_p)
                ligand_pos.append(ligand_p)

            except Exception as e:
                if "did not converge" in str(e):
                    logging.error(
                        "DEBUG | WARNING: numpy.linalg.LinAlgError: SVD did not converge"
                    )
                else:
                    raise e
        # to skip if no metric is computed
        if len(ligand_pos) == 0 or len(atom_pos) == 0:
            logging.debug("No ligand or proteina atoms found. No metric is computed!")
            return

        atom_pos = np.asarray(atom_pos)
        ligand_pos = np.asarray(ligand_pos)
        ligand_pos_before = np.asarray(ligand_pos_before)

        # TODO compute RMSD alignment of the overall position of atoms, ligand and receptors with the original complex

        orig_ligand_pos = np.expand_dims(batch["ligand"].orig_pos[filterHs], axis=0)

        orig_center = batch.original_center
        if isinstance(orig_center, list):
            orig_center = orig_center[0]

        if len(orig_center.shape) == 1:
            orig_center = orig_center[None, None, :]
        elif len(orig_center.shape) == 2:
            orig_center = orig_center[None]

        if isinstance(orig_center, torch.Tensor):
            orig_center = orig_center.cpu().numpy()

        rmsd = np.sqrt(
            ((ligand_pos + orig_center - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1)
        )
        self.metrics_dict["rmsds_lt1"]([100 * (rmsd < 1.0)])
        self.metrics_dict["rmsds_lt2"]([100 * (rmsd < 2.0)])
        self.metrics_dict["rmsds_lt5"]([100 * (rmsd < 5.0)])

        # We log this regardless
        calpha_mask = batch["atom"].ca_mask.numpy()
        calpha_pred_atoms = atom_pos[:, calpha_mask]
        calpha_holo_atoms = orig_atom_pos[None, calpha_mask]
        calpha_rmsd = np.sqrt(
            ((calpha_pred_atoms - calpha_holo_atoms) ** 2).sum(axis=2).mean(axis=1)
        )

        self.metrics_dict["bb_rmsds_lt2"]([100 * (calpha_rmsd < 2.0)])
        self.metrics_dict["bb_rmsds_lt1"]([100 * (calpha_rmsd < 1.0)])
        self.metrics_dict["bb_rmsds_lt05"]([100 * (calpha_rmsd < 0.5)])

        # We log this regardless
        aa_rmsd = scRMSD(nearby_atoms, atom_pos[0], orig_atom_pos)
        self.metrics_dict["aa_rmsds_lt2"]([100 * (aa_rmsd < 2.0)])
        self.metrics_dict["aa_rmsds_lt1"]([100 * (aa_rmsd < 1.0)])
        self.metrics_dict["aa_rmsds_lt05"]([100 * (aa_rmsd < 0.5)])

        # log pli-lddt score
        try:
            pli_lddt = pli_lddt_score(
                rec_coords_predicted=torch.from_numpy(atom_pos).float(),
                lig_coords_predicted=torch.from_numpy(ligand_pos).float(),
                rec_coords_true=torch.from_numpy(orig_atom_pos).float(),
                lig_coords_true=torch.from_numpy(orig_ligand_pos - orig_center).float(),
            )
        except ValueError as e:
            logging.error(
                f"Assigning 0 to pli_lddt metric for {batch['name']} due to error: {e}"
            )
            pli_lddt = torch.tensor([0.0])
        self.metrics_dict["pli_lddt"](100 * pli_lddt)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        self.ema.update(self.model.parameters())

    def on_validation_start(self):
        self.ema.store(self.model.parameters())
        if self.training_cfg.use_ema:
            rank_zero_info("Copying EMA parameters into model before validation")
            self.ema.copy_to(self.model.parameters())

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        if self.training_cfg.use_ema:
            checkpoint["ema_weights"] = copy.deepcopy(self.ema_state_dict)
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        if self.training_cfg.use_ema and "ema" in checkpoint:
            rank_zero_info("Loading EMA from checkpoint")
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(), decay=self.training_cfg.ema_rate
            )
            self.ema.load_state_dict(state_dict=checkpoint["ema"], device=self.device)
            # Just a sanity check that numbers look sensible
            rank_zero_info(
                f"decay={self.ema.decay}, num_updates={self.ema.num_updates}"
            )

    def on_train_start(self):
        if self.ema is None:
            rank_zero_info("Initializing EMA")
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(), decay=self.training_cfg.ema_rate
            )
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = timeit.default_timer()

        # Subsample complexes according to cluster ids if provided
        self.trainer.train_dataloader.dataset.subsample_clusters()

    def on_validation_epoch_start(self) -> None:
        self.validation_epoch_start_time = timeit.default_timer()
        # Subsample complexes according to cluster ids if provided
        self.trainer.val_dataloaders[0].dataset.subsample_clusters()

    def on_validation_epoch_end(self) -> None:
        # log metrics at epoch level
        for key, metric in self.metrics_dict.items():
            # if called
            if len(metric.values) > 0:
                self.log(f"valinf_{key}", metric.compute(), sync_dist=True)
                metric.reset()

        if self.training_cfg.use_ema:
            self.ema_state_dict = copy.deepcopy(self.model.state_dict())
            rank_zero_info("Restoring Model parameters...")
            self.ema.restore(self.model.parameters())

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    f"| WARNING: ran OOM error, skipping batch. Exception: {str(e)}"
                )
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def on_before_optimizer_step(self, optimizer):
        if self.training_cfg.check_unused_params:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logging.info(f"gradients were None for {name}")

        if self.training_cfg.check_nan_grads:
            had_nan_grads = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    had_nan_grads = True
                    logging.info(f"gradients were nan for {name}")
            if had_nan_grads and self.training_cfg.except_on_nan_grads:
                raise Exception(
                    "There were nan gradients and except_on_nan_grads was set to True"
                )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.training_cfg.skip_nan_grad_updates:
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    logging.info(
                        f"Gradients were nan for {name}, and skip_nan_grad_updates was enabled."
                        " Zeroing grad for this batch."
                    )
                    self.optimizer_zero_grad(epoch, batch_idx, optimizer)
                    break

        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        optimizer_cls = (
            torch.optim.AdamW
            if self.training_cfg.adamw == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.training_cfg.lr),
            weight_decay=self.training_cfg.w_decay,
        )

        scheduler = None
        if self.training_cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.training_cfg.inference_earlystop_goal.split(",")[0]
                if self.training_cfg.val_inference_freq is not None
                else "min",
                factor=0.7,
                patience=self.training_cfg.scheduler_patience,
                min_lr=float(self.training_cfg.lr) / 100,
            )
        elif self.training_cfg.scheduler == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=250,  # {800, 500, 250, 100}
                eta_min=1e-7,  # default: 0
            )
        elif self.training_cfg.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.99  # first choice tried: 0.95
            )
        elif self.training_cfg.scheduler == "AlphaFoldLRScheduler":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                last_epoch=-1,
                warmup_no_steps=3,  # PLINDER: 1  ; PDBBind: 3
                start_decay_after_n_steps=125,  # PLINDER: 50  ;  PDBBind: 125
                decay_every_n_steps=2,  # PLINDER: 1  ; PDBBind: 2
                decay_factor=0.99,  # PLINDER: 0.98  ;  PDBBind: 0.99
            )
        else:
            rank_zero_info("No scheduler")
            scheduler = None

        optim_dict = {"optimizer": optimizer}

        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.training_cfg.inference_earlystop_metric.split(",")[0]
                if self.training_cfg.val_inference_freq is not None
                else "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
                "name": self.training_cfg.scheduler,
            }

        return optim_dict


def _make_model_backward_compatible():
    # Restructured codebase a bit, so need this hack to ensure things load correctly
    import sys
    from flexdock.sampling.docking import diffusion

    sys.modules["flexdock.utils.diffusion"] = diffusion
    sys.modules["flexdock.components.docking.utils.diffusion"] = diffusion
    return


def load_pretrained_model(
    cfg, ckpt_file, use_ema_weights: bool = False, freeze: bool = False
):
    model = FlexDockModule(
        model_cfg=cfg.model,
        sigma_cfg=cfg.sigma,
        training_cfg=cfg.training,
        sampler_cfg=cfg.sampler,
        loss_cfg=cfg.get("loss", None),
    )
    _make_model_backward_compatible()
    loaded_weights = torch.load(ckpt_file, map_location="cpu")

    state_dict_key = "ema_weights" if use_ema_weights else "state_dict"
    state_dict = loaded_weights[state_dict_key]

    # Add a model. prefix to the keys
    keys = list(state_dict.keys())
    first_key = keys[0]
    if not first_key.startswith("model."):
        for key in keys:
            state_dict[f"model.{key}"] = state_dict.pop(key)

    # Replace batch_norm keys with norm
    keys = list(state_dict.keys())
    for key in keys:
        if "batch_norm" in key:
            new_key = key.replace("batch_norm", "norm")
            state_dict[new_key] = state_dict.pop(key)

    for key, value in state_dict.items():
        if value.isnan().any():
            raise ValueError("Values cannot be nan")

    # if "fsdp" in strategy_type:
    #    strat_context = FSDP.summon_full_params(model, offload_to_cpu=True)
    strat_context = contextlib.nullcontext()
    with strat_context:
        model.load_state_dict(state_dict, strict=True)
        if freeze:
            model.freeze()

    return model
