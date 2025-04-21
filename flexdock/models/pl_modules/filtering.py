import contextlib
from collections import defaultdict
from functools import partial
from typing import Any
import logging

import numpy as np
import lightning.pytorch as pl
import lightning.pytorch.utilities as pl_utils
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import wandb

from flexdock.models.networks import get_model
from flexdock.models.loss.filtering import FilteringLoss
from flexdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl


def gather_log_original(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_tensor_size(tensor):
    # Utility function to get the size of the tensor in bytes
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, dict):
        return sum(get_tensor_size(v) for v in tensor.values())
    else:
        return 0


def gather_log(log, world_size):
    if world_size == 1:
        return log

    # Free up GPU cache
    torch.cuda.empty_cache()

    # Check the size of log and move to CPU if larger than 1GB
    size_in_bytes = get_tensor_size(log)
    size_in_gb = size_in_bytes / (1024**3)  # Convert bytes to gigabytes

    if size_in_gb > 1:
        if isinstance(log, torch.Tensor):
            log = log.cpu() if log.is_cuda else log
        elif isinstance(log, dict):
            for key, value in log.items():
                if isinstance(value, torch.Tensor):
                    log[key] = value.cpu()

    log_list = []
    log_keys = list(log.keys())  # list of keys
    chunk_size = max(1, len(log) // world_size)

    for i in range(0, len(log), chunk_size):
        chunk_keys = log_keys[i : i + chunk_size]  # slicing the list of keys
        chunk_dict = {k: log[k] for k in chunk_keys}  # create a chunk dictionary
        chunk_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(chunk_list, chunk_dict)
        log_list.extend(chunk_list)

    # combine the gathered chunks back to a single dictionary
    gathered_log = dict()
    for chunk in log_list:
        for key, value in chunk.items():
            if key not in gathered_log:
                gathered_log[key] = []
            gathered_log[key].extend(value if isinstance(value, list) else [value])

    # Initialize all keys in the log dictionary
    for key in log_keys:
        if key not in gathered_log:
            gathered_log[key] = []

    # combine all gathered values for each key
    combined_log = {
        key: sum((chunk[key] for chunk in log_list if key in chunk), [])
        for key in log_keys
    }
    return combined_log


def get_log_mean(log):
    out = {}

    for key in log:
        if "predictions" in key or "labels" in key:
            continue

        try:
            if "accuracy" in key:
                values = np.array(log[key]).squeeze()
                if len(values):
                    out[key] = 100 * (values.sum()) / len(values)
            else:
                num_nans = log[key].isnan().sum().int()
                if num_nans > 0:
                    print(
                        f"DEBUG: `get_log_mean()`: There are {num_nans} in log[{key}]!"
                    )
                # out[key] = np.mean(log[key])  # original
                out[key] = np.nanmean(log[key])
        except Exception as e:
            logging.error(f"{key}: {e}")

    for key in ["train", "val"]:
        labels = log[f"{key}_labels"]
        predictions = log[f"{key}_predictions"]
        predictions_arr = (np.array(predictions) > 0).astype(np.float64)
        labels_arr = np.array(labels).astype(np.float64)

        try:
            out[f"{key}_roc_auc"] = roc_auc_score(
                y_true=np.asarray(labels), y_score=np.array(predictions)
            )
        except Exception:
            out["{key}_roc_auc"] = 0.0

        try:
            out[f"{key}_precision"] = precision_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_precision"] = 0.0

        try:
            out[f"{key}_recall"] = recall_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_recall"] = 0.0

    if "train_atom_labels" in log:
        labels = log["train_atom_labels"]
        predictions = log["train_atom_predictions"]
        predictions_arr = (np.array(predictions) > 0).astype(np.float64)
        labels_arr = np.array(labels).astype(np.float64)

        key = "train_atom"
        try:
            out[f"{key}_roc_auc"] = roc_auc_score(
                y_true=np.asarray(labels), y_score=np.array(predictions)
            )
        except Exception:
            out[f"{key}_roc_auc"] = 0.0

        try:
            out[f"{key}_precision"] = precision_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_precision"] = 0.0

        try:
            out[f"{key}_recall"] = recall_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_recall"] = 0.0

    if "val_atom_labels" in log:
        labels = log["val_atom_labels"]
        predictions = log["val_atom_predictions"]
        predictions_arr = (np.array(predictions) > 0).astype(np.float64)
        labels_arr = np.array(labels).astype(np.float64)

        key = "val_atom"
        try:
            out[f"{key}_roc_auc"] = roc_auc_score(
                y_true=np.asarray(labels), y_score=np.array(predictions)
            )
        except Exception:
            out["{key}_roc_auc"] = 0.0

        try:
            out[f"{key}_precision"] = precision_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_precision"] = 0.0

        try:
            out[f"{key}_recall"] = recall_score(
                y_true=labels_arr, y_pred=predictions_arr
            )
        except Exception:
            out[f"{key}_recall"] = 0.0

    return out


class FilteringModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        t_to_sigma = partial(t_to_sigma_compl, args=args)

        self.model = get_model(
            args=args,
            t_to_sigma=t_to_sigma,
            confidence_mode=True,
            device=self.device,  # PyL doesn't need a device argument
        )

        self.loss = FilteringLoss(args=args)
        self._log = defaultdict(list)

    def filtering_log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy().tolist()

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        log = self._log
        log[self.stage + "_" + key].extend(data)

    def training_step(self, batch, batch_idx, stage: str = "train"):
        self.stage = stage

        outputs = self.general_step_with_oom(batch, batch_idx)
        predictions = outputs["filtering_pred"]
        loss, loss_breakdown = self.loss(outputs, batch, apply_mean=True)

        for key, value in loss_breakdown.items():
            self.filtering_log(key, value)

        if self.args.rmsd_prediction:
            pass
        else:
            if isinstance(self.args.rmsd_classification_cutoff, list):
                labels = None
            else:
                labels = batch.y
                accuracy = (labels == (predictions > 0).float()).detach().cpu().numpy()

                self.filtering_log("accuracy", accuracy)
                self.filtering_log("predictions", predictions.detach().cpu().numpy())
                self.filtering_log("labels", labels.detach().cpu().numpy())

                if self.args.atom_lig_confidence:
                    atom_predictions = outputs["filtering_atom_pred"]
                    atom_labels = batch.y_aa
                    atom_accuracy = (
                        (atom_labels == (atom_predictions > 0).float())
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    self.filtering_log("atom_accuracy", atom_accuracy)
                    self.filtering_log(
                        "atom_predictions", atom_predictions.detach().cpu().numpy()
                    )
                    self.filtering_log(
                        "atom_labels", atom_labels.detach().cpu().numpy()
                    )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.stage = "val"

        # Validation dataloader
        if dataloader_idx == 0:
            with torch.no_grad():
                outputs = self.general_step_with_oom(batch, batch_idx)

            _, loss_breakdown = self.loss(outputs, batch, apply_mean=True)
            for key, value in loss_breakdown.items():
                self.filtering_log(key, value)

        predictions = outputs["filtering_pred"]

        if self.args.rmsd_prediction:
            pass
        else:
            if isinstance(self.args.rmsd_classification_cutoff, list):
                labels = None
            else:
                labels = batch.y
                accuracy = (labels == (predictions > 0).float()).detach().cpu().numpy()

                self.filtering_log("accuracy", accuracy)
                self.filtering_log("predictions", predictions.detach().cpu().numpy())
                self.filtering_log("labels", labels.detach().cpu().numpy())

                if self.args.atom_lig_confidence:
                    atom_predictions = outputs["filtering_atom_pred"]
                    atom_labels = batch.y_aa
                    atom_accuracy = (
                        (atom_labels == (atom_predictions > 0).float())
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    self.filtering_log("atom_accuracy", atom_accuracy)
                    self.filtering_log(
                        "atom_predictions", atom_predictions.detach().cpu().numpy()
                    )
                    self.filtering_log(
                        "atom_labels", atom_labels.detach().cpu().numpy()
                    )

    def general_step_with_oom(self, batch, batch_idx):
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

    def on_train_epoch_end(self):
        log = {
            key: self._log[key] for key in self._log if "train_" in key or "val_" in key
        }

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
            optimizer = self.optimizers()
            mean_log["current_lr"] = optimizer.param_groups[0]["lr"]

            if self.args.wandb:
                wandb.log(mean_log)

            print("Epoch completed. Printing metrics...")
            print_str = f"Epoch {self.trainer.current_epoch}:"
            print_str += f" Training loss: {mean_log['train_loss']:.4f}"

            for key in ["filtering"]:
                key_to_check = f"train_{key}_loss"
                if key_to_check in mean_log:
                    print_str += f" {key} {mean_log[key_to_check]:.4f}"

            print(print_str, flush=True)

            print_str = f"Epoch {self.trainer.current_epoch}:"
            print_str += f" Validation loss: {mean_log['val_loss']:.4f}"

            for key in [
                "filtering_loss",
                "accuracy",
                "roc_auc",
                "atom_filtering_loss",
                "atom_accuracy",
                "atom_roc_auc",
            ]:
                key_to_check = f"val_{key}"
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
                mode=self.args.main_metric_goal
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
                "monitor": self.args.main_metric
                if self.args.val_inference_freq is not None
                else "val_loss",
                "interval": "epoch",
                "strict": False,
                "frequency": self.args.val_inference_freq or 1,
            }

        return optim_dict


def load_pretrained_model(model_args, ckpt_file, freeze: bool = False):
    model = FilteringModule(args=model_args)
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
