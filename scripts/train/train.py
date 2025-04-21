import os
import yaml
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.strategies import (
    DDPStrategy,
    FSDPStrategy,
)
import torch
import logging

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.pytorch.utilities import rank_zero_info

from flexdock.data.parse.base import save_yaml_file
from flexdock.data.modules.training import setup_training_datamodule
from flexdock.models.pl_modules import setup_model
from flexdock.models.layers.tensor_product import TensorProductConvLayer

from flexdock.utils.parsing import parse_train_args
from flexdock.utils.callbacks import setup_callbacks


def setup_strategy(args):
    strategy_str = args.strategy.lower()  # make it lowercase

    if strategy_str == "auto":
        rank_zero_info(
            "INFO: Strategy automatically selected by lightning: pl_strategy='auto'"
        )
        pl_strategy = "auto"
        return pl_strategy

    if not torch.cuda.is_available():
        return DDPStrategy(find_unused_parameters=True)

    SHARDING_STRATEGY = {
        "full": "FULL_SHARD",
        "hybrid": "HYBRID_SHARD",
        "none": "NO_SHARD",
        "grad": "SHARD_GRAD_OP",
    }

    strategy_kwargs = {}
    if strategy_str == "ddp":
        rank_zero_info("DDP: pl_strategy=DDPStrategy(find_unused_parameters=True)")
        strategy_kwargs["sharding_strategy"] = "NO_SHARD"
    else:
        rank_zero_info(
            "INFO: Option 0: pl_strategy = FSDPStrategy(sharding_strategy=...)"
        )
        strategy_kwargs["sharding_strategy"] = SHARDING_STRATEGY.get(
            args.sharding_strategy
        )

    if "awp" in strategy_str:
        rank_zero_info("INFO: FSDP - Auto-Wrap-Policy")
        auto_wrap_policy = {TensorProductConvLayer}
        strategy_kwargs["auto_wrap_policy"] = auto_wrap_policy

    if "ac" in strategy_str:
        rank_zero_info("INFO: FSDP - Activation Checkpointing")
        ac_policy = {TensorProductConvLayer}
        strategy_kwargs["activation_checkpointing_policy"] = ac_policy

    pl_strategy = FSDPStrategy(**strategy_kwargs)

    return pl_strategy


def setup_logger(args) -> Optional[Logger]:
    logger = None
    if args.wandb:
        run_id = None
        if args.restart_dir is not None:
            model_parameters_file = os.path.join(
                args.restart_dir, "model_parameters.yml"
            )
            if os.path.exists(model_parameters_file):
                with open(model_parameters_file, "r") as file:
                    config = yaml.safe_load(file)
                run_id = config["wandb_run_id"]
        logger = WandbLogger(
            entity=args.entity,
            project=args.project,
            name=f"{args.run_name}",
            id=run_id,
            resume="allow",
            tags=[
                str(args.lr),
                str(args.batch_size),
                str(args.nv),
                str(args.ns),
                str(args.num_conv_layers),
                args.strategy,
                str(args.num_nodes),
                str(args.num_gpus_per_node),
            ],
            config=args,
        )
    return logger


def main():
    args = parse_train_args()
    logging.getLogger().setLevel("INFO")

    rank_zero_info(f"Running with seed {args.seed}")
    seed_everything(args.seed)

    # TODO: Move to LightningDataModule
    data_module = setup_training_datamodule(args=args)

    model = setup_model(args, task=args.task)

    run_dir = os.path.join(args.log_dir, args.run_name)
    strategy = setup_strategy(args)
    callbacks = setup_callbacks(args=args, run_dir=run_dir)
    logger = setup_logger(args=args)

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=args.precision,
        strategy=strategy,
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_epochs=args.n_epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        gradient_clip_val=args.grad_clip,
        # gradient_clip_algorithm='value',  # required for MP
        accumulate_grad_batches=args.accumulate_grad,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        detect_anomaly=args.use_anomaly_detection,
    )

    yaml_file_name = os.path.join(run_dir, "model_parameters.yml")
    save_yaml_file(yaml_file_name, args.__dict__)

    ckpt_file = None
    if args.restart_dir is not None:
        ckpt_file = f"{args.restart_dir}/last_model.pt"
        rank_zero_info(f"DEBUG: Load from checkpoint: ckpt_file: {ckpt_file}")
        if not os.path.exists(ckpt_file):
            ckpt_file = None
            rank_zero_info(
                f"DEBUG: No file found at {ckpt_file}, initialising a new run."
            )

    # fit model
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=ckpt_file,
    )


if __name__ == "__main__":
    main()
