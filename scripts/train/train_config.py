import sys
import os
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.strategies import (
    DDPStrategy,
    FSDPStrategy,
)
from omegaconf import OmegaConf
import torch
import logging

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.plugins.precision import FSDPPrecision

from flexdock.data.parse.base import save_config
from flexdock.data.modules.training import setup_training_datamodule
from flexdock.models.pl_modules import setup_model
from flexdock.models.layers.tensor_product import TensorProductConvLayer

from flexdock.utils.callbacks import setup_callbacks


def setup_strategy(cfg):
    strategy_str = cfg.type.lower()  # make it lowercase

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
            cfg.sharding_strategy
        )

    if "awp" in strategy_str:
        rank_zero_info("INFO: FSDP - Auto-Wrap-Policy")
        auto_wrap_policy = {TensorProductConvLayer}
        strategy_kwargs["auto_wrap_policy"] = auto_wrap_policy

    if "ac" in strategy_str:
        rank_zero_info("INFO: FSDP - Activation Checkpointing")
        ac_policy = {TensorProductConvLayer}
        strategy_kwargs["activation_checkpointing_policy"] = ac_policy

    precision = cfg.get("precision", None)
    if precision is not None:
        rank_zero_info(f"Precision={precision}")
        precision_plugin = FSDPPrecision(precision=precision)
        strategy_kwargs["precision_plugin"] = precision_plugin

    pl_strategy = FSDPStrategy(**strategy_kwargs)

    return pl_strategy


def setup_logger(cfg) -> Optional[Logger]:
    logger = None
    logger_cfg = cfg.logger
    if logger_cfg.wandb:
        logger_cfg = logger_cfg
        logger = WandbLogger(
            entity=logger_cfg.entity,
            project=logger_cfg.project,
            name=logger_cfg.name,
            tags=logger_cfg.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    return logger


def main(config_file, args):
    assert os.path.exists(config_file)
    raw_config = OmegaConf.load(config_file)

    # Apply input arguments
    args = OmegaConf.from_dotlist(args)
    cfg = OmegaConf.merge(raw_config, args)
    OmegaConf.resolve(cfg)

    logging.getLogger().setLevel("INFO")

    rank_zero_info(f"Running with seed {cfg.seed}")
    seed_everything(cfg.seed)

    data_module = setup_training_datamodule(
        data_cfg=cfg.data, transform_cfg=cfg.transforms
    )

    model = setup_model(cfg, task=cfg.data.task)

    run_dir = os.path.join(cfg.log_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    strategy = setup_strategy(cfg.strategy)
    callbacks = setup_callbacks(args=cfg.callbacks, run_dir=run_dir)
    logger = setup_logger(cfg=cfg)

    trainer_cfg = cfg.trainer
    trainer = pl.Trainer(
        **trainer_cfg, strategy=strategy, callbacks=callbacks, logger=logger
    )

    config_out = os.path.join(run_dir, "model_parameters.yml")
    save_config(cfg, config_out)

    # fit model
    trainer.fit(
        model=model, datamodule=data_module, ckpt_path=cfg.get("restart_ckpt", None)
    )


if __name__ == "__main__":
    config_file = sys.argv[1]
    args = sys.argv[2:]

    main(config_file, args)
