from typing import Optional
from dataclasses import dataclass, fields, asdict
import argparse

from omegaconf import OmegaConf, DictConfig


@dataclass(frozen=True)
class DockingModelConfig:
    type: str = "AAScoreModel"
    esm_embeddings_path: str = None

    # model args
    all_atoms: bool = True
    in_lig_edge_features: int = 4
    embedding_type: str = "sinusoidal"
    sigma_embed_dim: int = 64
    embedding_scale: int = 1000
    no_torsion: bool = False
    max_radius: int = 5
    scale_by_sigma: bool = True
    sh_lmax: int = 1
    distance_embed_dim: int = 64
    cross_distance_embed_dim: int = 64
    dropout: float = 0.1
    use_second_order_repr: bool = False
    cross_max_distance: float = 80.0
    dynamic_max_cross: bool = False
    separate_noise_schedule: bool = False
    smooth_edges: bool = False
    odd_parity: bool = False
    confidence_mode: bool = False
    num_confidence_outputs: int = 1
    fixed_center_conv: bool = True
    no_aminoacid_identities: bool = False
    not_fixed_center_conv: bool = True
    flexible_sidechains: bool = True
    flexible_backbone: bool = True
    receptor_radius: int = 15
    c_alpha_max_neighbors: int = 24
    atom_radius: float = 5.0
    atom_max_neighbors: int = 12
    sidechain_tor_bridge: bool = True
    use_bb_orientation_feats: bool = False
    only_nearby_residues_atomic: bool = True
    activation_func: str = "ReLU"
    clamped_norm_min: float = 0.0
    no_batch_norm: bool = False

    # norm layer in TensorProductConvLayer
    norm_type: str = "layer_norm"
    norm_affine: bool = True

    # determines model size
    nv: int = 2
    ns: int = 4
    num_conv_layers: int = 6

    @classmethod
    def from_dict(cls, data: dict) -> "DockingModelConfig":
        # Create an instance of the data class from a dictionary
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        data = vars(args)
        return cls.from_dict(data=data)


@dataclass
class DockingDataConfig:
    dataset: str

    # all paths
    cache_path: str
    split_train: str
    split_val: str
    cluster_file: Optional[str] = None

    require_ligand: bool = True

    # some cache cfg
    limit_complexes: int = 0
    complexes_per_cluster: int = 10
    multiplicity: int = 1

    # validation inference
    run_val_inference: bool = False
    num_inference_complexes: int = 500

    # data loading
    batch_size: int = 4
    num_workers: int = 1
    num_dataloader_workers: int = 26
    pin_memory: bool = False
    dataloader_drop_last: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        # Create an instance of the data class from a dictionary
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        data = vars(args)
        return cls.from_dict(data)


@dataclass(frozen=True)
class SigmaConfig:
    # Ligand diffusion
    tr_sigma_min: float
    tr_sigma_max: float
    rot_sigma_min: float
    rot_sigma_max: float
    tor_sigma_min: float
    tor_sigma_max: float

    bb_tr_sigma: Optional[float] = None
    bb_rot_sigma: Optional[float] = None
    sidechain_tor_sigma: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict):
        # Create an instance of the data class from a dictionary
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        data = vars(args)
        return cls.from_dict(data=data)


@dataclass(frozen=True)
class TimeConfig:
    # Parameters used for sampling time ~ [0, 1] and for initializing schedules
    sampling_alpha: float = 1
    sampling_beta: float = 1
    bb_tr_bridge_alpha: Optional[float] = None
    bb_rot_bridge_alpha: Optional[float] = None
    sc_tor_bridge_alpha: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict):
        # Create an instance of the data class from a dictionary
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        data = vars(args)
        return cls.from_dict(data=data)


# Hacky first version
def config_from_args(args: argparse.Namespace) -> DictConfig:
    time_config = {
        "sampling_alpha": args.sampling_alpha,
        "sampling_beta": args.sampling_beta,
        "bb_tr_bridge_alpha": args.bb_tr_bridge_alpha,
        "bb_rot_bridge_alpha": args.bb_rot_bridge_alpha,
        "sc_tor_bridge_alpha": args.sc_tor_bridge_alpha,
    }

    # Hardcoded for now
    if "dataset" not in args:
        args.dataset = "pdbbind"
        args.in_lig_edge_features = 4

    sigma_config = {
        "tr_sigma_min": args.tr_sigma_min,
        "tr_sigma_max": args.tr_sigma_max,
        "rot_sigma_min": args.rot_sigma_min,
        "rot_sigma_max": args.rot_sigma_max,
        "tor_sigma_min": args.tor_sigma_min,
        "tor_sigma_max": args.tor_sigma_max,
        "bb_tr_sigma": args.bb_tr_sigma,
        "bb_rot_sigma": args.bb_rot_sigma,
        "sidechain_tor_sigma": args.sidechain_tor_sigma,
    }

    transforms_config = {
        "time_args": time_config,
        "sigma_args": sigma_config,
        "flexible_backbone": args.flexible_backbone,
        "flexible_sidechains": args.flexible_sidechains,
        "fast_updates": True,
        "ligand": {"no_torsion": args.no_torsion},
        "protein": {
            "sidechain_tor_bridge": args.sidechain_tor_bridge,
            "use_bb_orientation_feats": args.use_bb_orientation_feats,
        },
        "pocket": {
            "pocket_reduction": True,
            "pocket_buffer": args.pocket_buffer,
            "pocket_radius": 5,
            "all_atoms": args.all_atoms,
        },
        "unbalanced": {"match_max_rmsd": args.match_max_rmsd},
        "nearby_atoms": {
            "only_nearby_residues_atomic": args.only_nearby_residues_atomic,
            "nearby_residues_atomic_radius": args.nearby_residues_atomic_radius,
            "nearby_residues_atomic_min": args.nearby_residues_atomic_min,
        },
        "bb_prior": {
            "bb_random_prior": args.bb_random_prior,
            "bb_random_prior_noise": args.bb_random_prior_noise,
            "bb_random_prior_ot": args.bb_random_prior_ot,
            "bb_random_prior_ot_inf": args.bb_random_prior_ot_inf,
            "bb_random_prior_std": args.bb_random_prior_std,
        },
    }

    sampler_config = {
        "inference_steps": args.inference_steps,
        "sampling_alpha": args.sampling_alpha,
        "sampling_beta": args.sampling_beta,
        "bb_tr_bridge_alpha": args.bb_tr_bridge_alpha,
        "bb_rot_bridge_alpha": args.bb_rot_bridge_alpha,
        "sc_tor_bridge_alpha": args.sc_tor_bridge_alpha,
        "no_torsion": args.no_torsion,
        "sidechain_tor_bridge": args.sidechain_tor_bridge,
        "flexible_backbone": args.flexible_backbone,
        "flexible_sidechains": args.flexible_sidechains,
        "all_atoms": args.all_atoms,
        "sigma": sigma_config,
    }

    strategy_config = {
        "type": getattr(args, "strategy", "auto"),
        "sharding_strategy": getattr(args, "sharding_strategy", None),
        "precision": getattr(args, "precision", None),
    }

    trainer_config = {
        "default_root_dir": f"{args.log_dir}/{args.run_name}",
        "accelerator": "auto",
        "num_nodes": getattr(args, "num_nodes", 1),
        "devices": getattr(args, "devices", 1),
        "max_epochs": args.n_epochs,
        "accumulate_grad_batches": args.accumulate_grad,
        "num_sanity_val_steps": 0,
    }

    loss_config = {
        "rot_weight": args.rot_weight,
        "tor_weight": args.tor_weight,
        "tr_weight": args.tr_weight,
        "bb_rot_weight": args.bb_rot_weight,
        "bb_tr_weight": args.bb_tr_weight,
        "sc_tor_weight": args.sc_tor_weight,
        "flexible_backbone": args.flexible_backbone,
        "flexible_sidechains": args.flexible_sidechains,
        "no_torsion": args.no_torsion,
        "lig_transform_type": "diffusion",
        "use_new_pipeline": getattr(args, "use_new_pipeline", False),
    }

    training_config = {
        "adamw": False,  # What should this be?
        "lr": args.lr,
        "scheduler": args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "w_decay": args.w_decay,
        "skip_nan_grad_updates": getattr(args, "skip_nan_grad_updates", False),
        "check_nan_grads": getattr(args, "check_nan_grads", False),
        "except_on_nan_grads": getattr(args, "except_on_nan_grads", False),
        "ema_rate": args.ema_rate,
        "use_ema": args.use_ema,
        "val_inference_freq": args.val_inference_freq,
        "inference_earlystop_metric": args.inference_earlystop_metric,
        "inference_earlystop_goal": args.inference_earlystop_goal,
        # check_unused_params: false
    }

    logger_config = {
        "wandb": args.wandb,
        "entity": args.entity,
        "project": args.project,
        "name": args.run_name,
    }

    callbacks_config = {
        "flexible_sidechains": args.flexible_sidechains,
        "flexible_backbone": args.flexible_backbone,
        "val_inference_freq": args.val_inference_freq,
        "inference_earlystop_goal": args.inference_earlystop_goal,
        "inference_earlystop_metric": args.inference_earlystop_metric,
        "wandb": args.wandb,
    }

    data_config = asdict(DockingDataConfig.from_args(args))
    model_config = asdict(DockingModelConfig.from_args(args))

    config_dict = {
        "task": args.task,
        "seed": getattr(args, "seed", 42),
        "log_dir": args.log_dir,
        "dataset": args.dataset,
        "run_name": args.run_name,
        "no_torsion": args.no_torsion,
        "flexible_sidechains": args.flexible_sidechains,
        "flexible_backbone": args.flexible_backbone,
        "sidechain_tor_bridge": args.sidechain_tor_bridge,
        "all_atoms": args.all_atoms,
        "strategy": strategy_config,
        "trainer": trainer_config,
        "time": time_config,
        "sigma": sigma_config,
        "transforms": transforms_config,
        "sampler": sampler_config,
        "loss": loss_config,
        "training": training_config,
        "data": data_config,
        "model": model_config,
        "logger": logger_config,
        "callbacks": callbacks_config,
    }

    config = OmegaConf.create(config_dict)
    return config
