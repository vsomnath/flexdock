import argparse
import yaml

from lightning.pytorch import Trainer, seed_everything
import torch
from omegaconf import OmegaConf

from flexdock.data.modules.inference import InferenceDataModule
from flexdock.data.feature.featurizer import FeaturizerConfig
from flexdock.data.write.writer import FlexDockWriter
from flexdock.models.pl_modules.inference import InferenceModule
from flexdock.utils.configs import config_from_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        default=None,
        type=str,
        help="Csv file containing protein paths and ligand SMILES",
    )
    parser.add_argument(
        "--esm_embeddings_path", default=None, help="Path to ESM Embeddings"
    )
    parser.add_argument(
        "--limit_complexes", type=int, default=None, help="Limit complexes"
    )

    parser.add_argument("--output_dir", default="inf_results/dummy_test")

    # Model Related Args
    parser.add_argument(
        "--docking_model_dir",
        type=str,
        default="workdir",
        help="Path to folder with trained score model and hyperparameters",
    )
    parser.add_argument(
        "--docking_ckpt",
        type=str,
        default="best_model.pt",
        help="Checkpoint to use inside the folder",
    )
    parser.add_argument(
        "--filtering_model_dir",
        type=str,
        default=None,
        help="Path to folder with trained confidence model and hyperparameters",
    )
    parser.add_argument(
        "--filtering_ckpt",
        type=str,
        default="best_model.pt",
        help="Checkpoint to use inside the folder",
    )
    parser.add_argument(
        "--use_ema_weights",
        action="store_true",
        default=True,
        help="Whether to use ema weights. This only works for Pytorch Lightning version",
    )

    parser.add_argument("--model_in_old_version", action="store_true")

    parser.add_argument("--pocket_reduction", action="store_true")
    # parser.add_argument("--pocket_cutoff", default=5.0, type=float)
    parser.add_argument("--pocket_buffer", default=20.0, type=float)
    parser.add_argument("--pocket_min_size", type=int, default=1)
    parser.add_argument("--only_nearby_residues_atomic", action="store_true")

    # Inference related args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of poses to sample in parallel",
    )
    parser.add_argument("--actual_steps", type=int, default=None)
    parser.add_argument(
        "--num_inference_complexes",
        type=int,
        default=None,
        help="Number of complexes to do inference on. By default, set to None for all complexes",
    )
    parser.add_argument(
        "--sigma_schedule", type=str, default="expbeta", help="Schedule for t"
    )
    parser.add_argument(
        "--samples_per_complex",
        type=int,
        default=1,
        help="Number of samplges per complex",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--inf_sched_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for t sched",
    )
    parser.add_argument(
        "--inf_sched_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for t sched",
    )
    parser.add_argument("--sidechain_tor_bridge", action="store_true", default=True)
    parser.add_argument("--ode", action="store_true", default=False)
    parser.add_argument("--no_random", action="store_true", default=False)
    parser.add_argument(
        "--no_final_step_noise",
        action="store_true",
        default=True,
        help="Whether to add noise after the final step",
    )

    # Low temperature sampling
    parser.add_argument("--diff_temp_sampling_tr", type=float, default=1.0)
    parser.add_argument("--diff_temp_psi_tr", type=float, default=0.0)
    parser.add_argument("--diff_temp_sigma_data_tr", type=float, default=0.5)
    parser.add_argument("--diff_temp_sampling_rot", type=float, default=1.0)
    parser.add_argument("--diff_temp_psi_rot", type=float, default=0.0)
    parser.add_argument("--diff_temp_sigma_data_rot", type=float, default=0.5)
    parser.add_argument("--diff_temp_sampling_tor", type=float, default=1.0)
    parser.add_argument("--diff_temp_psi_tor", type=float, default=0.0)
    parser.add_argument("--diff_temp_sigma_data_tor", type=float, default=0.5)

    parser.add_argument("--flow_temp_scale_0_tr", type=float, default=1.0)
    parser.add_argument("--flow_temp_scale_1_tr", type=float, default=1.0)
    parser.add_argument("--flow_temp_scale_0_rot", type=float, default=1.0)
    parser.add_argument("--flow_temp_scale_1_rot", type=float, default=1.0)
    parser.add_argument("--flow_temp_scale_0_tor", type=float, default=1.0)
    parser.add_argument("--flow_temp_scale_1_tor", type=float, default=1.0)

    parser.add_argument("--initial_noise_std_proportion", type=float, default=1.0)
    parser.add_argument("--use_fast_sampling", action="store_true")

    parser.add_argument("--flexible_backbone", action="store_true", default=True)
    parser.add_argument("--flexible_sidechains", action="store_true", default=True)

    parser.add_argument("--debug_backbone", action="store_true")
    parser.add_argument("--debug_sidechains", action="store_true")

    # Relaxation Model Args
    parser.add_argument("--only_run_relaxation", action="store_true", default=False)
    parser.add_argument("--run_relaxation", action="store_true", default=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--relax_model_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--relax_run_name", type=str)
    parser.add_argument("--relax_ckpt", type=str, default="best_inference_epoch_model")
    parser.add_argument("--use_posebusters_em", action="store_true", default=False)
    parser.add_argument("--add_solvent", action="store_true", default=False)
    parser.add_argument("--platform", default="fastest")
    parser.add_argument("--no_model", action="store_true", default=False)
    parser.add_argument("--no_energy_filtering", action="store_true", default=False)
    parser.add_argument("--no_rmsd_filtering", action="store_true", default=False)
    parser.add_argument("--relax_batch_size", type=int, default=20)
    parser.add_argument("--relax_inference_steps", type=int, default=2)
    parser.add_argument("--relax_num_workers", type=int, default=1)
    parser.add_argument("--relax_schedule_type", type=str, default="uniform")
    parser.add_argument("--relax_schedule_param", type=float, default=None)
    parser.add_argument("--relax_n_conformers", type=int, default=1)
    parser.add_argument("--relax_tr_sigma", type=float, default=None)
    parser.add_argument("--relax_rot_sigma", type=float, default=None)
    parser.add_argument("--relax_torsion_sigma", type=float, default=None)
    parser.add_argument("--relax_bond_sigma", type=float, default=None)
    parser.add_argument("--relax_angle_sigma", type=float, default=None)
    parser.add_argument("--relax_fragment_sigma", type=float, default=None)
    parser.add_argument("--relax_ligand_sigma", type=float, default=None)
    parser.add_argument("--relax_atom_sigma", type=float, default=None)

    return parser.parse_args()


def load_args_from_yaml(yaml_file):
    with open(yaml_file) as f:
        configs = yaml.full_load(f)
    return argparse.Namespace(**configs)


def prepare_ckpt_and_args(args):
    checkpoints = {}
    configs = {}

    if args.only_run_relaxation:
        checkpoints["relaxation"] = f"{args.relax_model_dir}/{args.relax_ckpt}"
        relaxation_args = load_args_from_yaml(
            f"{args.relax_model_dir}/model_parameters.yml"
        )
        if args.model_in_old_version:
            relaxation_args.fast_updates = getattr(args, "use_fast_sampling", False)
            relaxation_args.norm_type = "batch_norm"
            relaxation_args.norm_affine = True
        configs["relaxation"] = relaxation_args

    else:
        checkpoints["docking"] = f"{args.docking_model_dir}/{args.docking_ckpt}"
        if args.model_in_old_version:
            docking_args = load_args_from_yaml(
                f"{args.docking_model_dir}/model_parameters.yml"
            )
            # To ensure backward compatibility to changes made since training old model
            docking_args.fast_updates = getattr(args, "use_fast_sampling", False)
            docking_args.norm_type = "batch_norm"
            docking_args.norm_affine = True
            docking_args.batchnorm_affine = True
            if "task" not in docking_args:
                docking_args.task = "docking"

            docking_config = config_from_args(args=docking_args)
            docking_config.model_in_old_version = True
        else:
            docking_config = OmegaConf.load(
                f"{args.docking_model_dir}/model_parameters.yml"
            )
            docking_config.model_in_old_version = False

        configs["docking"] = docking_config

        if args.filtering_model_dir is not None:
            checkpoints[
                "filtering"
            ] = f"{args.filtering_model_dir}/{args.filtering_ckpt}"
            filtering_params = load_args_from_yaml(
                f"{args.filtering_model_dir}/model_parameters.yml"
            )
            if "task" not in filtering_params:
                filtering_params.task = "filtering"

            if "dataset" not in filtering_params:
                filtering_params.dataset = "pdbbind"
            filtering_params.in_lig_edge_features = 4

            if args.model_in_old_version:
                filtering_params.norm_type = "batch_norm"
                filtering_params.norm_affine = True
                filtering_params.batchnorm_affine = True
                filtering_params.model_in_old_version = True
                filtering_params.activation_func = "ReLU"
                filtering_params.clamped_norm_min = 0.0

            filtering_params.fast_updates = getattr(args, "use_fast_sampling", False)
            configs["filtering"] = filtering_params
        else:
            checkpoints["filtering"] = None
            configs["filtering"] = None

        if args.run_relaxation:
            checkpoints["relaxation"] = f"{args.relax_model_dir}/{args.relax_ckpt}"
            relaxation_args = load_args_from_yaml(
                f"{args.relax_model_dir}/model_parameters.yml"
            )
            if args.model_in_old_version:
                relaxation_args.fast_updates = getattr(args, "use_fast_sampling", False)
                relaxation_args.norm_type = "batch_norm"
                relaxation_args.norm_affine = True
            configs["relaxation"] = relaxation_args

    return checkpoints, configs


def predict():
    args = parse_args()
    args.use_new_pipeline = args.use_fast_sampling

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare FeaturizerConfig
    featurizer_cfg = FeaturizerConfig(
        matching=False,
        popsize=None,
        maxiter=None,
        keep_original=False,
        remove_hs=True,
        num_conformers=1,
        max_lig_size=None,
        flexible_backbone=args.flexible_backbone,
        flexible_sidechains=args.flexible_sidechains,
    )

    # Setup InferenceDataModule
    datamodule = InferenceDataModule(
        input_csv=args.input_csv,
        featurizer_cfg=featurizer_cfg,
        limit_complexes=args.limit_complexes,
        esm_embeddings_path=args.esm_embeddings_path,
        pocket_reduction=args.pocket_reduction,
        pocket_buffer=args.pocket_buffer,
        pocket_min_size=args.pocket_min_size,
        only_nearby_residues_atomic=args.only_nearby_residues_atomic,
    )

    # Gather checkpoint files and configs
    checkpoints, configs = prepare_ckpt_and_args(args)

    if not args.only_run_relaxation:
        docking_cfg = configs["docking"]
        sampler_cfg = docking_cfg.get("sampler", None)
        if sampler_cfg is None:
            sampler_cfg = OmegaConf.create(
                {
                    "inference_steps": args.inference_steps,
                    "sampling_alpha": docking_cfg.time.sampling_alpha,
                    "sampling_beta": docking_cfg.time.sampling_beta,
                    "bb_tr_bridge_alpha": docking_cfg.time.bb_tr_bridge_alpha,
                    "bb_rot_bridge_alpha": docking_cfg.time.bb_rot_bridge_alpha,
                    "sc_tor_bridge_alpha": docking_cfg.time.sc_tor_bridge_alpha,
                    "no_torsion": docking_cfg.no_torsion,
                    "flexible_sidechains": docking_cfg.flexible_sidechains,
                    "flexible_backbone": docking_cfg.flexible_backbone,
                    "all_atoms": docking_cfg.all_atoms,
                    "sidechain_tor_bridge": docking_cfg.sidechain_tor_bridge,
                    "sigma": docking_cfg.sigma,
                }
            )
        sampler_cfg.inference_steps = args.inference_steps
    else:
        sampler_cfg = None

    # Setup inference module
    model_module = InferenceModule(
        args=args,
        sampler_cfg=sampler_cfg,
        configs=configs,
        checkpoints=checkpoints,
    )
    model_module.eval()

    callbacks = [
        FlexDockWriter(args=args, output_dir=args.output_dir, write_interval="batch")
    ]

    trainer = Trainer(
        default_root_dir=args.output_dir,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        enable_checkpointing=False,
        callbacks=callbacks,
        logger=None,
    )

    # Compute predictions
    trainer.predict(model_module, datamodule, return_predictions=False)


if __name__ == "__main__":
    predict()
