from argparse import ArgumentParser, FileType


def parse_docking_args(parser):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--config", type=FileType(mode="r"), default=None)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="workdir",
        help="Folder in which to save model and logs",
    )
    parser.add_argument(
        "--restart_dir",
        type=str,
        help="Folder of previous training model from which to restart",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/cacheNew2",
        help="Folder from where to load/restore cached dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PDBBIND_atomCorrected/",
        help="Folder containing original structures",
    )
    parser.add_argument(
        "--split_train",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_train",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_val",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_val",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        default="data/splits/timesplit_test",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--test_sigma_intervals",
        action="store_true",
        default=False,
        help="Whether to log loss per noise interval",
    )
    parser.add_argument(
        "--test_bridge_intervals",
        action="store_true",
        default=False,
        help="Whether to log loss per bridge interval",
    )
    parser.add_argument(
        "--val_inference_freq",
        type=int,
        default=None,
        help="Frequency of epochs for which to run expensive inference on val data",
    )
    parser.add_argument(
        "--train_inference_freq",
        type=int,
        default=None,
        help="Frequency of epochs for which to run expensive inference on train data",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps for inference on val",
    )
    parser.add_argument(
        "--num_inference_complexes",
        type=int,
        default=100,
        help="Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)",
    )
    parser.add_argument(
        "--inference_earlystop_metric",
        type=str,
        default="valinf_rmsds_lt2",
        help="LR decay is triggered on this metric. We save the best model according to this metric."
        " Can save multiple models for best metrics separated by commas, but LR decay will be performed on only the first.",
    )
    parser.add_argument(
        "--inference_earlystop_goal",
        type=str,
        default="max",
        help="Whether to maximize or minimize metric(s)",
    )
    parser.add_argument("--run_name", type=str, default="", help="")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=False,
        help="CUDA optimization parameter for faster training",
    )
    parser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="pin_memory arg of dataloader",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        action="store_true",
        default=False,
        help="drop_last arg of dataloader",
    )
    parser.add_argument(
        "--filter_unloadable",
        action="store_true",
        default=False,
        help="Perform an initial walkthrough of the dataset during initialisation "
        "and remove any complex that produces an error on loading.",
    )

    parser.add_argument("--task", default="docking", type=str)
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        default=False,
        help="Only construct dataset and dataloaders (which preprocesses and caches dataset),"
        " useful if you want to use only CPUs for preprocessing",
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        default=None,
        help="CSV containing complex cluster ids, if not specified then no sampling per cluster is done.",
    )
    parser.add_argument(
        "--complexes_per_cluster",
        type=int,
        default=1,
        help="Maximum number of samples per cluster",
    )

    # Wandb related args
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument(
        "--entity", type=str, default="flexdock", help="Name of the wandb entity"
    )  # TODO
    parser.add_argument(
        "--project", type=str, default="flexdock_scaling_ligbind_tr", help=""
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for model training.",
    )
    parser.add_argument(
        "--num_gpus_per_node",
        type=int,
        default=1,
        help="Number GPUs on each node for model training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # Training arguments
    # Training arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["auto", "ddp", "fsdp", "fsdp_awp", "fsdp_ac", "fsdp_ac_awp"],
    )
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        default="full",
        choices=["full", "none", "grad", "hybrid"],
    )
    parser.add_argument(
        "--precision", default="32", type=str, help="Precision used for training"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=400, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--scheduler", type=str, default=None, help="LR scheduler")
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=20,
        help="Patience of the LR scheduler",
    )
    parser.add_argument(
        "--adamw",
        action="store_true",
        default=False,
        help="Use AdamW optimizer instead of Adam",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--restart_lr",
        type=float,
        default=None,
        help="If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.",
    )
    parser.add_argument(
        "--w_decay", type=float, default=0.0, help="Weight decay added to loss"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for preprocessing"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether or not to use ema for the model weights",
    )
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.999,
        help="decay rate for the exponential moving average model parameters ",
    )

    parser.add_argument(
        "--activation_func",
        type=str,
        default="ReLU",
        help="Activation function of the flexdock module.",
    )
    parser.add_argument(
        "--use_anomaly_detection",
        action="store_true",
        default=False,
        help="Whether to use PyTorch anomaly detection",
    )

    # Dataset
    parser.add_argument(
        "--limit_complexes",
        type=int,
        default=10,
        help="If positive, the number of training and validation complexes is capped",
    )  # TODO change
    parser.add_argument(
        "--all_atoms",
        action="store_true",
        default=True,
        help="Whether to use the all atoms model",
    )
    parser.add_argument("--multiplicity", type=int, default=1, help="")
    parser.add_argument(
        "--chain_cutoff",
        type=float,
        default=10,
        help="Cutoff on whether to include non-interacting chains",
    )
    parser.add_argument(
        "--receptor_radius",
        type=float,
        default=30,
        help="Cutoff on distances for receptor edges",
    )
    parser.add_argument(
        "--c_alpha_max_neighbors",
        type=int,
        default=10,
        help="Maximum number of neighbors for each residue",
    )
    parser.add_argument(
        "--atom_radius",
        type=float,
        default=5,
        help="Cutoff on distances for atom connections",
    )
    parser.add_argument(
        "--atom_max_neighbors",
        type=int,
        default=8,
        help="Maximum number of atom neighbours for receptor",
    )
    parser.add_argument(
        "--matching_popsize",
        type=int,
        default=20,
        help="Differential evolution popsize parameter in matching",
    )
    parser.add_argument(
        "--matching_maxiter",
        type=int,
        default=20,
        help="Differential evolution maxiter parameter in matching",
    )
    parser.add_argument(
        "--max_lig_size",
        type=int,
        default=None,
        help="Maximum number of heavy atoms in ligand",
    )
    parser.add_argument(
        "--remove_hs", action="store_true", default=True, help="remove Hs"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=1,
        help="Number of conformers to match to each ligand",
    )
    parser.add_argument(
        "--esm_embeddings_path",
        type=str,
        default=None,
        help="If this is set then the LM embeddings at that path will be used for the receptor features",
    )
    parser.add_argument(
        "--pocket_reduction",
        action="store_true",
        default=False,
        help="Remove atoms from receptor that are not in the binding pocket",
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=float,
        default=5,
        help="Radius from ligand atoms used to define the pocket residues.",
    )
    parser.add_argument(
        "--pocket_min_size",
        type=int,
        default=1,
        help="Minimum number of atoms in the pocket residues.",
    )
    parser.add_argument(
        "--pocket_buffer",
        type=float,
        default=10,
        help="Buffer that will be added to the radius of the pocket",
    )
    parser.add_argument(
        "--not_fixed_knn_radius_graph",
        action="store_true",
        default=False,
        help="Use knn graph and radius graph with closest neighbors instead of random ones as with radius_graph",
    )
    parser.add_argument(
        "--not_knn_only_graph",
        action="store_true",
        default=False,
        help="Use knn graph only and not restrict to a specific radius",
    )
    parser.add_argument(
        "--include_miscellaneous_atoms",
        action="store_true",
        default=False,
        help="include non amino acid atoms for the receptor",
    )
    parser.add_argument(
        "--conformer_match_score",
        type=str,
        default="dist",
        help='The scoring function used for conformer matching. Can be either "dist", "nearest" or "exp". All take the distance to the holo structure, nearest and exp also optimize steric clashes. Nearest takes the closest steric clash, exp weights the steric clashes with something similar to an rbf kernel.',
    )
    parser.add_argument(
        "--compare_true_protein",
        action="store_true",
        default=False,
        help="whether to calculate the rmsd to the holo structure (i.e., match_protein_file). this is only possible with flexible sidechains and if the proein_file is an apo structure. This is only applied to the validation set",
    )
    parser.add_argument(
        "--match_max_rmsd",
        type=float,
        default=2.0,
        help="Specify the maximum RMSD when conformer matching sidechains. This RMSD will only be calculated in the pocket with pocket_buffer. This parameter only influences the training set, and has no impact on validation.",
    )
    parser.add_argument(
        "--use_original_conformer",
        action="store_true",
        default=False,
        help="use the original conformer structure for training if the matching rmsd is further away than match_max_rmsd value",
    )
    parser.add_argument(
        "--use_original_conformer_fallback",
        action="store_true",
        default=False,
        help="use the original conformer structure for training if the protein_file does not exist. This only effects training.",
    )
    parser.add_argument(
        "--cache_individual",
        action="store_true",
        default=False,
        help="Cache each preprocessed complex individually instead of jointly",
    )
    parser.add_argument(
        "--add_maxrmsd_to_cache_path",
        action="store_true",
        default=False,
        help="Whether to construct cache based on adding max_rmsd to cache_path",
    )
    parser.add_argument(
        "--transform", default="noise", choices=["noise", "bridge_noise", "flow"]
    )
    parser.add_argument("--reweighting_var", type=float, default=None, help="")

    # Pocket selection
    parser.add_argument("--use_fpocket_predictor", action="store_true", default=False)
    parser.add_argument("--fpocket_cache_path", type=str, default=None)
    parser.add_argument("--gt_pocket_rate", type=float, default=0.5)
    parser.add_argument(
        "--fpocket_topk",
        type=int,
        default=5,
        help="Sample pocket from top k ranked pockets from fpocket.",
    )

    # Diffusion
    parser.add_argument(
        "--tr_weight", type=float, default=0.17, help="Weight of translation loss"
    )
    parser.add_argument(
        "--rot_weight", type=float, default=0.17, help="Weight of rotation loss"
    )
    parser.add_argument(
        "--tor_weight", type=float, default=0.17, help="Weight of torsional loss"
    )
    parser.add_argument("--sc_tor_weight", type=float, default=0.17, help="")
    parser.add_argument("--bb_tr_weight", type=float, default=0.17, help="")
    parser.add_argument("--bb_rot_weight", type=float, default=0.17, help="")
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.33,
        help="Weight of confidence loss",
    )
    parser.add_argument(
        "--rot_sigma_min",
        type=float,
        default=0.1,
        help="Minimum sigma for rotational component",
    )
    parser.add_argument(
        "--rot_sigma_max",
        type=float,
        default=1.65,
        help="Maximum sigma for rotational component",
    )
    parser.add_argument(
        "--tr_sigma_min",
        type=float,
        default=0.1,
        help="Minimum sigma for translational component",
    )
    parser.add_argument(
        "--tr_sigma_max",
        type=float,
        default=30,
        help="Maximum sigma for translational component",
    )
    parser.add_argument(
        "--tor_sigma_min",
        type=float,
        default=0.0314,
        help="Minimum sigma for torsional component",
    )
    parser.add_argument(
        "--tor_sigma_max",
        type=float,
        default=3.14,
        help="Maximum sigma for torsional component",
    )
    parser.add_argument(
        "--sidechain_tor_sigma_min",
        type=float,
        default=0.0314,
        help="Minimum sigma for torsional components of sidechains",
    )
    parser.add_argument(
        "--sidechain_tor_sigma_max",
        type=float,
        default=3.14,
        help="Maximum sigma for torsional components of sidechains",
    )
    parser.add_argument(
        "--no_torsion",
        action="store_true",
        default=False,
        help="If set only rigid matching",
    )

    # Bridge related sigmas
    parser.add_argument(
        "--sidechain_tor_sigma",
        type=float,
        default=0.0,
        help="Bridge sigma for sidechains; set to low values",
    )
    parser.add_argument(
        "--bb_tr_sigma",
        type=float,
        default=0.0,
        help="Bridge sigma for tr component of backbone frames; set to low values",
    )
    parser.add_argument(
        "--bb_rot_sigma",
        type=float,
        default=0.0,
        help="Bridge sigma for rot component of backbone; set to low values",
    )

    # Model args (from DiffDock)
    parser.add_argument("--in_lig_edge_features", default=4, type=int)
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of interaction layers"
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=5.0,
        help="Radius cutoff for geometric graph",
    )
    parser.add_argument(
        "--scale_by_sigma",
        action="store_true",
        default=True,
        help="Whether to normalise the score",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=16,
        help="Number of hidden features per node of order 0",
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=4,
        help="Number of hidden features per node of order >0",
    )
    parser.add_argument(
        "--distance_embed_dim",
        type=int,
        default=32,
        help="Embedding size for the distance",
    )
    parser.add_argument(
        "--cross_distance_embed_dim",
        type=int,
        default=32,
        help="Embeddings size for the cross distance",
    )
    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        default=False,
        help="If set, it removes the batch norm",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=["batch_norm", "layer_norm"],
        default=None,
        help="Normalization type. Choose between 'batch_norm' and 'layer_norm'. Default is None.",
    )
    parser.add_argument(
        "--norm_affine",
        action="store_true",
        default=False,
        help="Whether to use affine=True for normalization layer",
    )
    parser.add_argument(
        "--use_second_order_repr",
        action="store_true",
        default=False,
        help="Whether to use only up to first order representations or also second",
    )
    parser.add_argument(
        "--cross_max_distance",
        type=float,
        default=80,
        help="Maximum cross distance in case not dynamic",
    )
    parser.add_argument(
        "--dynamic_max_cross",
        action="store_true",
        default=False,
        help="Whether to use the dynamic distance cutoff",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument(
        "--smooth_edges",
        action="store_true",
        default=False,
        help="Whether to apply additional smoothing weight to edges",
    )
    parser.add_argument(
        "--odd_parity",
        action="store_true",
        default=False,
        help="Whether to impose odd parity in output",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="sinusoidal",
        help="Type of diffusion time embedding",
    )
    parser.add_argument(
        "--sigma_embed_dim",
        type=int,
        default=32,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--embedding_scale",
        type=int,
        default=1000,
        help="Parameter of the diffusion time embedding",
    )
    parser.add_argument(
        "--sh_lmax",
        type=int,
        default=2,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--clamped_norm_min",
        type=float,
        default=1.0e-6,
        help="Minimum norm of the clamped values for tr_pred and rot_pred",
    )

    # New args for FlexDock-PyL
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument(
        "--lig_transform_type", default="diffusion", choices=["diffusion", "flow"]
    )
    parser.add_argument(
        "--prot_transform_type", default="geoflow", choices=["baseflow", "geoflow"]
    )
    parser.add_argument("--check_nan_grads", default=False, action="store_true")
    parser.add_argument(
        "--except_on_nan_grads",
        default=False,
        action="store_true",
        help="Raise exception on NaN gradients.",
    )
    parser.add_argument(
        "--skip_nan_grad_updates",
        default=False,
        action="store_true",
        help="Zero grads where a batch has NaN gradients.",
    )
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument(
        "--accumulate_grad",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over",
    )

    # Model new extras
    parser.add_argument(
        "--flexible_sidechains",
        action="store_true",
        default=False,
        help="Diffuse over side chain torsions for residues within flexdist of pocket",
    )
    parser.add_argument(
        "--flexdist",
        type=float,
        default=3.5,
        help="If a residue has at least one atom within flexdist of the pocket, it will be made flexible",
    )
    parser.add_argument(
        "--flexible_backbone",
        action="store_true",
        default=False,
        help="Learn bridge over pocket backbone",
    )
    parser.add_argument(
        "--flexdist_distance_metric",
        type=str,
        default="L2",
        help="Distance metric used to select residues within flexdist to pocket center",
    )
    parser.add_argument(
        "--separate_noise_schedule",
        action="store_true",
        default=False,
        help="Use different t for tr, rot, and tor",
    )
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sampling_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--rot_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--rot_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--tor_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--tor_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sidechain_tor_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sidechain_tor_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument("--sidechain_tor_bridge", action="store_true", help="")
    parser.add_argument("--bridge_norm_clip", type=float, default=None, help="")
    parser.add_argument("--bb_tr_bridge_alpha", type=float, default=0.01, help="")
    parser.add_argument("--bb_rot_bridge_alpha", type=float, default=0.01, help="")
    parser.add_argument("--sc_tor_bridge_alpha", type=float, default=0.01, help="")
    parser.add_argument(
        "--use_bb_orientation_feats", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--only_nearby_residues_atomic", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--nearby_residues_atomic_radius", type=float, default=3.5, help=""
    )
    parser.add_argument("--nearby_residues_atomic_min", type=int, default=1, help="")
    parser.add_argument(
        "--bb_random_prior", action="store_true", default=False, help=""
    )
    parser.add_argument("--bb_random_prior_ot", type=int, default=1, help="")
    parser.add_argument(
        "--bb_random_prior_noise", type=str, default="gaussian", help=""
    )
    parser.add_argument("--bb_random_prior_ot_inf", type=int, default=1, help="")
    parser.add_argument("--bb_random_prior_std", type=float, default=0.1, help="")

    # Debugging if drift works correctly
    parser.add_argument("--run_checks", action="store_true", default=False)
    parser.add_argument("--debug_backbone", action="store_true", default=False, help="")
    parser.add_argument(
        "--debug_sidechain", action="store_true", default=False, help=""
    )

    # Confidence Predictor in Model
    parser.add_argument(
        "--include_confidence_prediction",
        action="store_true",
        default=False,
        help="Whether to predict an additional confidence metric for each predicted structure",
    )
    parser.add_argument(
        "--high_confidence_threshold",
        type=float,
        default=5.0,
        help="If this is 0 then the confidence predictor tries to predict the centroid_distance. Otherwise it is the Ångström below which a prediction is labeled as good for supervising the confidence predictor",
    )
    parser.add_argument(
        "--tr_only_confidence",
        action="store_true",
        default=True,
        help="Whether to only supervise the confidence predictor with the translation",
    )
    parser.add_argument(
        "--confidence_no_batchnorm", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--confidence_dropout",
        type=float,
        default=0.0,
        help="MLP dropout in confidence readout",
    )

    parser.add_argument(
        "--not_fixed_center_conv", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--protein_file", type=str, default="protein_processed", help=""
    )
    parser.add_argument(
        "--apo_protein_file", type=str, default="protein_esmfold_aligned_tr", help=""
    )
    parser.add_argument(
        "--holo_protein_file", type=str, default="protein_processed", help=""
    )
    parser.add_argument(
        "--no_aminoacid_identities", action="store_true", default=False, help=""
    )

    # More fixes and upcoming changes
    parser.add_argument("--use_new_pipeline", action="store_true")
    parser.add_argument("--use_origpos_pocket", action="store_true")
    parser.add_argument("--add_nearby_residues_in_pocket", action="store_true")


def parse_filtering_args(parser):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--task", default="filtering")

    # Wandb related args
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument(
        "--entity", type=str, default="coarse-graining-mit", help="Wandb entity"
    )
    parser.add_argument(
        "--project", type=str, default="flexdock_filtering", help="Project name"
    )
    parser.add_argument("--run_name", type=str, default="flexdock_filtering_test")

    ############################################################################
    # Scaling args from parse_docking_args
    ############################################################################

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for model training.",
    )
    parser.add_argument(
        "--num_gpus_per_node",
        type=int,
        default=1,
        help="Number GPUs on each node for model training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # Training arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "fsdp_awp", "fsdp_ac", "fsdp_ac_awp"],
    )
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        default="full",
        choices=["full", "none", "grad", "hybrid"],
    )
    parser.add_argument(
        "--precision", default="32", type=str, help="Precision used for training"
    )

    ############################################################################
    # Trained Model
    ############################################################################

    parser.add_argument(
        "--original_model_dir",
        type=str,
        required=True,
        help="Directory where trained model was saved.",
    )
    parser.add_argument(
        "--model_ckpt", type=str, default="best_model.pt", help="Name of the model"
    )
    parser.add_argument(
        "--use_ema_weights",
        action="store_true",
        help="Whether to use EMA weights from trained model",
    )
    parser.add_argument(
        "--model_in_old_version",
        action="store_true",
        help="Whether the trained model is not in PyTorch Lightning",
    )

    ############################################################################
    # Arguments concerning the Dataset
    ############################################################################

    # Dataset init setup args
    parser.add_argument(
        "--use_original_model_cache",
        action="store_true",
        default=False,
        help="Whether to use processed graphs from original model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PDBBIND_atomCorrected",
        help="Directory where data is stored",
    )
    parser.add_argument(
        "--sample_cache_path", type=str, default="data/processed/filtering/samples/"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/processed/filtering/cacheNew",
        help="Cache path for filtering model",
    )
    parser.add_argument(
        "--split_train",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_train",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_val",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_val",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        default="data/splits/timesplit_test",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        action="store_true",
        default=False,
        help="drop_last arg of dataloader",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for preprocessing"
    )

    parser.add_argument(
        "--only_generate_data",
        action="store_true",
        help="Only generating data; no training",
    )

    # Dataset core args
    parser.add_argument(
        "--cache_ids_to_combine",
        nargs="+",
        default=None,
        help="cache ids that will be combined",
    )
    parser.add_argument("--cache_creation_id", default=None, help="Cache creation id")
    parser.add_argument(
        "--inference_steps", type=int, default=2, help="Number of denoising steps"
    )
    parser.add_argument("--samples_per_complex", type=int, default=3, help="")
    parser.add_argument("--sigma_schedule", type=str, default="expbeta", help="")
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
    parser.add_argument(
        "--balance",
        action="store_true",
        default=False,
        help="If this is true than we do not force the samples seen during training"
        "to be the same amount of negatives as positives",
    )
    parser.add_argument(
        "--rmsd_prediction",
        action="store_true",
        default=False,
        help="If this is true than we do not force the samples seen during training "
        "to be the same amount of negatives as positives",
    )
    parser.add_argument(
        "--rmsd_classification_cutoff",
        nargs="+",
        type=float,
        default=2,
        help="Ligand RMSD value below which a prediction is considered a postitive."
        "This can also be multiple cutoffs.",
    )
    parser.add_argument(
        "--aa_rmsd_classification_cutoff",
        nargs="+",
        type=float,
        default=1,
        help="All atom RMSD value below which a prediction is considered a postitive. "
        "This can also be multiple cutoffs.",
    )
    parser.add_argument(
        "--bb_rmsd_classification_cutoff",
        nargs="+",
        type=float,
        default=1,
        help="calpha atom (backbone) RMSD value below which a prediction is considered a postitive. "
        "This can also be multiple cutoffs.",
    )
    parser.add_argument(
        "--only_rmsd_labels",
        action="store_true",
        help="Only RMSD labels to use for training model",
    )
    parser.add_argument(
        "--limit_complexes",
        type=int,
        default=None,
        help="Number of complexes to limit training to. 0/None indicates all.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Number of times to repeat a complex",
    )
    parser.add_argument(
        "--chain_cutoff",
        type=float,
        default=10,
        help="Cutoff on whether to include non-interacting chains",
    )
    parser.add_argument(
        "--all_atoms",
        action="store_true",
        default=True,
        help="Whether to use the all atoms model",
    )
    parser.add_argument(
        "--receptor_radius",
        type=float,
        default=30,
        help="Cutoff on distances for receptor edges",
    )
    parser.add_argument(
        "--c_alpha_max_neighbors",
        type=int,
        default=10,
        help="Maximum number of neighbors for each residue",
    )
    parser.add_argument(
        "--atom_radius",
        type=float,
        default=5,
        help="Cutoff on distances for atom connections",
    )
    parser.add_argument(
        "--atom_max_neighbors",
        type=int,
        default=8,
        help="Maximum number of atom neighbours for receptor",
    )
    parser.add_argument(
        "--matching_popsize",
        type=int,
        default=20,
        help="Differential evolution popsize parameter in matching",
    )
    parser.add_argument(
        "--matching_maxiter",
        type=int,
        default=20,
        help="Differential evolution maxiter parameter in matching",
    )
    parser.add_argument(
        "--max_lig_size",
        type=int,
        default=None,
        help="Maximum number of heavy atoms in ligand",
    )
    parser.add_argument(
        "--remove_hs", action="store_true", default=True, help="remove Hs"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=1,
        help="Number of conformers to match to each ligand",
    )
    parser.add_argument(
        "--esm_embeddings_path",
        type=str,
        default=None,
        help="If this is set then the LM embeddings at that path will be used for the receptor features",
    )
    parser.add_argument(
        "--old_pocket_selection",
        action="store_true",
        default=False,
        help="Whether to compute pocket center based on holo",
    )
    parser.add_argument(
        "--pocket_reduction",
        action="store_true",
        default=False,
        help="Remove atoms from receptor that are not in the binding pocket",
    )
    parser.add_argument(
        "--pocket_buffer",
        type=float,
        default=10,
        help="Buffer that will be added to the radius of the pocket",
    )
    parser.add_argument(
        "--not_fixed_knn_radius_graph",
        action="store_true",
        default=False,
        help="Use knn graph and radius graph with closest neighbors instead of random ones as with radius_graph",
    )
    parser.add_argument(
        "--not_knn_only_graph",
        action="store_true",
        default=False,
        help="Use knn graph only and not restrict to a specific radius",
    )
    parser.add_argument(
        "--include_miscellaneous_atoms",
        action="store_true",
        default=False,
        help="include non amino acid atoms for the receptor",
    )
    parser.add_argument(
        "--apo_protein_file",
        type=str,
        default="protein_esmfold_aligned_tr_fix",
        help="Apo protein file identifier",
    )
    parser.add_argument(
        "--holo_protein_file",
        type=str,
        default="protein_processed_fix",
        help="specify the protein we will use to conformer match the --protein_file argument",
    )
    parser.add_argument(
        "--use_old_wrong_embedding_order",
        action="store_true",
        default=False,
        help="for backward compatibility to prevent the chain embedding order fix from https://github.com/gcorso/DiffDock/issues/58",
    )
    # parser.add_argument('--conformer_match_sidechains', action='store_true', default=False, help='Conformer match the sidechains from --protein_file with the --match_protein_file')
    parser.add_argument(
        "--conformer_match_score",
        type=str,
        default="dist",
        help='The scoring function used for conformer matching. Can be either "dist", "nearest" or "exp". All take the distance to the holo structure, nearest and exp also optimize steric clashes. Nearest takes the closest steric clash, exp weights the steric clashes with something similar to an rbf kernel.',
    )
    parser.add_argument(
        "--compare_true_protein",
        action="store_true",
        default=False,
        help="whether to calculate the rmsd to the holo structure (i.e., match_protein_file). this is only possible with flexible sidechains and if the proein_file is an apo structure. This is only applied to the validation set",
    )
    parser.add_argument(
        "--match_max_rmsd",
        type=float,
        default=2.0,
        help="Specify the maximum RMSD when conformer matching sidechains. "
        "This RMSD will only be calculated in the pocket with pocket_buffer."
        "This parameter only influences the training set, and has no impact on validation.",
    )
    parser.add_argument(
        "--use_original_conformer",
        action="store_true",
        default=False,
        help="use the original conformer structure for training"
        "if the matching rmsd is further away than match_max_rmsd value",
    )
    parser.add_argument(
        "--cache_individual",
        action="store_true",
        default=False,
        help="Cache each preprocessed complex individually instead of jointly",
    )
    parser.add_argument(
        "--use_bb_orientation_feats",
        action="store_true",
        default=False,
        help="Backbone orientation features",
    )
    parser.add_argument(
        "--only_nearby_residues_atomic",
        action="store_true",
        default=False,
        help="Whether to only use atoms from nearby residues when building the atom graph",
    )
    parser.add_argument(
        "--nearby_residues_atomic_radius", type=float, default=3.5, help=""
    )
    parser.add_argument("--nearby_residues_atomic_min", type=int, default=1, help="")
    parser.add_argument(
        "--no_torsion",
        action="store_true",
        default=False,
        help="If set only rigid matching",
    )

    # New dataset args for FlexDock-PyL
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument(
        "--use_esmflow_update",
        action="store_true",
        help="Whether to use new pipeline to process ESMFlow",
    )

    ############################################################################
    # Model related args
    ############################################################################

    # Model args for E3nn
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of interaction layers"
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=5.0,
        help="Radius cutoff for geometric graph",
    )
    parser.add_argument(
        "--scale_by_sigma",
        action="store_true",
        default=True,
        help="Whether to normalise the score",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=16,
        help="Number of hidden features per node of order 0",
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=4,
        help="Number of hidden features per node of order >0",
    )
    parser.add_argument(
        "--distance_embed_dim",
        type=int,
        default=32,
        help="Embedding size for the distance",
    )
    parser.add_argument(
        "--cross_distance_embed_dim",
        type=int,
        default=32,
        help="Embeddings size for the cross distance",
    )
    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        default=False,
        help="If set, it removes the batch norm",
    )
    parser.add_argument(
        "--use_second_order_repr",
        action="store_true",
        default=False,
        help="Whether to use only up to first order representations or also second",
    )
    parser.add_argument(
        "--cross_max_distance",
        type=float,
        default=80,
        help="Maximum cross distance in case not dynamic",
    )
    parser.add_argument(
        "--dynamic_max_cross",
        action="store_true",
        default=False,
        help="Whether to use the dynamic distance cutoff",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument(
        "--smooth_edges",
        action="store_true",
        default=False,
        help="Whether to apply additional smoothing weight to edges",
    )
    parser.add_argument(
        "--odd_parity",
        action="store_true",
        default=False,
        help="Whether to impose odd parity in output",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="sinusoidal",
        help="Type of diffusion time embedding",
    )
    parser.add_argument(
        "--sigma_embed_dim",
        type=int,
        default=32,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--embedding_scale",
        type=int,
        default=1000,
        help="Parameter of the diffusion time embedding",
    )
    parser.add_argument(
        "--sh_lmax",
        type=int,
        default=2,
        help="Size of the embedding of the diffusion time",
    )

    # Model new extras
    parser.add_argument(
        "--flexible_sidechains",
        action="store_true",
        default=False,
        help="Diffuse over side chain torsions for residues within flexdist of pocket",
    )
    parser.add_argument(
        "--flexdist",
        type=float,
        default=3.5,
        help="If a residue has at least one atom within flexdist of the pocket, it will be made flexible",
    )
    parser.add_argument(
        "--flexible_backbone",
        action="store_true",
        default=False,
        help="Learn bridge over pocket backbone",
    )
    parser.add_argument(
        "--flexdist_distance_metric",
        type=str,
        default="L2",
        help="Distance metric used to select residues within flexdist to pocket center",
    )
    parser.add_argument(
        "--separate_noise_schedule",
        action="store_true",
        default=False,
        help="Use different t for tr, rot, and tor",
    )
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sampling_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--rot_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--rot_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--tor_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--tor_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sidechain_tor_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sidechain_tor_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )
    parser.add_argument("--sidechain_tor_bridge", action="store_true", help="")
    parser.add_argument("--flexible_model", action="store_true", default=False, help="")
    parser.add_argument("--update_position_every", type=int, default=2, help="")
    parser.add_argument("--bridge_norm_clip", type=float, default=None, help="")
    parser.add_argument("--scale_by_t", action="store_true", default=False, help="")
    parser.add_argument(
        "--use_quaternions", action="store_true", default=False, help=""
    )
    parser.add_argument("--bb_tr_bridge_alpha", type=float, default=0.01, help="")
    parser.add_argument("--bb_rot_bridge_alpha", type=float, default=0.01, help="")
    parser.add_argument("--sc_tor_bridge_alpha", type=float, default=0.01, help="")

    # Debugging if drift works correctly
    parser.add_argument("--run_checks", action="store_true", default=False)
    parser.add_argument("--debug_backbone", action="store_true", default=False, help="")
    parser.add_argument(
        "--debug_sidechain", action="store_true", default=False, help=""
    )

    # IPA model args
    parser.add_argument("--ipa_model", action="store_true", default=False, help="")
    parser.add_argument("--n_ipa_heads", default=4, type=int)
    parser.add_argument("--n_ipa_qk_points", type=int, default=16)
    parser.add_argument("--n_ipa_v_points", type=int, default=8)
    parser.add_argument("--use_pair_feats", action="store_true", default=False)

    # Confidence Predictor in Model
    parser.add_argument("--new_confidence_version", action="store_true")
    parser.add_argument(
        "--include_confidence_prediction",
        action="store_true",
        default=False,
        help="Whether to predict an additional confidence metric for each predicted structure",
    )
    parser.add_argument(
        "--high_confidence_threshold",
        type=float,
        default=5.0,
        help="If this is 0 then the confidence predictor tries to predict the centroid_distance."
        "Otherwise it is the Ångström below which a prediction is labeled as good for supervising"
        "the confidence predictor",
    )
    parser.add_argument(
        "--tr_only_confidence",
        action="store_true",
        default=True,
        help="Whether to only supervise the confidence predictor with the translation",
    )
    parser.add_argument(
        "--confidence_no_batchnorm", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--confidence_dropout",
        type=float,
        default=0.0,
        help="MLP dropout in confidence readout",
    )

    parser.add_argument(
        "--not_fixed_center_conv", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--no_aminoacid_identities", action="store_true", default=False, help=""
    )

    ############################################################################
    # Training Related Args
    ############################################################################

    # Training Args
    parser.add_argument("--log_dir", type=str, default="workdir", help="")
    parser.add_argument(
        "--main_metric",
        type=str,
        default="accuracy",
        help="Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]",
    )
    parser.add_argument(
        "--main_metric_goal", type=str, default="max", help="Can be [min, max]"
    )
    parser.add_argument(
        "--transfer_weights", action="store_true", default=False, help=""
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--w_decay", type=float, default=0.0, help="")
    parser.add_argument(
        "--adamw",
        action="store_true",
        default=False,
        help="Use AdamW optimizer instead of Adam",
    )
    parser.add_argument("--scheduler", type=str, default="plateau", help="")
    parser.add_argument("--scheduler_patience", type=int, default=20, help="")
    parser.add_argument("--n_epochs", type=int, default=5, help="")
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Number of batches to limit training / val to.",
    )
    parser.add_argument(
        "--accumulate_grad",
        type=int,
        default=1,
        help="Number of steps after which to accumulate gradients",
    )
    parser.add_argument("--check_nan_grads", default=False, action="store_true")
    parser.add_argument(
        "--except_on_nan_grads",
        default=False,
        action="store_true",
        help="Raise exception on NaN gradients.",
    )
    parser.add_argument(
        "--skip_nan_grad_updates",
        default=False,
        action="store_true",
        help="Zero grads where a batch has NaN gradients.",
    )
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--val_inference_freq",
        type=int,
        default=None,
        help="Frequency of epochs for which to run expensive inference on val data",
    )
    parser.add_argument(
        "--filtering_weight", type=float, default=0.33, help="Weight of confidence loss"
    )
    parser.add_argument("--atom_lig_confidence", action="store_true")
    parser.add_argument(
        "--filtering_weight_atom",
        type=float,
        default=0.33,
        help="Weight of confidence loss for atom",
    )


def parse_relaxation_args(parser):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--task", default="relaxation")

    # Wandb related args
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument(
        "--entity", type=str, default="coarse-graining-mit", help="Wandb entity"
    )
    parser.add_argument(
        "--project", type=str, default="train_diffusion_relaxation", help="Project name"
    )
    parser.add_argument("--run_name", type=str, default="relaxation_test")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    ############################################################################
    # Trained Model
    ############################################################################

    parser.add_argument(
        "--original_model_dir",
        type=str,
        required=True,
        help="Directory where trained model was saved.",
    )
    parser.add_argument(
        "--model_ckpt", type=str, default="best_model.pt", help="Name of the model"
    )
    parser.add_argument(
        "--use_ema_weights",
        action="store_true",
        help="Whether to use EMA weights from trained model",
    )

    ############################################################################
    # Arguments concerning the Dataset
    ############################################################################

    # Pocket Args
    parser.add_argument(
        "--pocket_reduction",
        action="store_true",
        default=False,
        help="Remove atoms from receptor that are not in the binding pocket",
    )
    parser.add_argument(
        "--nearby_residues_atomic_radius", type=float, default=3.5, help=""
    )
    parser.add_argument("--nearby_residues_atomic_min", type=int, default=1, help="")
    parser.add_argument(
        "--pocket_buffer",
        type=float,
        default=10,
        help="Buffer that will be added to the radius of the pocket",
    )

    # Graph Args
    parser.add_argument(
        "--ligand_max_radius",
        type=float,
        default=5,
        help="Cutoff on distances for ligand connections",
    )
    parser.add_argument(
        "--ligand_max_neighbors",
        type=int,
        default=None,
        help="Maximum number of ligand neighbours",
    )
    parser.add_argument(
        "--receptor_max_radius",
        type=float,
        default=15,
        help="Cutoff on distances for receptor edges",
    )
    parser.add_argument(
        "--receptor_max_neighbors",
        type=int,
        default=24,
        help="Maximum number of neighbors for each residue",
    )
    parser.add_argument(
        "--atom_max_radius",
        type=float,
        default=5,
        help="Cutoff on distances for atom connections",
    )
    parser.add_argument(
        "--atom_max_neighbors",
        type=int,
        default=12,
        help="Maximum number of atom neighbours for receptor",
    )
    parser.add_argument(
        "--cross_max_radius",
        type=float,
        default=80,
        help="Cutoff on distances for cross connections",
    )
    parser.add_argument(
        "--cross_max_neighbors",
        type=int,
        default=None,
        help="Maximum number of cross neighbours",
    )

    # Embedding Args
    parser.add_argument(
        "--sigma_embed_type",
        type=str,
        default="sinusoidal",
        help="Type of diffusion time embedding",
    )
    parser.add_argument(
        "--sigma_embed_dim",
        type=int,
        default=32,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--sigma_embed_scale",
        type=int,
        default=1000,
        help="Parameter of the diffusion time embedding",
    )
    parser.add_argument(
        "--sh_lmax",
        type=int,
        default=2,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--distance_embed_dim",
        type=int,
        default=32,
        help="Embedding size for the distance",
    )
    parser.add_argument(
        "--cross_distance_embed_dim",
        type=int,
        default=32,
        help="Embeddings size for the cross distance",
    )
    parser.add_argument("--embed_radii", action="store_true", default=False)
    parser.add_argument("--embed_bounds", action="store_true", default=False)
    parser.add_argument(
        "--esm_embeddings_path",
        type=str,
        default=None,
        help="If this is set then the LM embeddings at that path will be used for the receptor features",
    )
    parser.add_argument(
        "--esm_embeddings_model",
        type=str,
        default=None,
        help="If this is set then the LM embeddings at that path will be used for the receptor features",
    )
    # Noise Args
    parser.add_argument("--n_conformers", type=int, default=None, help="")
    parser.add_argument("--tr_sigma", type=float, default=None, help="")
    parser.add_argument("--rot_sigma", type=float, default=None, help="")
    parser.add_argument("--lig_bond_sigma", type=float, default=None, help="")
    parser.add_argument("--lig_angle_sigma", type=float, default=None, help="")
    parser.add_argument("--lig_torsion_sigma", type=float, default=None, help="")
    parser.add_argument("--lig_fragment_sigma", type=float, default=None, help="")
    parser.add_argument("--bb_tr_sigma", type=float, default=None, help="")
    parser.add_argument("--bb_rot_sigma", type=float, default=None, help="")
    parser.add_argument("--sidechain_bond_sigma", type=float, default=None, help="")
    parser.add_argument("--sidechain_angle_sigma", type=float, default=None, help="")
    parser.add_argument("--sidechain_torsion_sigma", type=float, default=None, help="")
    parser.add_argument("--sidechain_fragment_sigma", type=float, default=None, help="")
    parser.add_argument("--ligand_sigma", type=float, default=None, help="")
    parser.add_argument("--atom_sigma", type=float, default=None, help="")

    # Sampling Args
    parser.add_argument("--rmsd_cutoff", type=float, default=None, help="")
    parser.add_argument("--lig_rmsd_cutoff", type=float, default=None, help="")
    parser.add_argument("--atom_rmsd_cutoff", type=float, default=None, help="")

    parser.add_argument("--sampling_kappa", type=float, default=None, help="")
    parser.add_argument("--sampling_epsilon", type=float, default=None, help="")
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for sampling t",
    )
    parser.add_argument(
        "--sampling_beta",
        type=float,
        default=1,
        help="Beta parameter of beta distribution for sampling t",
    )

    # Loss Args
    parser.add_argument("--x_zero_pred", action="store_true", default=False)
    parser.add_argument("--x_zero_loss", action="store_true", default=False)
    parser.add_argument("--align_pred", action="store_true", default=False)
    parser.add_argument("--ligand_loss_weight", type=float, default=1.0)
    parser.add_argument("--atom_loss_weight", type=float, default=1.0)
    parser.add_argument("--posebusters_loss", action="store_true", default=False)
    parser.add_argument("--posebusters_loss_weight", type=float, default=1.0, help="")
    parser.add_argument("--posebusters_loss_cutoff", type=float, default=None, help="")
    parser.add_argument("--posebusters_loss_alpha", type=float, default=None, help="")
    parser.add_argument("--bond_loss_buffer", type=float, default=0.25, help="")
    parser.add_argument("--angle_loss_buffer", type=float, default=0.25, help="")
    parser.add_argument("--steric_loss_buffer", type=float, default=0.2, help="")
    parser.add_argument("--crystal_loss", action="store_true", default=False)
    parser.add_argument("--crystal_loss_weight", type=float, default=1.0, help="")
    parser.add_argument("--crystal_loss_cutoff", type=float, default=None, help="")
    parser.add_argument("--crystal_loss_alpha", type=float, default=None, help="")
    parser.add_argument("--overlap_loss_weight", type=float, default=1.0, help="")
    parser.add_argument("--overlap_loss", action="store_true", default=False)
    parser.add_argument("--overlap_loss_cutoff", type=float, default=None, help="")
    parser.add_argument("--overlap_loss_alpha", type=float, default=None, help="")
    parser.add_argument("--overlap_loss_buffer", type=float, default=0.75, help="")

    # Training Args
    parser.add_argument("--log_dir", type=str, default="workdir", help="")
    parser.add_argument(
        "--main_metric",
        type=str,
        default="val/loss",
        help="Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]",
    )
    parser.add_argument(
        "--main_metric_goal", type=str, default="min", help="Can be [min, max]"
    )
    parser.add_argument(
        "--val_inference_freq",
        type=int,
        default=None,
        help="Frequency of epochs for which to run expensive inference on val data",
    )
    parser.add_argument(
        "--inference_earlystop_metric",
        type=str,
        default="valinf/steps_4/final_lig_rmsds_all_fraction_below_2.0",
        help="This is the metric that is addionally used when val_inference_freq is not None",
    )
    parser.add_argument(
        "--inference_earlystop_goal",
        type=str,
        default="max",
        help="Whether to maximize or minimize metric",
    )
    parser.add_argument(
        "--transfer_weights", action="store_true", default=False, help=""
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--w_decay", type=float, default=0.0, help="")
    parser.add_argument(
        "--adamw",
        action="store_true",
        default=False,
        help="Use AdamW optimizer instead of Adam",
    )
    parser.add_argument("--scheduler", type=str, default="plateau", help="")
    parser.add_argument("--scheduler_patience", type=int, default=20, help="")
    parser.add_argument("--n_epochs", type=int, default=5, help="")
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Number of batches to limit training / val to.",
    )
    parser.add_argument(
        "--accumulate_grad",
        type=int,
        default=1,
        help="Number of steps after which to accumulate gradients",
    )
    parser.add_argument("--check_nan_grads", default=False, action="store_true")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--filtering_weight", type=float, default=0.33, help="Weight of confidence loss"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether or not to use ema for the model weights",
    )
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.999,
        help="decay rate for the exponential moving average model parameters ",
    )

    # Dataset init setup args
    parser.add_argument(
        "--use_original_model_cache",
        action="store_true",
        default=False,
        help="Whether to use processed graphs from original model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PDBBIND_atomCorrected",
        help="Directory where data is stored",
    )
    parser.add_argument(
        "--sample_cache_path", type=str, default="data/processed/filtering/samples/"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/processed/filtering/cacheNew",
        help="Cache path for filtering model",
    )
    parser.add_argument(
        "--split_train",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_train",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_val",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_val",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        default="data/splits/timesplit_test",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        action="store_true",
        default=False,
        help="drop_last arg of dataloader",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for preprocessing"
    )
    parser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=1,
        help="Number of workers for preprocessing",
    )

    parser.add_argument(
        "--only_generate_data",
        action="store_true",
        help="Only generating data; no training",
    )

    # Dataset core args
    parser.add_argument(
        "--cache_ids_to_combine",
        nargs="+",
        default=None,
        help="cache ids that will be combined",
    )
    parser.add_argument("--cache_creation_id", default=None, help="Cache creation id")
    parser.add_argument(
        "--inference_steps", nargs="+", help="Number of denoising steps", required=True
    )
    parser.add_argument("--samples_per_complex", type=int, default=3, help="")
    parser.add_argument("--sigma_schedule", type=str, default="expbeta", help="")
    parser.add_argument(
        "--inf_sched_alpha",
        type=float,
        default=1,
        help="Alpha parameter of beta distribution for t sched",
    )
    parser.add_argument("--inf_sched_beta", type=float, default=1)

    parser.add_argument(
        "--limit_complexes",
        type=int,
        default=None,
        help="Number of complexes to limit training to. 0/None indicates all.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Number of times to repeat a complex",
    )
    parser.add_argument(
        "--chain_cutoff",
        type=float,
        default=10,
        help="Cutoff on whether to include non-interacting chains",
    )
    parser.add_argument(
        "--all_atoms",
        action="store_true",
        default=True,
        help="Whether to use the all atoms model",
    )

    parser.add_argument(
        "--receptor_radius",
        type=float,
        default=30,
        help="Cutoff on distances for receptor edges",
    )
    parser.add_argument(
        "--c_alpha_max_neighbors",
        type=int,
        default=10,
        help="Maximum number of neighbors for each residue",
    )
    parser.add_argument(
        "--atom_radius",
        type=float,
        default=5,
        help="Cutoff on distances for atom connections",
    )

    parser.add_argument(
        "--matching_popsize",
        type=int,
        default=20,
        help="Differential evolution popsize parameter in matching",
    )
    parser.add_argument(
        "--matching_maxiter",
        type=int,
        default=20,
        help="Differential evolution maxiter parameter in matching",
    )
    parser.add_argument(
        "--max_lig_size",
        type=int,
        default=None,
        help="Maximum number of heavy atoms in ligand",
    )
    parser.add_argument(
        "--remove_hs", action="store_true", default=True, help="remove Hs"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=1,
        help="Number of conformers to match to each ligand",
    )

    parser.add_argument(
        "--not_fixed_knn_radius_graph",
        action="store_true",
        default=False,
        help="Use knn graph and radius graph with closest neighbors instead of random ones as with radius_graph",
    )
    parser.add_argument(
        "--not_knn_only_graph",
        action="store_true",
        default=False,
        help="Use knn graph only and not restrict to a specific radius",
    )
    parser.add_argument(
        "--include_miscellaneous_atoms",
        action="store_true",
        default=False,
        help="include non amino acid atoms for the receptor",
    )
    parser.add_argument(
        "--apo_protein_file",
        type=str,
        default="protein_esmfold_aligned_tr_fix",
        help="Apo protein file identifier",
    )
    parser.add_argument(
        "--holo_protein_file",
        type=str,
        default="protein_processed_fix",
        help="specify the protein we will use to conformer match the --protein_file argument",
    )
    parser.add_argument(
        "--use_old_wrong_embedding_order",
        action="store_true",
        default=False,
        help="for backward compatibility to prevent the chain embedding order fix from https://github.com/gcorso/DiffDock/issues/58",
    )
    # parser.add_argument('--conformer_match_sidechains', action='store_true', default=False, help='Conformer match the sidechains from --protein_file with the --match_protein_file')
    parser.add_argument(
        "--conformer_match_score",
        type=str,
        default="dist",
        help='The scoring function used for conformer matching. Can be either "dist", "nearest" or "exp". All take the distance to the holo structure, nearest and exp also optimize steric clashes. Nearest takes the closest steric clash, exp weights the steric clashes with something similar to an rbf kernel.',
    )
    parser.add_argument(
        "--compare_true_protein",
        action="store_true",
        default=False,
        help="whether to calculate the rmsd to the holo structure (i.e., match_protein_file). this is only possible with flexible sidechains and if the proein_file is an apo structure. This is only applied to the validation set",
    )
    parser.add_argument(
        "--match_max_rmsd",
        type=float,
        default=2.0,
        help="Specify the maximum RMSD when conformer matching sidechains. "
        "This RMSD will only be calculated in the pocket with pocket_buffer."
        "This parameter only influences the training set, and has no impact on validation.",
    )
    parser.add_argument(
        "--use_original_conformer",
        action="store_true",
        default=False,
        help="use the original conformer structure for training"
        "if the matching rmsd is further away than match_max_rmsd value",
    )
    parser.add_argument(
        "--cache_individual",
        action="store_true",
        default=False,
        help="Cache each preprocessed complex individually instead of jointly",
    )
    parser.add_argument(
        "--use_bb_orientation_feats",
        action="store_true",
        default=False,
        help="Backbone orientation features",
    )
    parser.add_argument(
        "--no_torsion",
        action="store_true",
        default=False,
        help="If set only rigid matching",
    )

    # New dataset args for FlexDock-PyL
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument(
        "--use_esmflow_update",
        action="store_true",
        help="Whether to use new pipeline to process ESMFlow",
    )

    ############################################################################
    # Model related args
    ############################################################################

    # Model args for E3nn
    parser.add_argument("--num_prot_emb_layers", type=int, default=0, help="")
    parser.add_argument(
        "--embed_also_ligand", action="store_true", default=True, help=""
    )
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of interaction layers"
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=16,
        help="Number of hidden features per node of order 0",
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=4,
        help="Number of hidden features per node of order >0",
    )

    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        default=False,
        help="If set, it removes the batch norm",
    )
    parser.add_argument(
        "--use_second_order_repr",
        action="store_true",
        default=False,
        help="Whether to use only up to first order representations or also second",
    )
    parser.add_argument(
        "--cross_max_distance",
        type=float,
        default=80,
        help="Maximum cross distance in case not dynamic",
    )
    parser.add_argument(
        "--dynamic_max_cross",
        action="store_true",
        default=False,
        help="Whether to use the dynamic distance cutoff",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument(
        "--smooth_edges",
        action="store_true",
        default=False,
        help="Whether to apply additional smoothing weight to edges",
    )
    parser.add_argument(
        "--odd_parity",
        action="store_true",
        default=False,
        help="Whether to impose odd parity in output",
    )
    parser.add_argument("--tp_weights_layers", type=int, default=2, help="")

    parser.add_argument(
        "--no_differentiate_convolutions", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--reduce_pseudoscalars", action="store_true", default=False, help=""
    )

    # Model new extras
    parser.add_argument(
        "--flexible_sidechains",
        action="store_true",
        default=False,
        help="Diffuse over side chain torsions for residues within flexdist of pocket",
    )
    parser.add_argument(
        "--flexdist",
        type=float,
        default=3.5,
        help="If a residue has at least one atom within flexdist of the pocket, it will be made flexible",
    )
    parser.add_argument(
        "--flexible_backbone",
        action="store_true",
        default=False,
        help="Learn bridge over pocket backbone",
    )
    parser.add_argument(
        "--flexdist_distance_metric",
        type=str,
        default="L2",
        help="Distance metric used to select residues within flexdist to pocket center",
    )
    parser.add_argument(
        "--separate_noise_schedule",
        action="store_true",
        default=False,
        help="Use different t for tr, rot, and tor",
    )
    parser.add_argument(
        "--asyncronous_noise_schedule", action="store_true", default=False, help=""
    )
    parser.add_argument("--flexible_model", action="store_true", default=False, help="")
    parser.add_argument("--update_position_every", type=int, default=2, help="")
    parser.add_argument("--bridge_norm_clip", type=float, default=None, help="")
    parser.add_argument("--scale_by_t", action="store_true", default=False, help="")
    parser.add_argument(
        "--use_quaternions", action="store_true", default=False, help=""
    )

    # IPA model args
    parser.add_argument("--ipa_model", action="store_true", default=False, help="")
    parser.add_argument("--n_ipa_heads", default=4, type=int)
    parser.add_argument("--n_ipa_qk_points", type=int, default=16)
    parser.add_argument("--n_ipa_v_points", type=int, default=8)
    parser.add_argument("--use_pair_feats", action="store_true", default=False)

    parser.add_argument(
        "--not_fixed_center_conv", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--no_aminoacid_identities", action="store_true", default=False, help=""
    )


def parse_train_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Subparsers based on task")

    parser_docking = subparsers.add_parser("docking")
    parse_docking_args(parser_docking)

    parser_filtering = subparsers.add_parser("filtering")
    parse_filtering_args(parser_filtering)

    parser_relaxation = subparsers.add_parser("relaxation")
    parse_relaxation_args(parser_relaxation)

    args = parser.parse_args()

    if args.task == "docking":
        # Parsing related checks and modifications
        if not args.flexible_backbone:
            if args.bb_random_prior:
                print(
                    "Backbone prior should not be true when flexible_backbone=False. Setting it to False"
                )
                args.bb_random_prior = False
                print()

        if args.use_new_pipeline:
            print("New pipeline being used....")

        if args.flexible_sidechains and not args.all_atoms:
            raise ValueError(
                "--all_atoms needs to be activated if --flexible_sidechains is used"
            )

        if args.pocket_reduction and args.flexible_sidechains:
            if args.flexdist > args.pocket_buffer:
                print(
                    "WARN: The specified flexdist of",
                    args.flexdist,
                    "is larger than the pocket_buffer of",
                    args.pocket_buffer,
                )

        if args.compare_true_protein and not args.flexible_sidechains:
            raise ValueError(
                "Comparing to a true protein file is only meaningful when there are flexible sidechains"
            )

        if (
            args.conformer_match_score != "dist"
            and args.conformer_match_score != "nearest"
            and args.conformer_match_score != "exp"
        ):
            raise ValueError(
                "Conformer match score must be either 'dist', 'nearest' or 'exp"
            )

    return args
