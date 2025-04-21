import argparse
import logging

from flexdock.data.modules.training.pipeline import (
    TrainingDataPipeline,
    TrainingPipelineConfig,
)
from flexdock.data.feature.featurizer import FeaturizerConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="pdbbind", help="Dataset")
    parser.add_argument("--data_dir", type=str, default="data/PDBBIND_atomCorrected")
    parser.add_argument("--cache_path", type=str, default="data/processed/cachev2")
    parser.add_argument(
        "--split_path", type=str, default="data/splits/timesplit_no_lig_overlap_train"
    )

    parser.add_argument("--limit_complexes", type=int, default=None)
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
        "--no_torsion",
        action="store_true",
        default=False,
        help="If set only rigid matching",
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
        "--match_max_rmsd",
        type=float,
        default=None,
        help="Specify the maximum RMSD when conformer matching sidechains. "
        "This RMSD will only be calculated in the pocket with pocket_buffer. "
        "This parameter only influences the training set, and has no impact on validation.",
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

    parser.add_argument(
        "--conformer_match_sidechains",
        action="store_true",
        default=False,
        help="Conformer match the sidechains from --protein_file with the --match_protein_file",
    )
    parser.add_argument(
        "--conformer_match_score",
        type=str,
        default="dist",
        help="The scoring function used for conformer matching. "
        'Can be either "dist", "nearest" or "exp". '
        "All take the distance to the holo structure, nearest and exp also optimize steric clashes."
        "Nearest takes the closest steric clash, exp weights the steric clashes with something similar to an rbf kernel.",
    )
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
        "--apo_protein_file", type=str, default="protein_esmfold_aligned_tr", help=""
    )
    parser.add_argument(
        "--holo_protein_file", type=str, default="protein_processed", help=""
    )
    parser.add_argument(
        "--only_nearby_residues_atomic", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--nearby_residues_atomic_radius", type=float, default=3.5, help=""
    )
    parser.add_argument("--nearby_residues_atomic_min", type=int, default=1, help="")
    parser.add_argument(
        "--compare_true_protein",
        action="store_true",
        default=False,
        help="whether to calculate the rmsd to the holo structure (i.e., match_protein_file). this is only possible with flexible sidechains and if the proein_file is an apo structure. This is only applied to the validation set",
    )

    parser.add_argument("--use_origpos_pocket", action="store_true")
    parser.add_argument("--add_nearby_residues_pocket", action="store_true")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for preprocessing data",
    )
    parser.add_argument(
        "--use_new_pipeline",
        action="store_true",
        help="Whether to use the old processing pipeline.",
    )

    args = parser.parse_args()

    # Parsing related checks and modifications
    if not args.flexible_backbone:
        if args.bb_random_prior:
            print(
                "Backbone prior should not be true when flexible_backbone=False. Setting it to False"
            )
            args.bb_random_prior = False
            print()

    if args.use_new_pipeline:
        print("New pipeline being used...")

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


def preprocess_dataset():
    args = parse_args()

    logging.getLogger().setLevel("INFO")

    config = TrainingPipelineConfig(
        dataset=args.dataset,
        complex_file=args.split_path,
        data_dir=args.data_dir,
        esm_embeddings_path=args.esm_embeddings_path,
        cache_path=args.cache_path,
        apo_protein_file=args.apo_protein_file,
        holo_protein_file=args.holo_protein_file,
        num_workers=args.num_workers,
    )

    featurizer_cfg = FeaturizerConfig(
        matching=not args.no_torsion,
        popsize=args.matching_popsize,
        maxiter=args.matching_maxiter,
        keep_original=True,
        remove_hs=True,
        num_conformers=1,
        max_lig_size=None,
        flexible_backbone=args.flexible_backbone,
        flexible_sidechains=args.flexible_sidechains,
    )

    pipeline = TrainingDataPipeline(config=config, featurizer_cfg=featurizer_cfg)
    pipeline.process_all_complexes()


if __name__ == "__main__":
    preprocess_dataset()
