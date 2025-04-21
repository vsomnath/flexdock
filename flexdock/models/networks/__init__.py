from flexdock.sampling.docking.diffusion import get_timestep_embedding
from flexdock.models.networks.score_network import (
    TensorProductScoreModel as AAScoreModel,
)
from flexdock.models.networks.flow_network import (
    TensorProductFlowModel as EuclideanAAFlowModel,
)


def get_model(
    args, device, t_to_sigma, confidence_mode=False, relaxation: bool = False
):
    if relaxation:
        timestep_emb_func = get_timestep_embedding(
            embedding_type=args.sigma_embed_type,
            embedding_dim=args.sigma_embed_dim,
            embedding_scale=args.sigma_embed_scale,
        )

        model = EuclideanAAFlowModel(
            device=device,
            timestep_emb_func=timestep_emb_func,
            lig_max_radius=args.ligand_max_radius,
            lig_max_neighbors=args.ligand_max_neighbors,
            rec_max_radius=args.receptor_max_radius,
            rec_max_neighbors=args.receptor_max_neighbors,
            atom_max_radius=args.atom_max_radius,
            atom_max_neighbors=args.atom_max_neighbors,
            cross_max_radius=args.cross_max_radius,
            cross_max_neighbors=args.cross_max_neighbors,
            sigma_embed_dim=args.sigma_embed_dim,
            distance_embed_dim=args.distance_embed_dim,
            cross_distance_embed_dim=args.cross_distance_embed_dim,
            lm_embedding_type="precomputed"
            if args.esm_embeddings_path is not None
            else args.esm_embeddings_model,
            smooth_edges=args.smooth_edges,
            num_prot_emb_layers=args.num_prot_emb_layers,
            embed_also_ligand=args.embed_also_ligand,
            num_conv_layers=args.num_conv_layers,
            ns=args.ns,
            nv=args.nv,
            differentiate_convolutions=not args.no_differentiate_convolutions,
            sh_lmax=args.sh_lmax,
            use_second_order_repr=args.use_second_order_repr,
            reduce_pseudoscalars=args.reduce_pseudoscalars,
            tp_weights_layers=args.tp_weights_layers,
            norm_type=args.norm_type,
            norm_affine=args.norm_affine,
            batch_norm=not args.no_batch_norm,
            dropout=args.dropout,
            embed_radii=False if "embed_radii" not in args else args.embed_radii,
            embed_bounds=False if "embed_bounds" not in args else args.embed_bounds,
        )
        return model

    else:
        assert args.all_atoms, "Only all atoms supported in this codebase"
        model_class = AAScoreModel

        timestep_emb_func = get_timestep_embedding(
            embedding_type=getattr(args, "embedding_type", "sinusoidal"),
            embedding_dim=args.sigma_embed_dim,
            embedding_scale=args.embedding_scale
            if hasattr(args, "embedding_type")
            else 10000,
        )

        lm_embedding_type = None
        if args.esm_embeddings_path is not None:
            lm_embedding_type = "esm"

        model_params = {
            "in_lig_edge_features": args.in_lig_edge_features,
            "t_to_sigma": t_to_sigma,
            "timestep_emb_func": timestep_emb_func,
            "no_torsion": args.no_torsion,
            "num_conv_layers": args.num_conv_layers,
            "lig_max_radius": args.max_radius,
            "scale_by_sigma": args.scale_by_sigma,
            "sh_lmax": args.sh_lmax,
            "sigma_embed_dim": args.sigma_embed_dim,
            "ns": args.ns,
            "nv": args.nv,
            "distance_embed_dim": args.distance_embed_dim,
            "cross_distance_embed_dim": args.cross_distance_embed_dim,
            "batch_norm": not args.no_batch_norm,
            "norm_type": args.norm_type,
            "dropout": args.dropout,
            "use_second_order_repr": args.use_second_order_repr,
            "cross_max_distance": args.cross_max_distance,
            "dynamic_max_cross": args.dynamic_max_cross,
            "separate_noise_schedule": args.separate_noise_schedule,
            "smooth_edges": getattr(args, "smooth_edges", False),
            "odd_parity": getattr(args, "odd_parity", False),
            "lm_embedding_type": lm_embedding_type,
            "confidence_mode": confidence_mode,
            "asyncronous_noise_schedule": False,
            "num_confidence_outputs": len(args.rmsd_classification_cutoff) + 1
            if hasattr(args, "rmsd_classification_cutoff")
            and isinstance(args.rmsd_classification_cutoff, list)
            else 1,
            "fixed_center_conv": not args.not_fixed_center_conv
            if hasattr(args, "not_fixed_center_conv")
            else False,
            "no_aminoacid_identities": getattr(args, "no_aminoacid_identities", False),
            "flexible_sidechains": args.flexible_sidechains,
            "flexible_backbone": args.flexible_backbone,
            "c_alpha_radius": args.receptor_radius,
            "c_alpha_max_neighbors": args.c_alpha_max_neighbors,
            "atom_radius": args.atom_radius,
            "atom_max_neighbors": args.atom_max_neighbors,
            "sidechain_tor_bridge": args.sidechain_tor_bridge,
            "use_bb_orientation_feats": args.use_bb_orientation_feats,
            "only_nearby_residues_atomic": args.only_nearby_residues_atomic,
            "activation_func": args.activation_func,
            "norm_affine": args.norm_affine,
            "clamped_norm_min": args.clamped_norm_min,
        }

        if args.all_atoms:
            model_params["new_confidence_version"] = getattr(
                args, "new_confidence_version", False
            )
            model_params["atom_lig_confidence"] = getattr(
                args, "atom_lig_confidence", False
            )

        model = model_class(**model_params)
        return model
