import os
import binascii


def get_full_cache_path(
    cache_path,
    split_path,
    receptor_radius,
    c_alpha_max_neighbors: int,
    atom_radius: float,
    atom_max_neighbors: int,
    limit_complexes: int,
    matching: bool,
    max_lig_size: int,
    remove_hs: bool,
    num_conformers: int = 1,
    all_atoms: bool = False,
    pocket_reduction: bool = False,
    old_pocket_selection: bool = False,
    pocket_buffer=10,
    chain_cutoff=10,
    nearby_residues_atomic_radius: float = 3.5,
    nearby_residues_atomic_min: int = 1,
    knn_only_graph: bool = False,
    fixed_knn_radius_graph: bool = False,
    apo_protein_file="protein_esmfold_aligned_tr",
    holo_protein_file="protein_processed",
    match_max_rmsd=None,
    add_maxrmsd_to_cache=False,
    include_miscellaneous_atoms: bool = False,
    esm_embeddings_path: str = None,
    protein_path_list=None,
    ligand_descriptions=None,
    use_new_pipeline: bool = False,
):
    if limit_complexes is None:
        print("Provided limit_complexes is None. Setting to 0")
        limit_complexes = 0

    cache_path_base = cache_path

    if matching or protein_path_list is not None and ligand_descriptions is not None:
        cache_path_base += "_torsion"
    if all_atoms:
        cache_path_base += "_allatoms"

    pocket_description = ""
    if pocket_reduction:
        if old_pocket_selection:
            pocket_description = "_reduced" + str(pocket_buffer)
        else:
            pocket_description = (
                "_reduced_Nr"
                + str(nearby_residues_atomic_radius)
                + "_Nmin"
                + str(nearby_residues_atomic_min)
                + "_Pr"
                + str(pocket_buffer)
            )

    rmsd_description = ""
    if add_maxrmsd_to_cache:
        if match_max_rmsd is None:
            rmsd_description += "_maxrmsdNone"
        else:
            rmsd_description += f"_maxrmsd{float(match_max_rmsd)}"

    keep_local_structures = False  # TODO this is default

    if use_new_pipeline:
        version = "v3"
    else:
        version = "v2"

    if not use_new_pipeline:
        full_cache_path = os.path.join(
            cache_path_base,
            f"{version}_limit{limit_complexes}"
            f"_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
            f"_maxLigSize{max_lig_size}_H{int(not remove_hs)}"
            f"_recRad{receptor_radius}_recMax{c_alpha_max_neighbors}"
            f"_chainCutoff{chain_cutoff}"
            + (
                ""
                if not all_atoms
                else f"_atomRad{atom_radius}_atomMax{atom_max_neighbors}"
            )
            + ("" if not matching or num_conformers == 1 else f"_confs{num_conformers}")
            + ("" if esm_embeddings_path is None else "_esmEmbeddings")
            + ("" if not keep_local_structures else "_keptLocalStruct")
            + (
                ""
                if protein_path_list is None or ligand_descriptions is None
                else str(
                    binascii.crc32(
                        "".join(ligand_descriptions + protein_path_list).encode()
                    )
                )
            )
            + (
                ""
                if holo_protein_file == "protein_processed"
                else "_" + holo_protein_file
            )
            + (
                ""
                if apo_protein_file == "protein_esmfold_aligned_tr"
                else "_" + apo_protein_file
            )
            + pocket_description
            + (
                ""
                if not fixed_knn_radius_graph
                else ("_fixedKNN" if not knn_only_graph else "_fixedKNNonly")
            )
            + ("" if not include_miscellaneous_atoms else "_miscAtoms")
            + rmsd_description,
        )
    else:
        full_cache_path = os.path.join(
            cache_path_base,
            f"{version}_limit{limit_complexes}"
            f"_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
            f"_maxLigSize{max_lig_size}_H{int(not remove_hs)}"
            + ("" if not matching or num_conformers == 1 else f"_confs{num_conformers}")
            + ("" if esm_embeddings_path is None else "_esmEmbeddings")
            + (
                ""
                if holo_protein_file == "protein_processed"
                else "_" + holo_protein_file
            )
            + (
                ""
                if apo_protein_file == "protein_esmfold_aligned_tr"
                else "_" + apo_protein_file
            )
            + (
                ""
                if not fixed_knn_radius_graph
                else ("_fixedKNN" if not knn_only_graph else "_fixedKNNonly")
            )
            + ("" if not include_miscellaneous_atoms else "_miscAtoms")
            + rmsd_description,
        )

    return full_cache_path


# This function
def gather_cache_path_kwargs(args, split_path):
    cache_path_kwargs = {
        "cache_path": args.cache_path,
        "split_path": split_path,
        "limit_complexes": args.limit_complexes,
        "chain_cutoff": args.chain_cutoff,
        "receptor_radius": args.receptor_radius,
        "c_alpha_max_neighbors": args.c_alpha_max_neighbors,
        "remove_hs": args.remove_hs,
        "max_lig_size": args.max_lig_size,
        "matching": not args.no_torsion,
        "all_atoms": args.all_atoms,
        "atom_radius": args.atom_radius,
        "atom_max_neighbors": args.atom_max_neighbors,
        "esm_embeddings_path": args.esm_embeddings_path,
        "pocket_reduction": args.pocket_reduction,
        "old_pocket_selection": False
        if not hasattr(args, "old_pocket_selection")
        else args.old_pocket_selection,
        "pocket_buffer": args.pocket_buffer,
        "nearby_residues_atomic_radius": args.nearby_residues_atomic_radius,
        "nearby_residues_atomic_min": args.nearby_residues_atomic_min,
        "include_miscellaneous_atoms": False
        if not hasattr(args, "include_miscellaneous_atoms")
        else args.include_miscellaneous_atoms,
        "holo_protein_file": args.holo_protein_file,
        "apo_protein_file": args.apo_protein_file,
        "knn_only_graph": False
        if not hasattr(args, "not_knn_only_graph")
        else not args.not_knn_only_graph,
        "fixed_knn_radius_graph": False
        if not hasattr(args, "not_fixed_knn_radius_graph")
        else not args.not_fixed_knn_radius_graph,
        "use_new_pipeline": getattr(args, "use_new_pipeline", False),
    }

    if "train" in split_path:
        cache_path_kwargs["match_max_rmsd"] = args.match_max_rmsd
        cache_path_kwargs["num_conformers"] = args.num_conformers
        cache_path_kwargs["add_maxrmsd_to_cache"] = (
            False
            if not hasattr(args, "add_maxrmsd_to_cache_path")
            else args.add_maxrmsd_to_cache_path
        )

    return cache_path_kwargs
