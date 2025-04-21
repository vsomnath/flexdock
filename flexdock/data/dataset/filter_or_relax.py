import argparse
import copy
from collections import defaultdict
from functools import partial
import math
import os
import pickle
import yaml

from multiprocessing.pool import Pool
import numpy as np
import random
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from tqdm import tqdm


# from flexdock.data.dataset.base import get_docking_dataset_from_args
from flexdock.data.dataset.utils import get_full_cache_path, gather_cache_path_kwargs
from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.conformers.protein import scRMSD
from flexdock.data.feature.helpers import (
    filter_flexible_residues,
    to_atom_grid_torch,
)
from flexdock.data.feature.protein import get_nearby_atom_mask

from flexdock.data.transforms.docking import bb_priors
from flexdock.data.pipeline import get_pipeline

from flexdock.geometry.ops import rigid_transform_kabsch_numpy
from flexdock.sampling.docking import sampling
from flexdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl, set_time


get_docking_dataset_from_args = None


def _save_positions_rmsds(positions, rmsds, basename, cache_creation_id=None):
    if cache_creation_id is not None:
        basename += f"_id{cache_creation_id}"
    basename += ".pkl"

    with open(basename, "wb") as f:
        pickle.dump((positions, rmsds), f)


def get_args_and_cache_path(original_model_dir, split_path):
    with open(f"{original_model_dir}/model_parameters.yml", "r") as f:
        model_args = argparse.Namespace(**yaml.full_load(f))

        if not hasattr(model_args, "lig_transform_type"):
            model_args.lig_transform_type = "diffusion"

    cache_path_kwargs = gather_cache_path_kwargs(args=model_args, split_path=split_path)
    full_cache_path = get_full_cache_path(**cache_path_kwargs)
    return model_args, full_cache_path


class PostDockingDataset(Dataset):
    def __init__(
        self,
        sample_cache_path,
        filtering_args,
        original_model_dir,
        split_path,
        inference_steps,
        samples_per_complex,
        limit_complexes,
        multiplicity,
        all_atoms,
        inf_sched_alpha=1,
        inf_sched_beta=1,
        sigma_schedule="expbeta",
        cache_ids_to_combine=None,
        cache_creation_id=None,
        include_miscellaneous_atoms: bool = False,
        asyncronous_noise_schedule: bool = False,
        cache_individual: bool = False,
        device: str = "cpu",
        pipeline=None,
    ):
        super().__init__()
        self.original_model_dir = original_model_dir
        self.filtering_args = filtering_args
        self.split_path = split_path

        self.inference_steps = inference_steps
        self.samples_per_complex = samples_per_complex
        self.inf_sched_alpha = inf_sched_alpha
        self.inf_sched_beta = inf_sched_beta
        self.sigma_schedule = sigma_schedule
        self.asyncronous_noise_schedule = asyncronous_noise_schedule

        self.all_atoms = all_atoms
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity

        self.cache_individual = cache_individual
        self.cache_creation_id = cache_creation_id
        self.cache_ids_to_combine = cache_ids_to_combine

        self.fixed_step = None
        self.fixed_sample = None
        self.device = device

        # 1. Prepare cache for complex graphs
        self.set_graph_cache_path(split_path=split_path)

        # 2. Prepare sample_cache_path
        self.set_samples_cache_path(sample_cache_path, split_path)

        ########################################################################
        # Check if preprocessing needed to get cache
        ########################################################################
        if not self.check_preprocessing_needed(cache_path=self.complex_graphs_cache):
            print("[Filtering Model]: Preprocessing graphs for filtering model...")

        ########################################################################
        # Generate samples from trained model if not found
        ########################################################################
        complex_names = self.gather_complex_names(split_path, pipeline=pipeline)

        if not self.check_docked_outputs_exist(
            self.samples_cache_path, complex_names_all=complex_names
        ):
            print("Generating samples from provided model...")
            os.makedirs(self.samples_cache_path, exist_ok=True)
            self.generate_data_for_model()

        ########################################################################
        # Loading inputs for the filtering model
        ########################################################################

        if not self.cache_individual:
            with open(
                os.path.join(self.complex_graphs_cache, "heterographs.pkl"), "rb"
            ) as f:
                complex_graphs = pickle.load(f)
            self.complex_graph_dict = {d.name: d for d in complex_graphs}

        self.all_samples_per_complex = samples_per_complex * (
            1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine)
        )

        self.maybe_combine_cache_ids()
        available_complexes = list(
            set(complex_names).intersection(set(self.dataset_names))
        )

        if self.limit_complexes is not None and self.limit_complexes > 0:
            if len(available_complexes) > self.limit_complexes:
                available_complexes = available_complexes[: self.limit_complexes]

        self.available_complexes = available_complexes

    def set_graph_cache_path(self, split_path):
        raise NotImplementedError("Subclasses must implement for themselves")

    def set_samples_cache_path(self, sample_cache_path, split_path):
        raise NotImplementedError("Subclasses must implement for themselves")

    def gather_complex_names(self, split_path, pipeline=None):
        raise NotImplementedError("Subclasses must implement for themselves")

    def check_docked_outputs_exist(self, cache_path, complex_names_all=None):
        if self.cache_ids_to_combine is not None:
            print(
                "Cache ids to combine is not None. This presumes the associated cache ids are created"
            )
            return True

        if not self.cache_individual:
            ligand_basename = f"{cache_path}/ligand_positions"
            if self.cache_creation_id is not None:
                ligand_basename += f"_id{self.cache_creation_id}"

            if not os.path.exists(f"{ligand_basename}.pkl"):
                return False

        else:
            if complex_names_all is None:
                complex_names_all = read_strings_from_txt(self.split_path)
                if (
                    self.filtering_args.limit_complexes is not None
                    and self.filtering_args.limit_complexes != 0
                ):
                    complex_names_all = complex_names_all[: self.limit_complexes]
            else:
                print("Complex names were supplied.")

            ligand_suffix = "ligand_positions"
            if self.cache_creation_id is not None:
                ligand_suffix += f"_id{self.cache_creation_id}"

            complexes_available = [
                complex_name
                for complex_name in complex_names_all
                if os.path.exists(f"{cache_path}/{complex_name}_{ligand_suffix}.pkl")
            ]

            if not len(complexes_available):
                print(f"No complexes found in {cache_path}")
                return False

        return True

    def generate_data_for_model(self):
        from flexdock.models.pl_modules.docking import load_pretrained_model

        print("Generating samples from the diffusion model ")
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)
        ckpt_file = f"{self.original_model_dir}/{self.filtering_args.model_ckpt}"

        model = load_pretrained_model(
            model_args=self.original_model_args,
            t_to_sigma=t_to_sigma,
            ckpt_file=ckpt_file,
            use_ema_weights=self.filtering_args.use_ema_weights,
            model_in_old_version=self.filtering_args.model_in_old_version,
            device=self.device,
        )

        schedules = sampling.get_schedules(
            inference_steps=self.inference_steps,
            bb_tr_bridge_alpha=self.original_model_args.bb_tr_bridge_alpha,
            bb_rot_bridge_alpha=self.original_model_args.bb_rot_bridge_alpha,
            sc_tor_bridge_alpha=self.original_model_args.sc_tor_bridge_alpha,
            sidechain_tor_bridge=self.original_model_args.sidechain_tor_bridge,
            lig_transform_type=self.original_model_args.lig_transform_type,
            inf_sched_alpha=self.inf_sched_alpha,
            inf_sched_beta=self.inf_sched_beta,
            sigma_schedule=self.sigma_schedule,
            asyncronous_noise_schedule=self.asyncronous_noise_schedule,
        )
        print("[Filtering Model:]. Generating samples with the following t schedule")
        for key in schedules:
            print(key, schedules[key])

        # TODO: Hacky, change later
        if "train" in self.split_path:
            mode = "train"
        elif "val" in self.split_path:
            mode = "val"
        elif "test" in self.split_path:
            mode = "test"

        pipeline = get_pipeline(args=self.original_model_args, mode=mode)

        base_dataset = get_docking_dataset_from_args(
            args=self.original_model_args,
            dataset_name=self.original_model_args.dataset,
            split_path=self.split_path,
            data_dir=self.filtering_args.data_dir,
            multiplicity=self.multiplicity,
            limit_complexes=self.limit_complexes,
            cache_path=self.complex_graphs_cache_base,  # Uses the original complex graphs
            pipeline=pipeline,
        )

        inf_loader = DataLoader(dataset=base_dataset, batch_size=1, shuffle=False)
        num_inf_complexes = len(base_dataset)

        # inf_complexes = [base_dataset.get(idx) for idx in range(len(base_dataset))]
        # inf_dataset = ListDataset(inf_complexes)
        # inf_loader = DataLoader(dataset=inf_dataset, batch_size=1, shuffle=False)

        rmsds, aa_rmsds, bb_rmsds = [], [], []
        if not self.cache_individual:
            full_ligand_positions, full_atom_positions, calpha_positions = [], [], []
        names = []

        bb_prior = bb_priors.construct_bb_prior(self.original_model_args)

        with tqdm(total=num_inf_complexes) as pbar:
            for idx, orig_complex_graph in tqdm(
                enumerate(inf_loader), total=num_inf_complexes
            ):
                if self.cache_individual:
                    rmsds_complex, aa_rmsds_complex, bb_rmsds_complex = [], [], []

                data_list = [
                    copy.deepcopy(orig_complex_graph)
                    for _ in range(self.samples_per_complex)
                ]

                sampling.randomize_position(
                    data_list=data_list,
                    no_torsion=self.original_model_args.no_torsion,
                    no_random=False,
                    tr_sigma_max=self.original_model_args.tr_sigma_max,
                    flexible_sidechains=self.original_model_args.flexible_sidechains,
                    flexible_backbone=self.original_model_args.flexible_backbone,
                    sidechain_tor_bridge=self.original_model_args.sidechain_tor_bridge,
                    use_bb_orientation_feats=self.original_model_args.use_bb_orientation_feats,
                    prior=bb_prior,
                )

                predictions_list = None
                failed_convergence_counter = 0
                while predictions_list == None:
                    try:
                        predictions_list, _ = sampling.sampling(
                            data_list=data_list,
                            model=model.model,
                            inference_steps=self.original_model_args.inference_steps,
                            schedules=schedules,
                            sidechain_tor_bridge=self.original_model_args.sidechain_tor_bridge,
                            device=self.device,
                            t_to_sigma=t_to_sigma,
                            model_args=self.original_model_args,
                            use_bb_orientation_feats=self.original_model_args.use_bb_orientation_feats,
                            asyncronous_noise_schedule=self.asyncronous_noise_schedule,
                            no_final_step_noise=True,
                            filtering_data_list=None,
                            filtering_model_args=None,
                            visualization_list=None,
                            sidechain_visualization_list=None,
                            debug_backbone=False,
                            debug_sidechain=False,
                        )
                    except Exception as e:
                        if "failed to converge" in str(e):
                            failed_convergence_counter += 1
                            if failed_convergence_counter > 5:
                                print(
                                    "| WARNING: SVD failed to converge 5 times - skipping the complex"
                                )
                                break
                            print(
                                "| WARNING: SVD failed to converge - trying again with a new sample"
                            )
                        else:
                            raise e

                    if failed_convergence_counter > 5:
                        pass

                if failed_convergence_counter > 5:
                    continue

                if self.original_model_args.no_torsion:
                    orig_center = orig_complex_graph.original_center.cpu().numpy()
                    centered_lig_pos = orig_complex_graph["ligand"].pos.cpu().numpy()
                    orig_complex_graph["ligand"].orig_pos = (
                        centered_lig_pos + orig_center
                    )

                filterHs = (
                    torch.not_equal(predictions_list[0]["ligand"].x[:, 0], 0)
                    .cpu()
                    .numpy()
                )

                if isinstance(orig_complex_graph["ligand"].orig_pos, list):
                    orig_complex_graph["ligand"].orig_pos = orig_complex_graph[
                        "ligand"
                    ].orig_pos[0]

                if isinstance(orig_complex_graph["atom"].orig_holo_pos, list):
                    orig_complex_graph["atom"].orig_holo_pos = orig_complex_graph[
                        "atom"
                    ].orig_holo_pos[0]

                orig_atom_pos = orig_complex_graph["atom"].orig_holo_pos.numpy()
                # print(orig_atom_pos.shape)
                orig_ligand_pos = (
                    orig_complex_graph["ligand"].orig_pos[filterHs]
                    - orig_complex_graph.original_center[0]
                )

                if self.original_model_args.only_nearby_residues_atomic:
                    nearby_atoms = orig_complex_graph["atom"].nearby_atoms.cpu().numpy()
                else:
                    nearby_atoms = np.full(len(orig_complex_graph["atom"].pos), True)

                ligand_pos = []
                atom_pos = []
                calpha_pos = []

                for complex_graph in predictions_list:
                    atom_p = complex_graph["atom"].pos.cpu().numpy()
                    ligand_p = complex_graph["ligand"].pos.cpu().numpy()

                    R, t, _rmsd = rigid_transform_kabsch_numpy(
                        orig_atom_pos[nearby_atoms], atom_p[nearby_atoms]
                    )
                    atom_p = (R @ atom_p.T).T + t
                    ligand_p = (R @ ligand_p.T).T + t

                    ligand_pos.append(ligand_p)
                    atom_pos.append(atom_p)

                    # Ligand RMSD
                    rmsd = np.sqrt(
                        ((ligand_p[filterHs] - orig_ligand_pos) ** 2)
                        .sum(axis=1)
                        .mean(axis=0)
                    )

                    # BB rmsd (measure by calpha)
                    calpha_mask = orig_complex_graph["atom"].calpha.numpy()
                    calpha_pred_atoms = atom_p[calpha_mask]
                    calpha_holo_atoms = orig_atom_pos[calpha_mask]
                    calpha_rmsd = np.sqrt(
                        ((calpha_pred_atoms - calpha_holo_atoms) ** 2)
                        .sum(axis=1)
                        .mean(axis=0)
                    )

                    calpha_pos.append(calpha_pred_atoms)

                    # AA RMSD
                    aa_rmsd = scRMSD(nearby_atoms, atom_p, orig_atom_pos)

                    if self.cache_individual:
                        rmsds_complex.append(rmsd)
                        bb_rmsds_complex.append(calpha_rmsd)
                        aa_rmsds_complex.append(aa_rmsd)

                    rmsds.append(rmsd)
                    bb_rmsds.append(calpha_rmsd)
                    aa_rmsds.append(aa_rmsd)

                successful_docking = (
                    np.array(rmsds).flatten() < self.rmsd_classification_cutoff
                )
                rmsd_lt = (successful_docking.mean() * 100).round(2)
                desc = f"rmsd: {rmsd_lt}%"

                if not isinstance(self.aa_rmsd_classification_cutoff, list):
                    successful_atoms = (
                        np.array(aa_rmsds).flatten()
                        < self.aa_rmsd_classification_cutoff
                    )
                    aa_rmsd_lt = (successful_atoms.mean() * 100).round(2)
                    both_lt = (
                        (successful_docking & successful_atoms).mean() * 100
                    ).round(2)

                    desc += f" | aa_rmsd: {aa_rmsd_lt}% | y: {both_lt}"

                name = orig_complex_graph.name
                if isinstance(name, list):
                    name = name[0]
                names.append(name)

                ligand_pos = np.asarray(ligand_pos)
                atom_pos = np.asarray(atom_pos)
                calpha_pos = np.asarray(calpha_pos)

                # Debug
                # print(f"{name}: ", ligand_pos.shape, atom_pos.shape, calpha_pos.shape)

                # Save each complex individually
                if self.cache_individual:
                    # Keep track of positions
                    full_ligand_positions = ligand_pos
                    full_atom_positions = atom_pos
                    calpha_positions = calpha_pos

                    lig_filename = f"{self.samples_cache_path}/{name}_ligand_positions"
                    _save_positions_rmsds(
                        full_ligand_positions,
                        rmsds_complex,
                        lig_filename,
                        self.cache_creation_id,
                    )

                    aa_filename = f"{self.samples_cache_path}/{name}_atom_positions"
                    _save_positions_rmsds(
                        full_atom_positions,
                        aa_rmsds_complex,
                        aa_filename,
                        self.cache_creation_id,
                    )

                    calpha_filename = (
                        f"{self.samples_cache_path}/{name}_calpha_positions"
                    )
                    _save_positions_rmsds(
                        calpha_positions,
                        bb_rmsds_complex,
                        calpha_filename,
                        cache_creation_id=self.cache_creation_id,
                    )

                else:
                    full_ligand_positions.append(ligand_pos)
                    full_atom_positions.append(atom_pos)
                    calpha_positions.append(calpha_pos)

                pbar.set_description(desc)
                pbar.update()

            names_file = f"{self.samples_cache_path}/complex_names_in_same_order"
            if self.cache_creation_id is not None:
                names_file += f"_id{self.cache_creation_id}"
            names_file += ".pkl"

            with open(names_file, "wb") as f:
                pickle.dump((names), f)

            # Save all positions together
            if not self.cache_individual:
                lig_filename = f"{self.samples_cache_path}/ligand_positions"
                _save_positions_rmsds(
                    full_ligand_positions, rmsds, lig_filename, self.cache_creation_id
                )

                atom_filename = f"{self.samples_cache_path}/atom_positions"
                _save_positions_rmsds(
                    full_atom_positions, aa_rmsds, atom_filename, self.cache_creation_id
                )

                calpha_filename = f"{self.samples_cache_path}/calpha_positions"
                _save_positions_rmsds(
                    calpha_positions, bb_rmsds, calpha_filename, self.cache_creation_id
                )

    def check_preprocessing_needed(self, cache_path):
        if not self.cache_individual:
            if not os.path.exists(os.path.join(cache_path, "heterographs.pkl")):
                return False

        else:
            if not os.path.exists(cache_path):
                print("Directory not found...")
                return False

            else:
                complex_names_all = read_strings_from_txt(self.split_path)
                if self.limit_complexes is not None and self.limit_complexes != 0:
                    complex_names_all = complex_names_all[: self.limit_complexes]

                if hasattr(self, "use_esmflow_update") and self.use_esmflow_update:
                    complexes_available = [
                        filename.split(".")[0].split("-")[-1].split("_")[0]
                        for filename in os.listdir(cache_path)
                        if "heterograph" in filename
                    ]

                else:
                    # if os.path.exists(self.samples_cache_path):
                    complexes_available = [
                        filename.split(".")[0].split("-")[1]
                        for filename in os.listdir(cache_path)
                        if "heterograph" in filename
                    ]
                    # else:
                    #    complexes_available = []

                if not len(complexes_available):
                    print("Directory found but no complexes available.", flush=True)
                    return False

                if not len(
                    set(complex_names_all).intersection(set(complexes_available))
                ):
                    print("No common complexes.", flush=True)
                    return False

        return True

    def maybe_combine_cache_ids(self):
        if self.cache_ids_to_combine is not None:
            if not self.cache_individual:
                all_rmsds_unsorted, all_aa_rmsds_unsorted, all_bb_rmsds_unsorted = (
                    [],
                    [],
                    [],
                )
                (
                    all_full_ligand_positions_unsorted,
                    all_full_atom_positions_unsorted,
                    all_calpha_positions_unsorted,
                ) = ([], [], [])
                all_names_unsorted = []

                for idx, cache_id in enumerate(self.cache_ids_to_combine):
                    if not os.path.exists(
                        os.path.join(
                            self.samples_cache_path,
                            f"ligand_positions_id{cache_id}.pkl",
                        )
                    ):
                        raise Exception(
                            f"The generated ligand positions with cache_id do not exist: {cache_id}"
                        )

                    with open(
                        os.path.join(
                            self.samples_cache_path,
                            f"ligand_positions_id{cache_id}.pkl",
                        ),
                        "rb",
                    ) as f:
                        full_ligand_positions, rmsds = pickle.load(f)

                    with open(
                        os.path.join(
                            self.samples_cache_path, f"atom_positions_id{cache_id}.pkl"
                        ),
                        "rb",
                    ) as f:
                        full_atom_positions, aa_rmsds = pickle.load(f)

                    with open(
                        os.path.join(
                            self.samples_cache_path,
                            f"calpha_positions_id{cache_id}.pkl",
                        ),
                        "rb",
                    ) as f:
                        calpha_positions, bb_rmsds = pickle.load(f)

                    with open(
                        os.path.join(
                            self.samples_cache_path,
                            f"complex_names_in_same_order_id{cache_id}.pkl",
                        ),
                        "rb",
                    ) as f:
                        names_unsorted = pickle.load(f)

                    all_names_unsorted.append(names_unsorted)
                    all_rmsds_unsorted.append(rmsds)
                    all_aa_rmsds_unsorted.append(aa_rmsds)
                    all_bb_rmsds_unsorted.append(bb_rmsds)

                    all_full_ligand_positions_unsorted.append(full_ligand_positions)
                    all_full_atom_positions_unsorted.append(full_atom_positions)
                    all_calpha_positions_unsorted.append(calpha_positions)

                names_order = list(set.intersection(*map(set, all_names_unsorted)))
                all_rmsds, all_aa_rmsds, all_bb_rmsds = [], [], []
                (
                    all_full_ligand_positions,
                    all_full_atom_positions,
                    all_calpha_positions,
                ) = ([], [], [])

                elem_tuple = (
                    all_rmsds_unsorted,
                    all_aa_rmsds_unsorted,
                    all_bb_rmsds_unsorted,
                )
                elem_tuple += (
                    all_full_ligand_positions_unsorted,
                    all_full_atom_positions_unsorted,
                    all_calpha_positions_unsorted,
                )
                elem_tuple += (all_names_unsorted,)

                for idx, elems in enumerate(zip(elem_tuple)):
                    (
                        rmsds_unsorted,
                        aa_rmsds_unsorted,
                        bb_rmsds_unsorted,
                        full_ligand_positions_unsorted,
                        full_atom_positions_unsorted,
                        calpha_positions_unsorted,
                        names_unsorted,
                    ) = elems

                    name_to_elem_dict = {
                        name: (pos, rmsd, atom_pos, aa_rmsd, calpha_pos, bb_rmsd)
                        for name, pos, rmsd, atom_pos, aa_rmsd, calpha_pos, bb_rmsd in zip(
                            names_unsorted,
                            full_ligand_positions_unsorted,
                            rmsds_unsorted,
                            full_atom_positions_unsorted,
                            aa_rmsds_unsorted,
                            calpha_positions_unsorted,
                            bb_rmsds_unsorted,
                        )
                    }

                    all_full_ligand_positions.append(
                        [name_to_elem_dict[name][0] for name in names_order]
                    )
                    all_rmsds.append(
                        [name_to_elem_dict[name][1] for name in names_order]
                    )
                    all_full_atom_positions.append(
                        [name_to_elem_dict[name][2] for name in names_order]
                    )
                    all_aa_rmsds.append(
                        [name_to_elem_dict[name][3] for name in names_order]
                    )
                    all_calpha_positions.append(
                        [name_to_elem_dict[name][4] for name in names_order]
                    )
                    all_bb_rmsds.append(
                        [name_to_elem_dict[name][5] for name in names_order]
                    )

                self.full_ligand_positions, self.rmsds = [], []
                self.full_atom_positions, self.aa_rmsds = [], []
                self.calpha_positions, self.bb_rmsds = [], []

                # Ligand
                for positions_tuple in list(zip(*all_full_ligand_positions)):
                    self.full_ligand_positions.append(
                        np.concatenate(positions_tuple, axis=0)
                    )
                for rmsd_tuple in list(zip(*all_rmsds)):
                    self.rmsds.append(np.concatenate(rmsd_tuple, axis=0))

                # All atoms
                for atom_pos_tuple in list(zip(*all_full_atom_positions)):
                    self.full_atom_positions.append(
                        np.concatenate(atom_pos_tuple, axis=0)
                    )
                for aa_rmsd_tuple in list(zip(*all_aa_rmsds)):
                    self.aa_rmsds.append(np.concatenate(aa_rmsd_tuple, axis=0))

                # calpha
                for calpha_pos_tuple in list(zip(*all_calpha_positions)):
                    self.calpha_positions.append(
                        np.concatenate(calpha_pos_tuple, axis=0)
                    )
                for bb_rmsd_tuple in list(zip(*all_bb_rmsds)):
                    self.bb_rmsds.append(np.concatenate(bb_rmsd_tuple, axis=0))

                generated_rmsd_complex_names = names_order
                assert len(self.rmsds) == len(generated_rmsd_complex_names)
                # if self.original_model_args.flexible_sidechains:
                assert len(self.sc_rmsds) == len(generated_rmsd_complex_names)

                self.lig_positions_rmsds_dict = {
                    name: (pos, rmsd)
                    for name, pos, rmsd in zip(
                        generated_rmsd_complex_names,
                        self.full_ligand_positions,
                        self.rmsds,
                    )
                }

                self.atom_positons_rmsds_dict = {
                    name: (pos, rmsd)
                    for name, pos, rmsd in zip(
                        generated_rmsd_complex_names,
                        self.full_atom_positions,
                        self.aa_rmsds,
                    )
                }

                self.calpha_positons_rmsds_dict = {
                    name: (pos, rmsd)
                    for name, pos, rmsd in zip(
                        generated_rmsd_complex_names,
                        self.calpha_positions,
                        self.bb_rmsds,
                    )
                }

                self.dataset_names = list(
                    set(self.lig_positions_rmsds_dict.keys())
                    & set(self.atom_positons_rmsds_dict.keys())
                    & set(self.calpha_positons_rmsds_dict.keys())
                    & set(self.complex_graph_dict.keys())
                )

            else:
                names_to_cache_id = defaultdict(list)
                for idx, cache_id in enumerate(self.cache_ids_to_combine):
                    with open(
                        f"{self.samples_cache_path}/complex_names_in_same_order_id{cache_id}.pkl",
                        "rb",
                    ) as f:
                        names_from_current_id = pickle.load(f)

                    for name in names_from_current_id:
                        names_to_cache_id[name].append(cache_id)

                print("Loaded names. Now combining positions", flush=True)

                for name, cache_ids in names_to_cache_id.items():
                    ligand_positions, rmsds = [], []
                    atom_positions, aa_rmsds = [], []
                    calpha_positions, bb_rmsds = [], []

                    lig_positions_file = (
                        f"{self.samples_cache_path}/{name}_ligand_positions.pkl"
                    )
                    atom_positions_file = (
                        f"{self.samples_cache_path}/{name}_atom_positions.pkl"
                    )
                    calpha_positions_file = (
                        f"{self.samples_cache_path}/{name}_calpha_positions.pkl"
                    )

                    if (
                        os.path.exists(lig_positions_file)
                        and os.path.exists(atom_positions_file)
                        and os.path.exists(calpha_positions_file)
                    ):
                        continue

                    else:
                        for cache_id in cache_ids:
                            with open(
                                f"{self.samples_cache_path}/{name}_ligand_positions_id{cache_id}.pkl",
                                "rb",
                            ) as f:
                                lig_positions_cache, rmsds_cache = pickle.load(f)
                            ligand_positions.append(lig_positions_cache)
                            rmsds.append(rmsds_cache)

                            with open(
                                f"{self.samples_cache_path}/{name}_atom_positions_id{cache_id}.pkl",
                                "rb",
                            ) as f:
                                atom_positions_cache, aa_rmsds_cache = pickle.load(f)
                            atom_positions.append(atom_positions_cache)
                            aa_rmsds.append(aa_rmsds_cache)

                            with open(
                                f"{self.samples_cache_path}/{name}_calpha_positions_id{cache_id}.pkl",
                                "rb",
                            ) as f:
                                calpha_positions_cache, bb_rmsds_cache = pickle.load(f)
                            calpha_positions.append(calpha_positions_cache)
                            bb_rmsds.append(bb_rmsds_cache)

                        ligand_positions = np.concatenate(ligand_positions, axis=0)
                        rmsds = np.concatenate(rmsds, axis=0)
                        atom_positions = np.concatenate(atom_positions, axis=0)
                        aa_rmsds = np.concatenate(aa_rmsds, axis=0)
                        calpha_positions = np.concatenate(calpha_positions, axis=0)
                        bb_rmsds = np.concatenate(bb_rmsds, axis=0)

                        _save_positions_rmsds(
                            ligand_positions,
                            rmsds,
                            f"{self.samples_cache_path}/{name}_ligand_positions",
                        )
                        _save_positions_rmsds(
                            atom_positions,
                            aa_rmsds,
                            f"{self.samples_cache_path}/{name}_atom_positions",
                        )
                        _save_positions_rmsds(
                            calpha_positions,
                            bb_rmsds,
                            f"{self.samples_cache_path}/{name}_calpha_positions",
                        )

                self.dataset_names = list(names_to_cache_id.keys())
                print("Cache combining complete...")

        else:
            if not self.cache_individual:
                lig_filename = f"{self.samples_cache_path}/ligand_positions"
                if self.cache_creation_id is not None:
                    lig_filename += f"_id{self.cache_creation_id}"
                lig_filename += ".pkl"

                with open(lig_filename, "rb") as f:
                    self.full_ligand_positions, self.rmsds = pickle.load(f)

                atom_filename = f"{self.samples_cache_path}/atom_positions"
                if self.cache_creation_id is not None:
                    atom_filename += f"_id{self.cache_creation_id}"
                atom_filename += ".pkl"

                with open(atom_filename, "rb") as f:
                    self.atom_positions, self.aa_rmsds = pickle.load(f)

                calpha_filename = f"{self.samples_cache_path}/calpha_positions"
                if self.cache_creation_id is not None:
                    calpha_filename += f"_id{self.cache_creation_id}"
                calpha_filename += ".pkl"

                with open(calpha_filename, "rb") as f:
                    self.calpha_positions, self.bb_rmsds = pickle.load(f)

            names_file = f"{self.samples_cache_path}/complex_names_in_same_order"
            if self.cache_creation_id is not None:
                names_file += f"_id{self.cache_creation_id}"
            names_file += ".pkl"

            with open(names_file, "rb") as f:
                names = pickle.load(f)
                self.dataset_names = names

    def len(self):
        return len(self.available_complexes) * self.multiplicity


class FilteringDataset(PostDockingDataset):
    def __init__(
        self,
        balance: bool = False,
        use_original_model_cache: bool = True,
        rmsd_classification_cutoff: float = 2,
        aa_rmsd_classification_cutoff: float = 1,
        bb_rmsd_classification_cutoff: float = 1,
        only_rmsd_labels: bool = False,
        atom_lig_confidence: bool = False,
        **kwargs,
    ):
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.aa_rmsd_classification_cutoff = aa_rmsd_classification_cutoff
        self.bb_rmsd_classification_cutoff = bb_rmsd_classification_cutoff
        super().__init__(**kwargs)
        self.balance = balance

        self.only_rmsd_labels = only_rmsd_labels
        self.atom_lig_confidence = atom_lig_confidence

        print(f"Available complexes: {self.available_complexes}")

    def set_graph_cache_path(self, split_path):
        self.original_model_args, original_model_cache = get_args_and_cache_path(
            self.original_model_dir, split_path
        )

        filtering_cache_kwargs = gather_cache_path_kwargs(
            self.filtering_args, split_path=split_path
        )
        filtering_cache_path = get_full_cache_path(**filtering_cache_kwargs)

        self.complex_graphs_cache = (
            original_model_cache
            if self.use_original_model_cache
            else filtering_cache_path
        )
        self.complex_graphs_cache_base = (
            self.original_model_args.cache_path
            if self.use_original_model_cache
            else self.filtering_args.cache_path
        )

    def set_samples_cache_path(self, sample_cache_path, split_path):
        model_name_base = os.path.splitext(self.filtering_args.model_ckpt)[0]
        self.samples_cache_path = os.path.join(
            sample_cache_path,
            f"model_{self.original_model_args.run_name}_{model_name_base}"
            f"_split_{os.path.splitext(os.path.basename(split_path))[0]}_limit_{self.limit_complexes}",
        )
        print(self.samples_cache_path)

    def gather_complex_names(self, split_path, pipeline=None):
        args_to_use = (
            self.original_model_args
            if self.use_original_model_cache
            else self.filtering_args
        )
        dataset = get_docking_dataset_from_args(
            args=args_to_use,
            dataset_name=self.filtering_args.dataset,
            split_path=split_path,
            data_dir=self.filtering_args.data_dir,
            multiplicity=self.multiplicity,
            limit_complexes=self.limit_complexes,
            num_workers=self.filtering_args.num_workers,
            pipeline=pipeline,
        )

        if self.cache_individual:
            split_fn = lambda x: x.split(".")[0].split("-")[1]

            complex_names = [split_fn(filename) for filename in dataset.complex_files]
        else:
            complex_names = [graph.name for graph in dataset.complex_graphs]
        return complex_names

    def get(self, idx):
        if self.multiplicity > 1:
            idx = idx % len(self.available_complexes)

        name = self.available_complexes[idx]

        if self.cache_individual:
            complex_graph = torch.load(
                f"{self.complex_graphs_cache}/heterograph-{name}-0.pt",
                map_location="cpu",
            )
            with open(
                f"{self.samples_cache_path}/{name}_ligand_positions.pkl", "rb"
            ) as f:
                ligand_positions, rmsds = pickle.load(f)

            with open(
                f"{self.samples_cache_path}/{name}_atom_positions.pkl", "rb"
            ) as f:
                atom_positions, aa_rmsds = pickle.load(f)

            with open(
                f"{self.samples_cache_path}/{name}_calpha_positions.pkl", "rb"
            ) as f:
                calpha_positions, bb_rmsds = pickle.load(f)

        else:
            complex_graph = self.complex_graph_dict[name]
            ligand_positions, rmsds = self.lig_positions_rmsds_dict[name]
            atom_positions, aa_rmsds = self.atom_positions_rmsds_dict[name]
            calpha_positions, bb_rmsds = self.calpha_positions_rmsds_dict[name]

        t = 0
        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list):
                raise ValueError(
                    "a list for --rmsd_classification_cutoff can only be used without --balance"
                )
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
            n_success = np.count_nonzero(success)
            if (label == 0 and n_success != self.all_samples_per_complex) or (
                n_success == 0 and self.trajectory_sampling
            ):
                # sample negative complex
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = ligand_positions[~success][sample]
                complex_graph["ligand"].pos = torch.from_numpy(lig_pos).float()
            else:
                # sample positive complex
                if (
                    n_success > 0
                ):  # if no successfull sample returns the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = ligand_positions[success][sample]
                    complex_graph["ligand"].pos = torch.from_numpy(lig_pos).float()
            complex_graph.y = torch.tensor(label).float()

            if self.original_model_args.flexible_sidechains:
                raise NotImplementedError(
                    "parallel not implemented for flexible sidechains"
                )
        else:
            n_samples = min(
                ligand_positions.shape[0],
                atom_positions.shape[0],
                calpha_positions.shape[0],
            )
            sample = (
                random.randint(0, n_samples - 1)
                if self.fixed_sample is None
                else self.fixed_sample
            )

            if self.filtering_args.only_nearby_residues_atomic:
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].orig_holo_pos,
                    complex_graph["receptor"].lens_receptors,
                )
                ligand_atoms = (
                    torch.from_numpy(complex_graph["ligand"].orig_pos).float()
                    - complex_graph.original_center
                )
                minimum_distance = (
                    torch.cdist(atom_grid, ligand_atoms)
                    .min(dim=1)
                    .values.min(dim=1)
                    .values
                )
                nearby_residues = (
                    minimum_distance < self.filtering_args.nearby_residues_atomic_radius
                )

                # if there are less than nearby_residues_atomic_min residues nearby, we take the nearby_residues_atomic_min closest residues
                if (
                    torch.sum(nearby_residues)
                    < self.filtering_args.nearby_residues_atomic_min
                ):
                    # print(f'Found only {nearby_residues.sum()} nearby residues for {complex_graph.name}')
                    _, closest_residues = torch.topk(
                        minimum_distance,
                        k=self.filtering_args.nearby_residues_atomic_min,
                        largest=False,
                    )
                    nearby_residues = torch.zeros_like(nearby_residues)
                    nearby_residues[closest_residues] = True
                    # print(f'Found {nearby_residues.sum()} nearby residues for {complex_graph.name} from total of {len(nearby_residues)} residues')

                complex_graph["receptor"].nearby_residues = nearby_residues
                nearby_atoms = torch.zeros(
                    (atom_grid.shape[0], atom_grid.shape[1]), dtype=torch.bool
                )
                nearby_atoms[nearby_residues] = True
                nearby_atoms = nearby_atoms[x, y]
                complex_graph["atom"].nearby_atoms = nearby_atoms

                # test_filter_flexible_residues(complex_graph)  # unit test

                filter_flexible_residues(
                    complex_graph, complex_graph["atom"].nearby_atoms
                )

            complex_graph["ligand"].pos = torch.from_numpy(
                ligand_positions[sample]
            ).float()
            complex_graph.y = (
                torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff)
                .float()
                .unsqueeze(0)
            )
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(
                    np.logical_and(
                        rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],
                        rmsds[sample] >= [0] + self.rmsd_classification_cutoff,
                    ),
                    dtype=torch.float,
                ).unsqueeze(0)
                complex_graph.y = (
                    torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0])
                    .unsqueeze(0)
                    .float()
                )
            if self.atom_lig_confidence:
                complex_graph.y_aa = (
                    torch.tensor(aa_rmsds[sample] < self.aa_rmsd_classification_cutoff)
                    .float()
                    .unsqueeze(0)
                )
            complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

            complex_graph["atom"].pos = torch.from_numpy(atom_positions[sample]).float()
            complex_graph["receptor"].pos = torch.from_numpy(
                calpha_positions[sample]
            ).float()

            if not self.only_rmsd_labels:
                # TODO: Whether to use this
                try:
                    complex_graph.y *= (
                        torch.tensor(
                            aa_rmsds[sample] < self.aa_rmsd_classification_cutoff
                        )
                        .float()
                        .unsqueeze(0)
                    )
                #     complex_graph.y *= torch.tensor(bb_rmsds[sample] < self.bb_rmsd_classification_cutoff).float().unsqueeze(0)
                except Exception as e:
                    print(e)
                    print(complex_graph.name)
                #     print(complex_graph.y)
                #     print(aa_rmsds[sample].shape)
                #     print(aa_rmsds[sample] < self.aa_rmsd_classification_cutoff)
                #     print(bb_rmsds[sample].shape)
                #     print(bb_rmsds[sample] < self.bb_rmsd_classification_cutoff)
                #     raise e

            complex_graph.aa_rmsd = torch.tensor(aa_rmsds[sample]).unsqueeze(0).float()
            complex_graph.bb_rmsd = torch.tensor(bb_rmsds[sample]).unsqueeze(0).float()

        if self.original_model_args.sidechain_tor_bridge:
            t_sidechain_tor = 1
        else:
            t_sidechain_tor = 0

        t_bb_tr, t_bb_rot = 1, 1

        set_time(
            complex_graph,
            t=t,
            t_tr=t,
            t_rot=t,
            t_tor=t,
            t_sidechain_tor=t_sidechain_tor,
            t_bb_rot=t_bb_rot,
            t_bb_tr=t_bb_tr,
            batchsize=1,
            all_atoms=self.original_model_args.all_atoms,
            asyncronous_noise_schedule=self.asyncronous_noise_schedule,
            device="cpu",
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
        )

        return complex_graph


class RelaxationDataset(PostDockingDataset):
    def __init__(self, filter_clashes: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_clashes = filter_clashes

        available_complexes = copy.deepcopy(self.available_complexes)

        if filter_clashes:
            to_remove = []
            with tqdm(
                total=len(available_complexes), desc="Filtering complexes"
            ) as pbar:
                with Pool(self.filtering_args.num_workers) as p:
                    for idx, invalid in p.imap_unordered(
                        self.filter_complex, range(len(available_complexes))
                    ):
                        if invalid:
                            to_remove.append(idx)
                        pbar.update(1)
            to_remove.sort(reverse=True)
            for idx in to_remove:
                self.available_complexes.pop(idx)

            print("Number of complexes removed due to steric clash: ", len(to_remove))

        print("Number of available complexes: ", len(self.available_complexes))

    def set_graph_cache_path(self, split_path):
        filtering_args = self.filtering_args
        filtering_cache_kwargs = gather_cache_path_kwargs(
            filtering_args, split_path=split_path
        )
        self.complex_graphs_cache = get_full_cache_path(**filtering_cache_kwargs)
        self.complex_graphs_cache_base = filtering_args.cache_path

    def set_samples_cache_path(self, sample_cache_path, split_path):
        self.samples_cache_path = os.path.join(
            sample_cache_path,
            f"model_pl-newpocket-esm-buf=20-geoflow-flexbb=0,0.1scbridge_best_inference_epoch_model"
            f"_split_{os.path.splitext(os.path.basename(split_path))[0]}_limit_{self.limit_complexes}",
        )

    def gather_complex_names(self, split_path, pipeline=None):
        args_to_use = filtering_args = self.filtering_args
        dataset = get_docking_dataset_from_args(
            args=args_to_use,
            dataset_name=filtering_args.dataset,
            split_path=split_path,
            data_dir=filtering_args.data_dir,
            multiplicity=self.multiplicity,
            limit_complexes=self.limit_complexes,
            num_workers=filtering_args.num_workers,
            pipeline=pipeline,
        )

        if self.cache_individual:
            if (
                hasattr(filtering_args, "use_esmflow_update")
                and filtering_args.use_esmflow_update
            ):
                split_fn = lambda x: x.split(".")[0].split("-")[-1].split("_")[0]
            else:
                split_fn = lambda x: x.split(".")[0].split("-")[1]

            complex_names = [split_fn(filename) for filename in dataset.complex_files]
            # pass
        else:
            complex_names = [graph.name for graph in dataset.complex_graphs]

        return complex_names

    def filter_complex(self, idx):
        complex_graph = self.get(idx)
        cross_dists = torch.linalg.norm(
            complex_graph["ligand"].orig_pos[:, None, :]
            - complex_graph["atom"].orig_holo_pos[None, :, :],
            dim=-1,
        )
        clashes = torch.clip(
            complex_graph["ligand"].vdw_radii[:, None]
            + complex_graph["atom"].vdw_radii[None, :]
            - cross_dists
            - self.filtering_args.overlap_loss_buffer,
            min=0,
        )
        lig_dists = torch.linalg.norm(
            complex_graph["ligand"].orig_pos[
                complex_graph["ligand", "lig_edge", "ligand"].posebusters_edge_index[1]
            ]
            - complex_graph["ligand"].orig_pos[
                complex_graph["ligand", "lig_edge", "ligand"].posebusters_edge_index[0]
            ],
            dim=-1,
        )
        covaelent_shrinks = torch.clip(
            complex_graph["ligand", "lig_edge", "ligand"].lower_bound
            * (1.0 - self.filtering_args.bond_loss_buffer)
            - lig_dists,
            min=0,
        )[
            complex_graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask
            + complex_graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask
        ]
        covaelent_stretches = torch.clip(
            lig_dists
            - complex_graph["ligand", "lig_edge", "ligand"].upper_bound
            * (1.0 + self.filtering_args.bond_loss_buffer),
            min=0,
        )[
            complex_graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask
            + complex_graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask
        ]
        sterics = torch.clip(
            complex_graph["ligand", "lig_edge", "ligand"].lower_bound
            * (1.0 - self.filtering_args.steric_loss_buffer)
            - lig_dists,
            min=0,
        )[
            ~(
                complex_graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask
                + complex_graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask
            )
        ]
        return idx, torch.any(clashes > 0) or torch.any(
            covaelent_shrinks > 0
        ) or torch.any(covaelent_stretches > 0) or torch.any(sterics > 0)

    def load_graphs(self):
        # print('Starting dataset get')
        complex_graphs = []
        for name in self.available_complexes:
            if self.cache_individual:
                # print('Loading complex_graph')
                complex_graph = torch.load(
                    f"{self.complex_graphs_cache}/heterograph-{name}-0.pt",
                    map_location="cpu",
                )
                # print('Loading rdkit mol')
                with open(
                    os.path.join(
                        self.complex_graphs_cache, f"rdkit_ligand-{name}-0.pkl"
                    ),
                    "rb",
                ) as file:
                    complex_graph.mol = pickle.load(file)
                # print('Loading ligand_positions')
                with open(
                    f"{self.samples_cache_path}/{name}_ligand_positions.pkl", "rb"
                ) as f:
                    ligand_positions, rmsds = pickle.load(f)
                # print('Loading atom_positions')
                with open(
                    f"{self.samples_cache_path}/{name}_atom_positions.pkl", "rb"
                ) as f:
                    atom_positions, aa_rmsds = pickle.load(f)
                # print('Loading calpha_positions')
                with open(
                    f"{self.samples_cache_path}/{name}_calpha_positions.pkl", "rb"
                ) as f:
                    calpha_positions, bb_rmsds = pickle.load(f)
            else:
                complex_graph = self.complex_graph_dict[name]
                ligand_positions, _ = self.lig_positions_rmsds_dict[name]
                atom_positions, _ = self.atom_positions_rmsds_dict[name]

            complex_graph["ligand"].flexdock_pos = torch.from_numpy(
                ligand_positions
            ).float()
            complex_graph["atom"].flexdock_pos = torch.from_numpy(
                atom_positions
            ).float()
            complex_graph["receptor"].flexdock_pos = torch.from_numpy(
                calpha_positions
            ).float()
            complex_graph["atom"].nearby_atom_mask = get_nearby_atom_mask(
                complex_graph["atom"].orig_holo_pos,
                complex_graph["ligand"].orig_pos,
                complex_graph["atom", "receptor"].edge_index[1],
                cutoff=5.0,
            )
            holo_center = (
                complex_graph["atom"]
                .orig_holo_pos[complex_graph["atom"].nearby_atom_mask]
                .mean(axis=0)
            )
            complex_graph["atom"].orig_holo_pos -= holo_center
            complex_graph["ligand"].orig_pos -= holo_center
            complex_graph["receptor"].holo_pos = complex_graph["atom"].orig_holo_pos[
                complex_graph["atom"].ca_mask
            ]
            complex_graphs.append(complex_graph)
        self.complex_graphs = complex_graphs

    def get(self, idx):
        if self.multiplicity > 1:
            idx = idx % len(self.available_complexes)
        name = self.available_complexes[idx]
        if self.cache_individual:
            complex_graph = torch.load(
                f"{self.complex_graphs_cache}/heterograph-{name}-0.pt",
                map_location="cpu",
            )
            with open(
                os.path.join(self.complex_graphs_cache, f"rdkit_ligand-{name}-0.pkl"),
                "rb",
            ) as file:
                complex_graph.mol = pickle.load(file)
            with open(
                f"{self.samples_cache_path}/{name}_ligand_positions.pkl", "rb"
            ) as f:
                ligand_positions, rmsds = pickle.load(f)
            with open(
                f"{self.samples_cache_path}/{name}_atom_positions.pkl", "rb"
            ) as f:
                atom_positions, aa_rmsds = pickle.load(f)
            with open(
                f"{self.samples_cache_path}/{name}_calpha_positions.pkl", "rb"
            ) as f:
                calpha_positions, bb_rmsds = pickle.load(f)
        else:
            raise ValueError("Not suppourted")

        complex_graph["ligand"].flexdock_pos = torch.from_numpy(
            ligand_positions
        ).float()
        complex_graph["atom"].flexdock_pos = torch.from_numpy(atom_positions).float()
        complex_graph["receptor"].flexdock_pos = torch.from_numpy(
            calpha_positions
        ).float()

        return complex_graph
