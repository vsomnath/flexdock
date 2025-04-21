import os
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.loader.dataloader import Collater
from torch_geometric.loader import DataListLoader

from flexdock.data.dataset.filter_or_relax import RelaxationDataset
from flexdock.data.transforms.relaxation import construct_transform


class IterableConfDataset(IterableDataset):
    def __init__(self, graph, multiplicity=1):
        self.graph = graph
        self.multiplicity = multiplicity

    def generate(self):
        for conf_idx in range(self.graph.n_samples):
            conf_graph = self.graph.clone("pos").apply(
                lambda feature: feature[conf_idx], "pos"
            )
            for _ in range(self.multiplicity):
                yield conf_graph.clone("pos")

    def __iter__(self):
        return iter(self.generate())


class SafeDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch=None,
        exclude_keys=None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        default_collate_fn = Collater(dataset, follow_batch, exclude_keys)

        def safe_collate_fn(data_list):
            if data_list is None:
                return None
            else:
                return default_collate_fn(
                    [data for data in data_list if data is not None]
                )

        # safe_collate_fn = (
        #     lambda data_list: None
        #     if all(data is None for data in data_list)
        #     else default_collate_fn()
        # )
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=safe_collate_fn,
            **kwargs,
        )


class RelaxationDataModule(LightningDataModule):
    def __init__(self, args, device: str = "cpu"):
        dataset_kwargs = {
            "sample_cache_path": args.sample_cache_path,
            "filtering_args": args,
            "original_model_dir": args.original_model_dir,
            "inference_steps": args.inference_steps,
            "samples_per_complex": args.samples_per_complex,
            "limit_complexes": args.limit_complexes,
            "all_atoms": args.all_atoms,
            "inf_sched_alpha": args.inf_sched_alpha,
            "inf_sched_beta": args.inf_sched_beta,
            "sigma_schedule": args.sigma_schedule,
            "use_original_model_cache": args.use_original_model_cache,
            "cache_ids_to_combine": args.cache_ids_to_combine,
            "cache_creation_id": args.cache_creation_id,
            "include_miscellaneous_atoms": args.include_miscellaneous_atoms,
            "asyncronous_noise_schedule": args.asyncronous_noise_schedule,
            "cache_individual": args.cache_individual,
            "device": device,
        }

        transform = construct_transform(
            n_conformers=args.n_conformers,
            tr_sigma=args.tr_sigma,
            rot_sigma=args.rot_sigma,
            lig_bond_sigma=args.lig_bond_sigma,
            lig_angle_sigma=args.lig_angle_sigma,
            lig_torsion_sigma=args.lig_torsion_sigma,
            lig_fragment_sigma=args.lig_fragment_sigma,
            bb_tr_sigma=args.bb_tr_sigma,
            bb_rot_sigma=args.bb_rot_sigma,
            sidechain_bond_sigma=args.sidechain_bond_sigma,
            sidechain_angle_sigma=args.sidechain_angle_sigma,
            sidechain_torsion_sigma=args.sidechain_torsion_sigma,
            sidechain_fragment_sigma=args.sidechain_fragment_sigma,
            lig_sigma=args.ligand_sigma,
            atom_sigma=args.atom_sigma,
            nearby_atom_cutoff=args.nearby_residues_atomic_radius,
            rmsd_cutoff=args.rmsd_cutoff,
            sampling_kappa=args.sampling_kappa,
            sampling_epsilon=args.sampling_epsilon,
            sampling_alpha=args.sampling_alpha,
            sampling_beta=args.sampling_beta,
        )

        self._train_dataset = RelaxationDataset(
            **dataset_kwargs,
            split_path=os.path.abspath(args.split_train),
            return_conformers=True,
            randomize_conformers=True,
            transform=transform,
            multiplicity=args.multiplicity,
            filter_clashes=True,
        )

        self._val_dataset = RelaxationDataset(
            **dataset_kwargs,
            split_path=os.path.abspath(args.split_val),
            return_conformers=True,
            randomize_conformers=False,
            transform=transform,
            multiplicity=args.multiplicity,
            filter_clashes=False,
        )

        self._inf_dataset = RelaxationDataset(
            **dataset_kwargs,
            split_path=os.path.abspath(args.split_val),
            return_conformers=False,
            multiplicity=1,
            filter_clashes=False,
        )

    def train_dataloader(self):
        return SafeDataLoader(
            dataset=self._train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_dataloader_workers,
            exclude_keys=[
                "angle_2_index",
                "rotate_mask",
                "squeeze_mask",
                "fragment_index",
                "flexdock_pos",
            ],
        )

    def val_dataloader(self):
        val_loaders = [
            SafeDataLoader(
                dataset=self._val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_dataloader_workers,
                exclude_keys=[
                    "angle_2_index",
                    "rotate_mask",
                    "squeeze_mask",
                    "fragment_index",
                    "flexdock_pos",
                ],
            )
        ]

        val_loaders.append(
            DataListLoader(dataset=self._inf_dataset, batch_size=1, shuffle=False)
        )

        loader = CombinedLoader(val_loaders, mode="sequential")
        return loader
