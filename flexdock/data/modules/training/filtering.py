import os

from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader

from flexdock.data.dataset.filter_or_relax import FilteringDataset


class FilteringDataModule(LightningDataModule):
    def __init__(self, args, device: str = "cpu"):
        super().__init__()

        self.args = args

        dataset_kwargs = {
            "sample_cache_path": args.sample_cache_path,
            "filtering_args": args,
            "original_model_dir": args.original_model_dir,
            "inference_steps": args.inference_steps,
            "samples_per_complex": args.samples_per_complex,
            "limit_complexes": args.limit_complexes,
            "multiplicity": args.multiplicity,
            "all_atoms": args.all_atoms,
            "inf_sched_alpha": args.inf_sched_alpha,
            "inf_sched_beta": args.inf_sched_beta,
            "sigma_schedule": args.sigma_schedule,
            "balance": args.balance,
            "use_original_model_cache": args.use_original_model_cache,
            "rmsd_classification_cutoff": args.rmsd_classification_cutoff,
            "aa_rmsd_classification_cutoff": args.aa_rmsd_classification_cutoff,
            "bb_rmsd_classification_cutoff": args.bb_rmsd_classification_cutoff,
            "cache_ids_to_combine": args.cache_ids_to_combine,
            "cache_creation_id": args.cache_creation_id,
            "include_miscellaneous_atoms": args.include_miscellaneous_atoms,
            "asyncronous_noise_schedule": args.asyncronous_noise_schedule,
            "cache_individual": args.cache_individual,
            "only_rmsd_labels": getattr(args, "only_rmsd_labels", False),
            "atom_lig_confidence": getattr(args, "atom_lig_confidence", False),
            "device": device,
        }

        self._train_dataset = FilteringDataset(
            **dataset_kwargs, split_path=os.path.abspath(args.split_train)
        )

        self._val_set = FilteringDataset(
            **dataset_kwargs, split_path=os.path.abspath(args.split_val)
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
