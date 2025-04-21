import os
import random
import csv

from typing import Optional, Any
from dataclasses import dataclass, fields

import torch
from torch_geometric.data import Dataset

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from torch_geometric.loader import DataLoader

from flexdock.data.constants import AVAILABLE_DATASETS
from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.transforms.docking import construct_transform


class ListDataset(Dataset):
    def __init__(self, list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]


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


class DockingDataset(Dataset):
    def __init__(
        self,
        transform: callable,
        split_path: str,
        dataset: str = "pdbbind",
        cache_path: str = "data/cache",  # Full cache path
        cluster_file: Optional[str] = None,
        complexes_per_cluster: int = 1,
        limit_complexes: int = 0,
        multiplicity: int = 1,
        num_workers: int = 1,
        require_ligand: bool = False,
    ):
        super().__init__(transform=transform)
        self.dataset_name = dataset
        self.split_path = split_path
        self.cache_path = cache_path
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity
        self.num_workers = num_workers
        self.cluster_file = cluster_file
        self.complexes_per_cluster = complexes_per_cluster
        self.require_ligand = require_ligand

        if not self.check_processed_inputs():
            raise ValueError("Inputs must be processed before running training.")

        self.gather_processed_inputs()
        self.prepare_clustering_dicts()
        self.subsample_clusters()

    def prepare_clustering_dicts(self) -> None:
        if self.cluster_file is not None:
            with open(self.cluster_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                cluster_data = list(reader)

            self.all_complex_files = self.complex_files
            # Remove complexes that did not pass preprocessing and loading tests
            self.complex_to_cluster = {
                f"heterograph-{row['complex_name']}": row["cluster_id"]
                for row in cluster_data
                if f"heterograph-{row['complex_name']}" in self.complex_files
            }
            self.cluster_to_complex: dict[str, list] = {}
            for key, value in self.complex_to_cluster.items():
                if value not in self.cluster_to_complex:
                    self.cluster_to_complex[value] = [key]
                else:
                    self.cluster_to_complex[value].append(key)

    def subsample_clusters(self) -> None:
        if self.cluster_file is not None:
            subsampled_cluster_files = []
            for cluster_complexes in self.cluster_to_complex.values():
                random.shuffle(cluster_complexes)
                subsampled_cluster_files.extend(
                    cluster_complexes[: self.complexes_per_cluster]
                )
            self.complex_files = subsampled_cluster_files

    def check_processed_inputs(self):
        if not os.path.exists(self.cache_path):
            return False

        else:
            complex_names_all = read_strings_from_txt(self.split_path)
            if self.limit_complexes is not None and self.limit_complexes != 0:
                complex_names_all = complex_names_all[: self.limit_complexes]

            complexes_available = [
                complex_name
                for complex_name in complex_names_all
                if os.path.exists(f"{self.cache_path}/heterograph-{complex_name}-0.pt")
            ]

            # complexes_available = [
            #     filename.removeprefix("heterograph-").removesuffix("-0.pt")
            #     for filename in os.listdir(self.cache_path)
            #     if "heterograph" in filename
            # ]

            if not len(complexes_available):
                print("Directory found but no complexes available.", flush=True)
                return False

            if not len(set(complex_names_all).intersection(set(complexes_available))):
                print("No common complexes.", flush=True)
                return False

        return True

    def gather_processed_inputs(self):
        print(
            f"Loading each complex individually from: {self.cache_path}",
            flush=True,
        )
        print(flush=True)
        complex_names_all = read_strings_from_txt(self.split_path)

        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]

        self.complex_files = [
            f"heterograph-{name}-0"
            for name in complex_names_all
            if os.path.exists(f"{self.cache_path}/heterograph-{name}-0.pt")
        ]

    def len(self):
        return len(self.complex_files) * self.multiplicity

    def get(self, idx):
        if self.multiplicity:
            idx = idx % len(self.complex_files)

        name = self.complex_files[idx]
        complex_graph = torch.load(f"{self.cache_path}/{name}.pt")
        return complex_graph


class DockingDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: DockingDataConfig,
        transform_cfg: dict[str, Any],
    ):
        super().__init__()
        assert (
            data_cfg.dataset in AVAILABLE_DATASETS
        ), f"Dataset={data_cfg.dataset} not found in {AVAILABLE_DATASETS}"

        self.data_cfg = data_cfg
        self.transform_cfg = transform_cfg

        train_transform = construct_transform(cfg=transform_cfg, mode="train")
        self._train_dataset = DockingDataset(
            transform=train_transform,
            dataset=data_cfg.dataset,
            cache_path=data_cfg.cache_path,
            split_path=data_cfg.split_train,
            cluster_file=data_cfg.cluster_file,
            complexes_per_cluster=data_cfg.complexes_per_cluster,
            limit_complexes=data_cfg.limit_complexes,
            multiplicity=data_cfg.multiplicity,
            num_workers=data_cfg.num_workers,
            require_ligand=data_cfg.require_ligand,
        )

        val_transform = construct_transform(cfg=transform_cfg, mode="val")
        self._val_dataset = DockingDataset(
            transform=val_transform,
            dataset=data_cfg.dataset,
            cache_path=data_cfg.cache_path,
            split_path=data_cfg.split_val,
            cluster_file=data_cfg.cluster_file,
            complexes_per_cluster=data_cfg.complexes_per_cluster,
            limit_complexes=data_cfg.limit_complexes,
            multiplicity=min(data_cfg.multiplicity, 5),
            num_workers=data_cfg.num_workers,
            require_ligand=data_cfg.require_ligand,
        )

        if data_cfg.run_val_inference:
            inf_transform = construct_transform(cfg=transform_cfg, mode="inference")
            inf_complexes = [
                self._val_dataset.get(idx)
                for idx in range(
                    min(data_cfg.num_inference_complexes, len(self._val_dataset))
                )
            ]
            if len(inf_complexes) == 1:
                inf_complexes = inf_complexes * 20

            self._inf_dataset = ListDataset(inf_complexes, transform=inf_transform)

    def setup(self, stage):
        return

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_dataloader_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=self.data_cfg.dataloader_drop_last,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_dataloader_workers,
            shuffle=False,
            pin_memory=self.data_cfg.pin_memory,
            drop_last=self.data_cfg.dataloader_drop_last,
        )
        val_loaders = [val_loader]

        if self.data_cfg.run_val_inference:
            inf_dataset = self._inf_dataset
            val_loaders.append(
                DataLoader(dataset=inf_dataset, batch_size=1, shuffle=False)
            )

        loader = CombinedLoader(val_loaders, mode="sequential")
        return loader
