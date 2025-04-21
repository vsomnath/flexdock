import os
import dataclasses
from joblib import Parallel, delayed
import pickle
import logging

import torch

from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.parse.parser import ComplexParser
from flexdock.data.feature.featurizer import Featurizer


@dataclasses.dataclass
class TrainingPipelineConfig:
    dataset: str
    complex_file: str
    data_dir: str
    cache_path: str
    apo_protein_file: str
    holo_protein_file: str
    num_workers: int = 1
    esm_embeddings_path: str = None


class TrainingDataPipeline:
    def __init__(
        self,
        config: TrainingPipelineConfig,
        featurizer_cfg,
    ):
        self.config = config
        self.featurizer = Featurizer.from_config(featurizer_cfg)
        self.parser = ComplexParser(esm_embeddings_path=self.config.esm_embeddings_path)
        self.apo_protein_file = config.apo_protein_file
        self.holo_protein_file = config.holo_protein_file
        self.base_dir = config.data_dir

    def process_all_complexes(self):
        logging.info(
            f"Processing complexes from [{self.config.complex_file}]"
            f"and saving it to [{self.config.cache_path}]"
        )
        os.makedirs(self.config.cache_path, exist_ok=True)

        complex_names_all = read_strings_from_txt(self.config.complex_file)
        logging.info(f"Loading {len(complex_names_all)} complexes.")

        CHUNK_SIZE = 1000

        processed_names = []

        list_indices = list(range(len(complex_names_all) // CHUNK_SIZE + 1))
        # random.shuffle(list_indices)
        for i in list_indices:
            complex_names = complex_names_all[CHUNK_SIZE * i : CHUNK_SIZE * (i + 1)]

            complex_inputs_shard = [
                self.parser.parse_complex(self.prepare_input_files(complex_name))
                for idx, complex_name in enumerate(complex_names)
            ]

            logging(f"Num workers={self.config.num_workers}")
            with Parallel(n_jobs=self.config.num_workers, verbose=5) as parallel:
                results = parallel(
                    delayed(self.featurizer.featurize_complex)(complex_inputs)
                    for complex_inputs in complex_inputs_shard
                )

            for result in results:
                if result is None:
                    continue

                complex_graph = result["complex_graph"]
                ligand = result["ligand"]
                name = result["name"]

                torch.save(
                    complex_graph,
                    f"{self.config.cache_path}/heterograph-{name}.pt",
                )

                with open(
                    f"{self.config.cache_path}/rdkit_ligand-{name}.pkl", "wb"
                ) as f:
                    pickle.dump((ligand[0]), f)
                processed_names.append(name)

        with open(f"{self.full_cache_path}/complex_names.pkl", "wb") as f:
            pickle.dump(processed_names, f)

    def prepare_input_files(self, complex_name):
        if self.config.dataset == "pdbbind":
            complex_dict = {
                "dataset": self.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "apo_protein_file": f"{self.base_dir}/{complex_name}/{complex_name}_{self.apo_protein_file}.pdb",
                "holo_protein_file": f"{self.base_dir}/{complex_name}/{complex_name}_{self.holo_protein_file}.pdb",
            }

        elif self.config.dataset == "plinder":
            complex_dict = {
                "dataset": self.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "apo_protein_file": f"{self.base_dir}/{complex_name}/{self.apo_protein_file}.pdb",
                "holo_protein_file": f"{self.base_dir}/{complex_name}/{self.holo_protein_file}.pdb",
            }
        return complex_dict
