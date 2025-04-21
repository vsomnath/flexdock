import logging
import os

from rdkit import Chem
import torch

from flexdock.data.parse.molecule import read_mols
from flexdock.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from flexdock.data.conformers.molecule import generate_conformer


class ComplexParser:
    def __init__(self, esm_embeddings_path: str = None):
        self.esm_embeddings_path = esm_embeddings_path

    def parse_complex(self, complex_dict):
        # Load ligand(s) from SMILES or sdf/mol2 file
        ligand_inputs = self.parse_ligand(complex_dict)
        if ligand_inputs is None:
            return None

        # Load apo and optionally, holo protein
        protein_inputs = self.parse_protein(complex_dict)
        if protein_inputs is None:
            return None

        # Load ESM Embeddings
        esm_embeddings = self.load_esm_embeddings(complex_dict["name"])
        protein_inputs["lm_embedding"] = esm_embeddings

        return {**complex_dict, **ligand_inputs, **protein_inputs}

    def parse_ligand(self, complex_dict):
        base_dir = complex_dict["base_dir"]
        name = complex_dict["name"]
        ligand_description = complex_dict["ligand_description"]

        if ligand_description == "filename":
            try:
                ligs = read_mols(base_dir, name, remove_hs=False)
            except Exception as e:
                logging.error(
                    f"{name}: Failed to load ligand due to {e}", exc_info=True
                )
                return None
        elif ligand_description == "smiles":
            mol = Chem.MolFromSmiles(complex_dict["ligand_input"])

            if mol is not None:
                mol = Chem.AddHs(mol)
                generate_conformer(mol)
                ligs = [mol]
            else:
                return None

        return {"ligand": ligs}

    def parse_protein(self, complex_dict):
        name = complex_dict["name"]
        apo_rec_path = complex_dict.get("apo_rec_path", None)
        holo_rec_path = complex_dict.get("holo_rec_path", None)

        if apo_rec_path is None and holo_rec_path is None:
            raise ValueError(
                f"Apo Path={apo_rec_path} and Holo Path={holo_rec_path} not found"
            )

        apo_rec_struct = None
        # Load apo structure if it exists
        if apo_rec_path is not None:
            try:
                apo_rec_struct = parse_pdb_pmd(
                    apo_rec_path, remove_hs=True, reorder=True
                )
            except Exception as e:
                logging.error(
                    f"{name}: Could not parse apo structure at {apo_rec_path} due to {e}"
                )
                apo_rec_struct = None

        holo_rec_struct = None
        # Load holo structure if it exists
        if holo_rec_path is not None:
            try:
                holo_rec_struct = parse_pdb_pmd(
                    holo_rec_path, remove_hs=True, reorder=True
                )

            except Exception as e:
                logging.error(
                    f"{name}: Unable to parse holo structure at {holo_rec_path} due to {e}"
                )
                holo_rec_struct = None

        # Check number of residues and atoms if both apo and holo structures are loaded
        if apo_rec_struct is not None and holo_rec_struct is not None:
            try:
                assert len(holo_rec_struct.residues) == len(
                    apo_rec_struct.residues
                ), "APO and HOLO structures do not have the same number of residues"
                assert all(
                    holo_res.name == apo_res.name
                    for holo_res, apo_res in zip(
                        holo_rec_struct.residues, apo_rec_struct.residues
                    )
                ), "APO and HOLO structures do not have the same atoms"
                assert len(holo_rec_struct.atoms) == len(
                    apo_rec_struct.atoms
                ), "APO and HOLO structures do not have the same number of atoms"
                assert all(
                    holo_atom.name == apo_atom.name
                    for holo_atom, apo_atom in zip(
                        holo_rec_struct.atoms, apo_rec_struct.atoms
                    )
                ), "APO and HOLO structures do not have the same atoms"

            except Exception as e:
                logging.error(
                    f"Failed matching checks for apo and holo structures due to {e}"
                )
                return None

        # Check if both structures are None
        if apo_rec_struct is None and holo_rec_struct is None:
            logging.info(f"{name}: Preprocessing failed")
            return None

        return {"apo_rec_struct": apo_rec_struct, "holo_rec_struct": holo_rec_struct}

    def load_esm_embeddings(self, name):
        if self.esm_embeddings_path is None:
            return None
        else:
            if not os.path.exists(f"{self.esm_embeddings_path}/{name}.pt"):
                return None
            else:
                lm_embeddings = torch.load(f"{self.esm_embeddings_path}/{name}.pt")
                chain_idx_sorted = sorted(list(lm_embeddings.keys()))
                sorted_chains = [lm_embeddings[idx] for idx in chain_idx_sorted]
                return sorted_chains
