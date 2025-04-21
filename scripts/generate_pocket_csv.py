import os
import pandas as pd
import torch
import logging

from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from flexdock.data.parse.molecule import read_mols
from flexdock.data.feature.protein import get_nearby_residue_mask


BASE_DIR = "../ligbind/data/PDBBIND_atomCorrected"


def get_pocket_residue_str(
    holo_pos,
    ligand_pos,
    atom_rec_index,
    pocket_cutoff: float = 5.0,
    pocket_min_size: int = 1,
):
    nearby_residue_mask = get_nearby_residue_mask(
        atom_pos=holo_pos,
        lig_pos=ligand_pos,
        atom_rec_index=atom_rec_index,
        cutoff=pocket_cutoff,
        min_residues=pocket_min_size,
    )

    nearby_residue_idxs = torch.argwhere(nearby_residue_mask).squeeze()
    pocket_residue_str = ",".join(str(idx.item() + 1) for idx in nearby_residue_idxs)
    return pocket_residue_str


def run():
    logging.getLogger().setLevel("INFO")

    df_list = []
    complexes_test = read_strings_from_txt("data/splits/timesplit_test")

    for complex_id in complexes_test:
        try:
            lig_mol = read_mols(BASE_DIR, complex_id, remove_hs=True)[0]
            lig_pos = torch.tensor(lig_mol.GetConformer().GetPositions()).float()

            apo_rec_path = f"{BASE_DIR}/{complex_id}/{complex_id}_protein_esmfold_aligned_tr_fix.pdb"
            holo_rec_path = (
                f"{BASE_DIR}/{complex_id}/{complex_id}_protein_processed_fix.pdb"
            )

            if not os.path.exists(apo_rec_path) or not os.path.exists(holo_rec_path):
                continue

            holo_rec_struct = parse_pdb_pmd(
                path=holo_rec_path, remove_hs=True, reorder=True
            )
            holo_pos = torch.tensor(holo_rec_struct.get_coordinates(0)).float()

            atom_rec_index = torch.tensor(
                [atom.residue.idx for atom in holo_rec_struct.atoms]
            ).long()

            pocket_residue_str = get_pocket_residue_str(
                holo_pos=holo_pos,
                ligand_pos=lig_pos,
                atom_rec_index=atom_rec_index,
                pocket_cutoff=5.0,
                pocket_min_size=1,
            )

            complex_dict = {
                "pdbid": complex_id,
                "apo_protein_file": apo_rec_path,
                "holo_protein_file": holo_rec_path,
                "base_dir": BASE_DIR,
                "ligand_input": None,
                "ligand_description": "filename",
                "pocket_residues": pocket_residue_str,
            }
            df_list.append(complex_dict)

        except Exception as e:
            logging.error(f"Failed to add to inference file due to {e}", exc_info=True)
            continue

    df = pd.DataFrame.from_dict(df_list)
    logging.info(f"Number of examples processed={df.shape[0]}")
    df.to_csv("inference_pdbbind.csv", index=None)


if __name__ == "__main__":
    run()
