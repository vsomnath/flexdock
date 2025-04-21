from copy import deepcopy
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import pytraj as pt
from posebusters import PoseBusters
from tempfile import TemporaryDirectory

buster = PoseBusters("dock")


def add_conformer(mol, coords, add_hs: bool = False):
    if add_hs:
        # Hydrogens are needed to get good structures, and prevent chirality issues
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol, params)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer(cid)
    for atom_idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, Point3D(*coords[atom_idx].tolist()))
    return mol


def bust(input_dict):
    try:
        # TODO: uncomment this later if possible
        # avg_energy_df = pd.read_csv("test_avg_energies.csv").set_index("pdb_id")

        pred_mol = deepcopy(input_dict["mol"])
        pred_mol.RemoveAllConformers()
        filterHs = input_dict["filterHs"]
        pred_mol = add_conformer(
            pred_mol,
            input_dict["lig_pred"][
                filterHs
            ],  # Typically for SDF files, Hs are at the end
            add_hs=input_dict["add_hs"],
        )

        # conf_energy = get_conf_energy(pred_mol, conf_id=0)
        # avg_energy = avg_energy_df.loc[input_dict['pdb_id']]['avg_energy']
        # energy_check = (conf_energy / avg_energy) <= 100

        with TemporaryDirectory() as tmp_dir:
            mol_cond_path = os.path.join(
                tmp_dir, f'{input_dict["pdb_id"]}_cond_protein.pdb'
            )
            apo_traj = pt.load(input_dict["apo_rec_path"])
            full_atom_pred = apo_traj.xyz
            full_atom_pred[
                :, pt.select(input_dict["pocket_mask"], apo_traj.topology)
            ] = input_dict["atom_pred"]
            pred_traj = pt.Trajectory(xyz=full_atom_pred, top=apo_traj.topology)
            pt.write_traj(mol_cond_path, pred_traj, overwrite=True)

            bust_df = buster.bust(mol_pred=pred_mol, mol_cond=mol_cond_path)

            bust_df["pb_valid"] = bust_df.all(axis=1)

        for key in [
            "pdb_id",
            "time",
            "success",
            "lig_rmsds",
            "lig_centered_rmsds",
            "lig_aligned_rmsds",
            "lig_tr_mags",
            "lig_rot_mags",
            "aa_rmsds",
            "bb_rmsds",
            "lig_scrmsds",
        ]:
            bust_df[key] = input_dict[key]
        return bust_df.set_index("pdb_id")

    except Exception as e:
        print(f"Failed bust due to {e}")
        return None
