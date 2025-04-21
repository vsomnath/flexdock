import copy

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, RemoveHs
from scipy.optimize import differential_evolution
from spyrmsd import rmsd, molecule

from flexdock.data.conformers.exceptions import time_limit


def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print(
            "rdkit coords could not be generated without using random coords. using random coords now."
        )
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)


LigRMSD = AllChem.AlignMol


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]
    )


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, _ in enumerate(mol.GetAtoms()):
        G.add_node(i)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        comp = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(comp) < 2:
            continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append((n0[0], e[0], e[1], n1[0]))
    return torsions_list


class LigandOptimizer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super().__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return LigRMSD(self.mol, self.true_mol, self.probe_id, self.ref_id)

    def apply_rotations(self, mol, values, rotable_bonds, conf_id):
        opt_mol = copy.copy(mol)
        [
            SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r])
            for r in range(len(rotable_bonds))
        ]
        return opt_mol


def ligand_conformer_matching(
    mol,
    true_mol,
    rotable_bonds,
    probe_id=-1,
    ref_id=-1,
    seed=0,
    popsize=15,
    maxiter=500,
    mutation=(0.5, 1),
    recombination=0.8,
):
    opt = LigandOptimizer(
        mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id
    )
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    result = differential_evolution(
        opt.score_conformation,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        disp=False,
        seed=seed,
    )
    opt_mol = opt.apply_rotations(
        opt.mol, result["x"], opt.rotable_bonds, conf_id=probe_id
    )

    return opt_mol


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = (
            mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        )
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)
