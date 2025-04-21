import copy

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RemoveHs, BondType as BT, rdDistGeom, PeriodicTable
import torch
import torch.nn.functional as F

from flexdock.data.constants import allowable_features, bonds
from flexdock.data.feature.helpers import (
    safe_index,
    get_fragment_index,
    get_transformation_mask,
)
from flexdock.data.conformers.molecule import (
    generate_conformer,
    get_torsion_angles,
    ligand_conformer_matching,
)

RDLogger.DisableLog("rdApp.*")


################################################################################
# Graph Generation
################################################################################


def get_posebusters_edge_index(complex_graph):
    mol = Chem.RemoveHs(complex_graph.mol)
    complex_graph["ligand"].vdw_radii = torch.FloatTensor(
        [
            PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum())
            for atom in mol.GetAtoms()
        ]
    )
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    complex_graph[
        "ligand", "lig_edge", "ligand"
    ].posebusters_edge_index = torch.triu_indices(*bounds.shape, offset=1)
    complex_graph["ligand", "lig_edge", "ligand"].upper_bound = torch.from_numpy(
        bounds[
            complex_graph[
                "ligand", "lig_edge", "ligand"
            ].posebusters_edge_index.unbind()
        ]
    ).float()
    complex_graph["ligand", "lig_edge", "ligand"].lower_bound = torch.from_numpy(
        bounds[
            complex_graph["ligand", "lig_edge", "ligand"]
            .posebusters_edge_index[[1, 0]]
            .unbind()
        ]
    ).float()
    complex_graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask = torch.zeros(
        (
            complex_graph["ligand", "lig_edge", "ligand"].posebusters_edge_index.shape[
                1
            ],
        ),
        dtype=torch.bool,
    )
    complex_graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask = torch.zeros(
        (
            complex_graph["ligand", "lig_edge", "ligand"].posebusters_edge_index.shape[
                1
            ],
        ),
        dtype=torch.bool,
    )
    bond_index = torch.tensor(
        mol.GetSubstructMatches(Chem.MolFromSmarts("*~*")), dtype=torch.int64
    ).T
    angle_index = torch.tensor(
        mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*")), dtype=torch.int64
    ).T
    torsion_index = torch.tensor(
        mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*~*")), dtype=torch.int64
    ).T
    complex_graph["ligand", "lig_angle", "ligand"].angle_index = angle_index
    complex_graph["ligand", "lig_torsion", "ligand"].torsion_index = torsion_index

    complex_graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask[
        (
            (bond_index[0] * (2 * mol.GetNumAtoms() - bond_index[0] - 3) // 2)
            + bond_index[1]
            - 1
        )
    ] = True
    complex_graph["ligand", "lig_edge", "ligand"].posebusters_angle_mask[
        (
            (angle_index[0] * (2 * mol.GetNumAtoms() - angle_index[0] - 3) // 2)
            + angle_index[2]
            - 1
        )
    ] = True
    return complex_graph


def get_lig_graph(mol, complex_graph):
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += (
            2 * [bonds[bond.GetBondType()]]
            if bond.GetBondType() != BT.UNSPECIFIED
            else [0, 0]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph["ligand"].x = atom_feats
    complex_graph["ligand"].vdw_radii = torch.FloatTensor(
        [
            PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum())
            for atom in mol.GetAtoms()
        ]
    )
    complex_graph["ligand"].pos = lig_coords
    complex_graph["ligand", "lig_bond", "ligand"].edge_index = edge_index
    complex_graph["ligand", "lig_bond", "ligand"].edge_attr = edge_attr
    return


def get_lig_graph_with_matching(
    mol_,
    complex_graph,
    popsize,
    maxiter,
    matching,
    keep_original,
    num_conformers,
    remove_hs,
    new_version: bool = False,
):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
        if keep_original:
            complex_graph[
                "ligand"
            ].orig_pos = mol_maybe_noh.GetConformer().GetPositions()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)
        if not rotable_bonds:
            print("no_rotable_bonds but still using it")

        for i in range(num_conformers):
            mol_rdkit = copy.deepcopy(mol_)

            mol_rdkit.RemoveAllConformers()
            mol_rdkit = AllChem.AddHs(mol_rdkit)
            generate_conformer(mol_rdkit)
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
            mol = copy.deepcopy(mol_maybe_noh)
            if rotable_bonds:
                ligand_conformer_matching(
                    mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter
                )
            mol.AddConformer(mol_rdkit.GetConformer())
            rms_list = []
            AllChem.AlignMolConformers(mol, RMSlist=rms_list)
            mol_rdkit.RemoveAllConformers()
            mol_rdkit.AddConformer(mol.GetConformers()[1])

            if i == 0:
                complex_graph.rmsd_matching = rms_list[0]
                get_lig_graph(mol_rdkit, complex_graph)
            else:
                if torch.is_tensor(complex_graph["ligand"].pos):
                    complex_graph["ligand"].pos = [complex_graph["ligand"].pos]
                complex_graph["ligand"].pos.append(
                    torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float()
                )

    else:  # no matching
        complex_graph.rmsd_matching = 0
        if remove_hs:
            mol_ = RemoveHs(mol_)
        get_lig_graph(mol_, complex_graph)

    if new_version:
        (
            squeeze_mask,
            edge_mask,
            ring_sub_mask,
            ring_flip_mask,
            fragment_index,
            angle_2_index,
        ) = get_fragment_index(
            complex_graph["ligand", "lig_bond", "ligand"].edge_index.numpy()
        )
        complex_graph["ligand"].lig_fragment_index = torch.from_numpy(
            fragment_index
        ).long()
        complex_graph["ligand"].edge_mask = torch.tensor(edge_mask)
    else:
        edge_mask, mask_rotate = get_transformation_mask(complex_graph)
        complex_graph["ligand"].edge_mask = torch.tensor(edge_mask)
        complex_graph["ligand"].mask_rotate = mask_rotate

    return


################################################################################
# Featurizing routines
################################################################################


def lig_atom_featurizer(mol):
    # ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        # g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append(
            [
                safe_index(
                    allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()
                ),
                allowable_features["possible_chirality_list"].index(
                    str(atom.GetChiralTag())
                ),
                safe_index(
                    allowable_features["possible_degree_list"], atom.GetTotalDegree()
                ),
                safe_index(
                    allowable_features["possible_formal_charge_list"],
                    atom.GetFormalCharge(),
                ),
                safe_index(
                    allowable_features["possible_implicit_valence_list"],
                    atom.GetImplicitValence(),
                ),
                safe_index(
                    allowable_features["possible_numH_list"], atom.GetTotalNumHs()
                ),
                safe_index(
                    allowable_features["possible_number_radical_e_list"],
                    atom.GetNumRadicalElectrons(),
                ),
                safe_index(
                    allowable_features["possible_hybridization_list"],
                    str(atom.GetHybridization()),
                ),
                allowable_features["possible_is_aromatic_list"].index(
                    atom.GetIsAromatic()
                ),
                safe_index(
                    allowable_features["possible_numring_list"],
                    ringinfo.NumAtomRings(idx),
                ),
                allowable_features["possible_is_in_ring3_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 3)
                ),
                allowable_features["possible_is_in_ring4_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 4)
                ),
                allowable_features["possible_is_in_ring5_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 5)
                ),
                allowable_features["possible_is_in_ring6_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 6)
                ),
                allowable_features["possible_is_in_ring7_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 7)
                ),
                allowable_features["possible_is_in_ring8_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 8)
                ),
                # g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
            ]
        )
    return torch.tensor(atom_features_list)
