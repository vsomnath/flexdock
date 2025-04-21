import os
import torch
import numpy as np
from torch_geometric.data import HeteroData
import traceback


from flexdock.data.parse.molecule import read_mols
from flexdock.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from flexdock.data.feature.molecule import (
    get_lig_graph_with_matching,
    get_posebusters_edge_index,
)
from flexdock.data.conformers.protein import (
    sidechain_conformer_matching as sc_conformer_matching,
)
from flexdock.data.feature.protein import (
    get_fullrec_graph,
    construct_protein_edge_index,
)
from flexdock.data.feature.helpers import align_sidechains_to_backbone_torch
from flexdock.geometry.ops import rigid_transform_kabsch


class ComplexData(HeteroData):
    def __cat_dim__(self, key, value, store=None, *args, **kwargs):
        if key in {
            "posebusters_edge_index",
            "angle_2_index",
            "angle_index",
            "torsion_index",
        }:
            return -1
        elif key in {"atom_fragment_index", "lig_fragment_index"}:
            return -1
        return super().__cat_dim__(key, value, store=store, *args, **kwargs)

    def __inc__(self, key, value, store=None, *args, **kwargs):
        if key in {"posebusters_edge_index", "angle_index", "torsion_index"}:
            return self["ligand"].num_nodes
        elif key == "atom_fragment_index":
            return torch.tensor(
                [self["atom", "atom_bond", "atom"].num_edges, self["atom"].num_nodes]
            ).view(-1, 1)
        elif key == "lig_fragment_index":
            return torch.tensor(
                [
                    self["ligand", "lig_bond", "ligand"].num_edges,
                    self["ligand"].num_nodes,
                ]
            ).view(-1, 1)
        elif key == "angle_2_index":
            return self["ligand", "lig_bond", "ligand"].num_edges
        return super().__inc__(key, value, store=store, *args, **kwargs)


class RMSDTooLarge(Exception):
    def __init__(self, rmsd, max_rmsd):
        super().__init__(
            "RMSD between the two specified structures "
            + f"is too large. rmsd = {rmsd} > {max_rmsd} = max rmsd"
        )


class DataPipeline:
    def __init__(
        self,
        matching: bool = True,
        popsize: int = 15,
        maxiter: int = 15,
        keep_original: bool = False,
        max_lig_size: int = None,
        remove_hs: bool = False,
        num_conformers: int = 1,
        all_atoms: bool = False,
        require_ligand: bool = False,
        keep_local_structures: bool = False,
        match_max_rmsd: float = None,
        pocket_reduction: bool = False,
        old_pocket_selection: bool = False,
        pocket_buffer: float = 10,
        nearby_residues_atomic_radius: float = 5,
        nearby_residues_atomic_min: int = 1,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
        include_miscellaneous_atoms: bool = False,
        cache_individual: bool = False,
        use_origpos_pocket: bool = False,
        add_nearby_residues_pocket: bool = False,
        **kwargs,
    ):
        self.flexible_backbone, self.flexible_sidechains = (
            flexible_backbone,
            flexible_sidechains,
        )
        self.match_max_rmsd = match_max_rmsd
        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.old_pocket_selection = old_pocket_selection
        self.nearby_residues_atomic_radius = nearby_residues_atomic_radius
        self.nearby_residues_atomic_min = nearby_residues_atomic_min

        self.require_ligand = require_ligand
        self.remove_hs = remove_hs
        self.max_lig_size = max_lig_size
        self.keep_local_structures = keep_local_structures
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.all_atoms = all_atoms
        self.include_miscellaneous_atoms = include_miscellaneous_atoms

        self.cache_individual = cache_individual

        self.use_origpos_pocket = use_origpos_pocket
        self.add_nearby_residues_pocket = add_nearby_residues_pocket

    def process_complex(self, complex_dict):
        raise NotImplementedError("Subclasses must implement for themselves")


class DockingPipeline(DataPipeline):
    def process_complex(self, complex_dict):
        base_dir = complex_dict["base_dir"]
        name = complex_dict["name"]
        lm_embedding_chains = complex_dict["lm_embedding"]
        ligand = complex_dict["ligand"]
        ligand_description = complex_dict["ligand_desc"]
        holo_rec_path = complex_dict.get("holo_protein_file", None)
        apo_rec_path = complex_dict["apo_protein_file"]

        if not os.path.exists(os.path.join(base_dir, name)) and ligand is None:
            print("Folder not found", name)
            if self.cache_individual:
                return [], [], []
            return [], []

        if ligand is not None:
            # apo_rec_path, holo_rec_path = name, None
            apo_rec_struct = parse_pdb_pmd(apo_rec_path, remove_hs=True, reorder=True)
            name = f"{name}____{ligand_description}"
            ligs = [ligand]

        else:
            try:
                # apo_rec_path = os.path.join(base_dir, name, f'{name}_{apo_protein_file}.pdb')
                # holo_rec_path = os.path.join(base_dir, name, f'{name}_{holo_protein_file}.pdb')
                apo_rec_struct = parse_pdb_pmd(
                    apo_rec_path, remove_hs=True, reorder=True
                )
                holo_rec_struct = parse_pdb_pmd(
                    holo_rec_path, remove_hs=True, reorder=True
                )
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
                print(f"Skipping {name} because of the error:")
                print(e)
                print(traceback.format_exc())
                if self.cache_individual:
                    return [], [], []
                return [], []

            ligs = read_mols(base_dir, name, remove_hs=False)

        complex_graphs = []
        if self.cache_individual:
            names = []

        for lig_idx, lig in enumerate(ligs):
            if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(
                    f"Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is "
                    f"larger than max_lig_size {self.max_lig_size}. "
                    f"Not including {name} in preprocessed data."
                )
                continue

            try:
                complex_graph = ComplexData()
                complex_graph["name"] = name
                import copy

                complex_graph.mol = copy.deepcopy(lig)
                complex_graph["apo_rec_path"] = apo_rec_path
                complex_graph["holo_rec_path"] = holo_rec_path

                get_lig_graph_with_matching(
                    lig,
                    complex_graph,
                    self.popsize,
                    self.maxiter,
                    self.matching,
                    self.keep_original,
                    self.num_conformers,
                    remove_hs=self.remove_hs,
                    new_version=True,
                )
                apo_rec_pos = torch.tensor(
                    apo_rec_struct.get_coordinates(0), dtype=torch.float32
                )
                holo_rec_pos = (
                    None
                    if holo_rec_struct is None
                    else torch.tensor(
                        holo_rec_struct.get_coordinates(0), dtype=torch.float32
                    )
                )

                # # Cutout Binding Pocket
                # if self.pocket_reduction:
                #     pocket_center, res_pocket_mask, atom_pocket_mask, nearby_residues = get_binding_pocket_masks(
                #         apo_rec_pos,
                #         holo_rec_pos if holo_rec_pos is not None else apo_rec_pos,
                #         complex_graph['ligand'].pos,
                #         torch.tensor([atom.name == 'CA' for atom in apo_rec_struct.atoms], dtype=bool),
                #         torch.tensor([atom.residue.idx for atom in apo_rec_struct.atoms], dtype=torch.int64),
                #         pocket_cutoff=self.nearby_residues_atomic_radius,
                #         pocket_min_size=self.nearby_residues_atomic_min,
                #         pocket_buffer=self.pocket_buffer
                #     )
                # else:
                #     res_pocket_mask = torch.ones(len(holo_rec_struct.residues), dtype=torch.bool)
                #     atom_pocket_mask = torch.ones(len(holo_rec_struct.atoms), dtype=torch.bool)

                # amber_pocket_mask = ':' + ','.join([str(idx + 1) for idx in torch.argwhere(res_pocket_mask).squeeze().numpy().tolist()])
                # apo_rec_struct = apo_rec_struct[amber_pocket_mask]
                # holo_rec_struct = holo_rec_struct[amber_pocket_mask]
                # apo_rec_pos = apo_rec_pos[atom_pocket_mask]

                # if holo_rec_pos is not None:
                #     holo_rec_pos = holo_rec_pos[atom_pocket_mask]
                if lm_embedding_chains is not None:
                    lm_embeddings = torch.cat(lm_embedding_chains, dim=0)
                else:
                    lm_embeddings = None

                # complex_graph.pocket_mask = amber_pocket_mask
                if apo_rec_struct is not None:
                    get_fullrec_graph(
                        apo_rec_struct, complex_graph, lm_embeddings=lm_embeddings
                    )
                    (
                        edge_index,
                        squeeze_mask,
                        edge_mask,
                        ring_sub_mask,
                        ring_flip_mask,
                        fragment_index,
                        angle_2_index,
                        res_to_rotate,
                    ) = construct_protein_edge_index(apo_rec_struct)
                else:
                    get_fullrec_graph(
                        holo_rec_struct, complex_graph, lm_embeddings=lm_embeddings
                    )
                    (
                        edge_index,
                        squeeze_mask,
                        edge_mask,
                        ring_sub_mask,
                        ring_flip_mask,
                        fragment_index,
                        angle_2_index,
                        res_to_rotate,
                    ) = construct_protein_edge_index(holo_rec_struct)

                if holo_rec_pos is not None:
                    complex_graph["atom"].orig_holo_pos = holo_rec_pos
                if apo_rec_pos is not None:
                    complex_graph["atom"].orig_apo_pos = apo_rec_pos

                complex_graph[
                    "atom", "atom_bond", "atom"
                ].edge_index = torch.from_numpy(edge_index)
                complex_graph[
                    "atom", "atom_bond", "atom"
                ].squeeze_mask = torch.from_numpy(squeeze_mask)
                complex_graph["atom", "atom_bond", "atom"].edge_mask = torch.from_numpy(
                    edge_mask
                )
                complex_graph[
                    "atom", "atom_bond", "atom"
                ].ring_sub_mask = torch.from_numpy(ring_sub_mask)
                complex_graph[
                    "atom", "atom_bond", "atom"
                ].ring_flip_mask = torch.from_numpy(ring_flip_mask)
                complex_graph[
                    "atom_bond", "atom"
                ].atom_fragment_index = torch.from_numpy(fragment_index)
                complex_graph[
                    "atom", "atom_bond", "atom"
                ].res_to_rotate = torch.from_numpy(res_to_rotate)
                complex_graph[
                    "atom_bond", "atom_angle", "atom_bond"
                ].angle_2_index = torch.from_numpy(angle_2_index)

                if self.flexible_backbone or self.flexible_sidechains:
                    R, tr = rigid_transform_kabsch(
                        apo_rec_pos[complex_graph["atom"].ca_mask],
                        holo_rec_pos[complex_graph["atom"].ca_mask],
                    )
                    aligned_apo_rec_pos = apo_rec_pos @ R.T + tr.unsqueeze(-2)

                    rmsd = torch.sqrt(
                        torch.mean(
                            torch.sum((holo_rec_pos - aligned_apo_rec_pos) ** 2, axis=1)
                        )
                    ).item()
                    print(
                        "RMSD between aligned apo and holo full atomic structures pocket",
                        rmsd,
                    )

                    # if self.match_max_rmsd is not None and rmsd > self.match_max_rmsd:
                    #     raise RMSDTooLarge(rmsd, self.match_max_rmsd)

                    complex_graph[
                        "atom"
                    ].orig_aligned_apo_pos = aligned_apo_rec_pos.clone()
                    aligned_apo_rec_pos, rot_vec = align_sidechains_to_backbone_torch(
                        aligned_apo_rec_pos,
                        holo_rec_pos,
                        complex_graph["atom"].ca_mask,
                        complex_graph["atom"].c_mask,
                        complex_graph["atom"].n_mask,
                        complex_graph["atom", "receptor"].edge_index[1],
                    )
                    complex_graph["receptor"].rot_vec = rot_vec
                    lens_receptors = np.asarray(
                        [len([a for a in res.atoms]) for res in apo_rec_struct.residues]
                    )
                    complex_graph["receptor"].lens_receptors = torch.from_numpy(
                        lens_receptors
                    ).long()

                    rmsd = torch.sqrt(
                        torch.mean(
                            torch.sum((holo_rec_pos - aligned_apo_rec_pos) ** 2, dim=1)
                        )
                    )
                    print(
                        "RMSD between aligned apo and holo full atomic structures after sidechain alignment",
                        rmsd.item(),
                    )

                if self.flexible_backbone or self.flexible_sidechains:
                    (
                        conformer_matched_apo_rec_pos,
                        sc_conformer_match_rotations,
                        sc_conformer_match_improvements,
                    ) = sc_conformer_matching(
                        aligned_apo_rec_pos,
                        holo_rec_pos,
                        edge_index=complex_graph[
                            "atom", "atom_bond", "atom"
                        ].edge_index,
                        mask_rotate=complex_graph[
                            "atom", "atom_bond", "atom"
                        ].edge_mask,
                        atom_rec_index=complex_graph["atom", "receptor"].edge_index[1],
                        fragment_index=complex_graph[
                            "atom_bond", "atom"
                        ].atom_fragment_index,
                        res_to_rotate=complex_graph[
                            "atom", "atom_bond", "atom"
                        ].res_to_rotate,
                        ligand=complex_graph["ligand"].pos,
                        score="dist",
                    )

                    complex_graph.sc_conformer_match_rotations = torch.tensor(
                        sc_conformer_match_rotations
                    )
                    complex_graph.sc_conformer_match_improvements = (
                        sc_conformer_match_improvements
                    )

                    conformer_matched_apo_rec_pos = torch.tensor(
                        conformer_matched_apo_rec_pos
                    ).float()
                    complex_graph["atom"].pos_sc_matched = conformer_matched_apo_rec_pos
                    complex_graph["atom"].pos = conformer_matched_apo_rec_pos
                    complex_graph["receptor"].pos = complex_graph["atom"].pos[
                        complex_graph["atom"].ca_mask
                    ]
                    complex_graph[
                        "atom", "atom_bond", "atom"
                    ].sc_conformer_match_rotations = torch.zeros(
                        complex_graph["atom", "atom_bond", "atom"].edge_mask.shape,
                        dtype=torch.float32,
                    )
                    complex_graph[
                        "atom", "atom_bond", "atom"
                    ].sc_conformer_match_rotations[
                        complex_graph["atom", "atom_bond", "atom"].edge_mask
                    ] = torch.tensor(
                        sc_conformer_match_rotations
                    ).float()
                    complex_graph.rmsd_matching = torch.sqrt(
                        torch.mean(
                            torch.sum(
                                (conformer_matched_apo_rec_pos - holo_rec_pos) ** 2,
                                dim=-1,
                            ),
                            dim=-1,
                        )
                    )

                    rmsd = torch.sqrt(
                        torch.mean(
                            torch.sum(
                                (holo_rec_pos - conformer_matched_apo_rec_pos) ** 2,
                                dim=1,
                            )
                        )
                    )
                    print(
                        "RMSD between aligned apo and holo full atomic structures after sidechain conformer matching",
                        rmsd.item(),
                    )

            except Exception as e:
                print(f"Skipping {name} because of the error:")
                print(e)
                print(traceback.format_exc())
                continue

            # if self.pocket_reduction:
            #     protein_center = pocket_center[None, :]
            # else:
            #     # Center the protein around the mean C-alpha position
            #     protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)

            complex_graph = get_posebusters_edge_index(complex_graph)

            # complex_graph = self.center_complex(complex_graph, protein_center)
            complex_graphs.append(complex_graph)
            if self.cache_individual:
                names.append(name + f"-{lig_idx}")

        if self.cache_individual:
            return complex_graphs, ligs, names
        return complex_graphs, ligs

    # def center_complex(self, complex_graph, protein_center):
    #     # Center the protein around the specified pos
    #     complex_graph['receptor'].pos -= protein_center
    #     if self.flexible_backbone or self.flexible_sidechains:
    #         complex_graph['receptor'].orig_apo_pos -= protein_center
    #         complex_graph['receptor'].orig_aligned_apo_pos -= protein_center
    #         complex_graph['receptor'].orig_holo_pos -= protein_center

    #     if self.all_atoms:
    #         complex_graph['atom'].pos -= protein_center

    #     if self.flexible_backbone or self.flexible_sidechains:
    #         complex_graph['atom'].orig_apo_pos -= protein_center
    #         complex_graph['atom'].orig_aligned_apo_pos -= protein_center
    #         complex_graph['atom'].orig_holo_pos -= protein_center

    #     if (not self.matching) or self.num_conformers == 1:
    #         complex_graph['ligand'].pos -= protein_center
    #     else:
    #         for p in complex_graph['ligand'].pos:
    #             p -= protein_center

    #     complex_graph.original_center = protein_center

    #     return complex_graph


def get_pipeline_base_args(args, mode="train"):
    pipeline_args = {
        "matching": not args.no_torsion,
        "popsize": args.matching_popsize,
        "maxiter": args.matching_maxiter,
        "keep_original": True,
        "keep_local_structures": getattr(args, "keep_local_structures", False),
        "max_lig_size": args.max_lig_size,
        "match_max_rmsd": args.match_max_rmsd if mode == "train" else None,
        "remove_hs": args.remove_hs,
        "num_conformers": args.num_conformers,
        "all_atoms": args.all_atoms,
        "require_ligand": True,
        "pocket_reduction": args.pocket_reduction,
        "pocket_buffer": args.pocket_buffer,
        "nearby_residues_atomic_radius": args.nearby_residues_atomic_radius,
        "nearby_residues_atomic_min": args.nearby_residues_atomic_min,
        "flexible_backbone": args.flexible_backbone,
        "flexible_sidechains": args.flexible_sidechains,
        "cache_individual": args.cache_individual,
        # Whether to use complex_graph['ligand'].orig_pos or complex_graph['ligand'].pos
        "use_origpos_pocket": getattr(args, "use_origpos_pocket", False),
        # Whether to add nearby residues to pocket
        "add_nearby_residues_pocket": getattr(
            args, "add_nearby_residues_pocket", False
        ),
    }

    return pipeline_args


def get_pipeline(args, mode="train"):
    pipeline_args = get_pipeline_base_args(args=args, mode=mode)
    return DockingPipeline(**pipeline_args)
