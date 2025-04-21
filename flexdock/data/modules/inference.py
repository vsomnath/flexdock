import pandas as pd

import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from lightning.pytorch import LightningDataModule

from flexdock.data.feature.featurizer import Featurizer, FeaturizerConfig
from flexdock.data.parse.parser import ComplexParser
from flexdock.data.modules import ComplexData
from flexdock.data.feature.molecule import get_posebusters_edge_index


class PredictionDataset(Dataset):
    def __init__(
        self,
        input_csv: str,
        featurizer_cfg: FeaturizerConfig,
        esm_embeddings_path: str = None,
        limit_complexes: int = None,
        pocket_reduction: bool = False,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        only_nearby_residues_atomic: bool = False,
    ):
        super().__init__()
        self.input_df = pd.read_csv(input_csv, index_col=None)
        self.parser = ComplexParser(esm_embeddings_path=esm_embeddings_path)
        self.featurizer = Featurizer.from_config(cfg=featurizer_cfg)
        assert self.featurizer.cfg.matching is False, "For inference, matching=False"

        self.limit_complexes = limit_complexes

        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.only_nearby_residues_atomic = only_nearby_residues_atomic

    def get(self, idx):
        complex_info = self.input_df.iloc[idx]

        input_dict = self.prepare_input_dict(complex_info)

        complex_inputs = self.parser.parse_complex(complex_dict=input_dict)
        output_features = self.featurizer.featurize_complex(
            complex_inputs=complex_inputs
        )
        if output_features is None:
            complex_graph = ComplexData()
            complex_graph["success"] = False
            return complex_graph

        complex_graph = output_features["complex_graph"]
        complex_graph["atom"].pos = complex_graph["atom"].orig_apo_pos
        complex_graph["receptor"].pos = complex_graph["atom"].pos[
            complex_graph["atom"].ca_mask
        ]
        complex_graph["atom"].orig_aligned_apo_pos = complex_graph["atom"].orig_apo_pos

        if self.pocket_reduction:
            pocket_info = self.prepare_pocket_info(
                complex_info=complex_info, complex_graph=complex_graph
            )
            complex_graph = self.select_pocket_and_buffer(
                complex_graph, pocket_info=pocket_info
            )
            if self.only_nearby_residues_atomic:
                complex_graph = self.select_nearby_atoms(complex_graph)

        complex_graph = get_posebusters_edge_index(complex_graph)
        return complex_graph

    def prepare_input_dict(self, complex_info):
        try:
            apo_rec_path = complex_info.apo_protein_file
        except Exception:
            apo_rec_path = complex_info.apo_rec_path
        base_dir = complex_info.base_dir
        ligand_input = complex_info.ligand_input
        ligand_description = complex_info.ligand_description
        name = complex_info.pdbid

        return {
            "name": name,
            "base_dir": base_dir,
            "apo_rec_path": apo_rec_path,
            "ligand_input": ligand_input,
            "ligand_description": ligand_description,
            "holo_rec_path": None,
        }

    def prepare_pocket_info(self, complex_info, complex_graph):
        pocket_idxs_str = complex_info.pocket_residues

        # The pocket indices are labelled starting from 1. Thus we subtract 1
        pocket_indices = [int(x) - 1 for x in pocket_idxs_str.split(",")]
        pocket_residue_idxs = torch.tensor(pocket_indices).long()
        amber_pocket_mask = ":" + pocket_idxs_str

        ca_mask = complex_graph["atom"].ca_mask
        atom_pos = complex_graph["atom"].orig_apo_pos
        atom_rec_index = complex_graph[
            "atom", "atom_rec_contact", "receptor"
        ].edge_index[1]

        # Get the pocket + buffer mask for atom and residues, and pocket center
        ca_atom_idxs = torch.argwhere(ca_mask).squeeze()
        ca_pos = torch.index_select(atom_pos, -2, ca_atom_idxs)
        pocket_center = torch.index_select(ca_pos, -2, pocket_residue_idxs).mean(dim=-2)
        res_subset_mask = (
            torch.linalg.norm(ca_pos - pocket_center, dim=-1) < self.pocket_buffer
        )
        amber_subset_mask = ":" + ",".join(
            [
                str(idx + 1)
                for idx in torch.argwhere(res_subset_mask).squeeze().numpy().tolist()
            ]
        )
        atom_subset_mask = torch.index_select(res_subset_mask, -1, atom_rec_index)

        # Mask for only pocket residues
        num_residues = complex_graph["receptor"].x.shape[0]
        pocket_residue_mask = torch.zeros((num_residues,), dtype=torch.bool)
        pocket_residue_mask[pocket_indices] = True

        # Mask for atoms associated with pocket residues
        pocket_atom_mask = torch.index_select(
            pocket_residue_mask,
            dim=-1,
            index=complex_graph["atom", "atom_rec_contact", "receptor"].edge_index[1],
        )

        pocket_info = {
            "amber_pocket_mask": amber_pocket_mask,
            "amber_subset_mask": amber_subset_mask,
            "pocket_residues_idxs": pocket_residue_idxs,
            "res_subset_mask": res_subset_mask,
            "atom_subset_mask": atom_subset_mask,
            "pocket_center": pocket_center,
            "pocket_atom_mask": pocket_atom_mask,
        }
        return pocket_info

    def select_pocket_and_buffer(self, complex_graph, pocket_info):
        res_mask = pocket_info["res_subset_mask"]
        atom_mask = pocket_info["atom_subset_mask"]
        amber_subset_mask = pocket_info["amber_subset_mask"]

        pocket_atom_mask = pocket_info["pocket_atom_mask"]

        complex_graph.amber_subset_mask = amber_subset_mask

        complex_graph["atom"].atom_mask = atom_mask
        complex_graph["atom"].nearby_atoms = pocket_atom_mask.clone()

        # Update atom numbering
        atom_numbering_old = torch.arange(complex_graph["atom"].pos.size(0))
        atom_numbering_old = atom_numbering_old[atom_mask]
        atom_numbering_new = torch.arange(atom_mask.sum())
        atom_numbering_dict = dict(
            zip(atom_numbering_old.numpy(), atom_numbering_new.numpy())
        )

        residue_numbering_old = torch.arange(complex_graph["receptor"].x.size(0))
        residue_numbering_old = residue_numbering_old[res_mask]
        residue_numbering_new = torch.arange(res_mask.sum())
        residue_numbering_dict = dict(
            zip(residue_numbering_old.numpy(), residue_numbering_new.numpy())
        )

        # Update pocket + buffer residue attributes
        complex_graph["receptor"].x = complex_graph["receptor"].x[res_mask]
        complex_graph["receptor"].pos = complex_graph["receptor"].pos[res_mask]
        complex_graph["receptor"].lens_receptors = complex_graph[
            "receptor"
        ].lens_receptors[res_mask]

        # Update pocket + buffer atom attributes
        complex_graph["atom"].x = complex_graph["atom"].x[atom_mask]
        complex_graph["atom"].vdw_radii = complex_graph["atom"].vdw_radii[atom_mask]

        complex_graph["atom"].pos = complex_graph["atom"].pos[atom_mask]
        complex_graph["atom"].orig_apo_pos = complex_graph["atom"].orig_apo_pos[
            atom_mask
        ]
        complex_graph["atom"].orig_aligned_apo_pos = complex_graph[
            "atom"
        ].orig_aligned_apo_pos[atom_mask]

        complex_graph["atom"].ca_mask = complex_graph["atom"].ca_mask[atom_mask]
        complex_graph["atom"].c_mask = complex_graph["atom"].c_mask[atom_mask]
        complex_graph["atom"].n_mask = complex_graph["atom"].n_mask[atom_mask]
        complex_graph["atom"].nearby_atoms = complex_graph["atom"].nearby_atoms[
            atom_mask
        ]

        # Gather edges between atoms in pocket + buffer
        atom_edge_index = complex_graph["atom", "atom_bond", "atom"].edge_index
        edges_in_subset = atom_mask[atom_edge_index[0]] & atom_mask[atom_edge_index[1]]

        # Create new edge numbering (used in fragment_index)
        edges_order_old = torch.arange(atom_edge_index.size(1))
        edges_order_old = edges_order_old[edges_in_subset]
        edges_order_new = torch.arange(edges_order_old.size(0))
        edge_numbering_dict = dict(
            zip(edges_order_old.numpy(), edges_order_new.numpy())
        )

        # Which edge rotates which atoms in topological sorted order
        atom_fragment_index = complex_graph["atom_bond", "atom"].atom_fragment_index
        fragment_old_edge_order, fragment_old_atom_idx = atom_fragment_index
        fragment_edge_subset_mask = edges_in_subset[fragment_old_edge_order]
        fragment_atom_subset_mask = atom_mask[fragment_old_atom_idx]

        # Gather edges in pocket and renumber them
        fragment_edge_subset = fragment_old_edge_order[fragment_edge_subset_mask]
        fragment_edge_subset.apply_(lambda x: edge_numbering_dict[x])

        # Gather atoms in pocket and renumber them
        fragment_atom_idx_subset = fragment_old_atom_idx[fragment_atom_subset_mask]
        fragment_atom_idx_subset.apply_(lambda x: atom_numbering_dict[x])

        # Update to new fragment index
        atom_fragment_index_subset = torch.stack(
            [fragment_edge_subset, fragment_atom_idx_subset], dim=0
        )
        complex_graph[
            "atom_bond", "atom"
        ].atom_fragment_index = atom_fragment_index_subset

        # Update receptor edge index
        atom_idx, res_idx = complex_graph[
            "atom", "atom_rec_contact", "receptor"
        ].edge_index
        atoms_subset = atom_idx[atom_mask]
        atom_res_idx_subset = res_idx[atom_mask]

        atoms_subset.apply_(lambda x: atom_numbering_dict[x])
        atom_res_idx_subset.apply_(lambda x: residue_numbering_dict[x])
        complex_graph["atom", "atom_rec_contact", "receptor"].edge_index = torch.stack(
            [atoms_subset, atom_res_idx_subset], dim=0
        )

        # Update edge index and edge mask
        complex_graph["atom", "atom_bond", "atom"].edge_index = atom_edge_index[
            :, edges_in_subset
        ]
        complex_graph["atom", "atom_bond", "atom"].edge_index.apply_(
            lambda x: atom_numbering_dict[x]
        )
        complex_graph["atom", "atom_bond", "atom"].edge_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].edge_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].squeeze_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].squeeze_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].ring_sub_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].ring_sub_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].ring_flip_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].ring_flip_mask[edges_in_subset]

        res_ids_rotatable = complex_graph["atom", "atom_bond", "atom"].res_to_rotate[
            :, 0
        ]
        res_ids_rotatable_subset = res_ids_rotatable[res_mask[res_ids_rotatable]]
        res_ids_rotatable_subset.apply_(lambda x: residue_numbering_dict[x])

        complex_graph["atom", "atom_bond", "atom"].res_to_rotate = torch.stack(
            [res_ids_rotatable_subset, torch.arange(len(res_ids_rotatable_subset))],
            dim=1,
        )
        return complex_graph

    def select_nearby_atoms(self, complex_graph):
        nearby_atom_mask = complex_graph["atom"].nearby_atoms
        # This presently captures only sidechains
        atom_edge_index = complex_graph["atom", "atom_bond", "atom"].edge_index
        nearby_atom_mask_edges = (
            nearby_atom_mask[atom_edge_index[0]] & nearby_atom_mask[atom_edge_index[1]]
        )

        # Update rotatable mask to only edges composed of nearby atoms
        complex_graph["atom", "atom_bond", "atom"].edge_mask[
            ~nearby_atom_mask_edges
        ] = False
        return complex_graph

    def len(self):
        if self.limit_complexes is not None:
            return self.limit_complexes
        return self.input_df.shape[0]


class InferenceDataModule(LightningDataModule):
    def __init__(
        self,
        input_csv,
        featurizer_cfg: FeaturizerConfig,
        limit_complexes: int = None,
        esm_embeddings_path: str = None,
        pocket_reduction: bool = False,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        only_nearby_residues_atomic: bool = False,
    ):
        super().__init__()
        self.input_csv = input_csv
        self.featurizer_cfg = featurizer_cfg
        self.limit_complexes = limit_complexes

        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.esm_embeddings_path = esm_embeddings_path

    def predict_dataloader(self):
        dataset = PredictionDataset(
            input_csv=self.input_csv,
            featurizer_cfg=self.featurizer_cfg,
            pocket_buffer=self.pocket_buffer,
            pocket_reduction=self.pocket_reduction,
            pocket_min_size=self.pocket_min_size,
            limit_complexes=self.limit_complexes,
            only_nearby_residues_atomic=self.only_nearby_residues_atomic,
            esm_embeddings_path=self.esm_embeddings_path,
        )
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False)
