import dataclasses
import copy
import logging

import numpy as np
import torch

from flexdock.data.modules import ComplexData

from flexdock.data.feature.molecule import (
    get_lig_graph_with_matching,
    get_posebusters_edge_index,
)
from flexdock.data.conformers.protein import sidechain_conformer_matching
from flexdock.data.feature.protein import (
    get_fullrec_graph,
    construct_protein_edge_index,
)
from flexdock.data.feature.helpers import align_sidechains_to_backbone_torch
from flexdock.geometry.ops import rigid_transform_kabsch


@dataclasses.dataclass
class FeaturizerConfig:
    # Ligand processing args
    matching: bool = True
    popsize: int = 15
    maxiter: int = 15
    keep_original: bool = False
    remove_hs: bool = False
    num_conformers: int = 1
    max_lig_size: int = None

    # Protein args
    flexible_backbone: bool = False
    flexible_sidechains: bool = False


class Featurizer:
    def __init__(self, cfg: FeaturizerConfig):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg=cfg)

    def featurize_complex(self, complex_inputs):
        if complex_inputs is None:
            return None

        # Try to featurize the ligand
        complex_graph = self.featurize_ligand(complex_inputs)
        if complex_graph is None:
            return None

        # Try to featurize the protein
        complex_graph = self.featurize_protein(complex_graph, complex_inputs)
        if complex_graph is None:
            return None

        complex_graph["success"] = True

        return {
            "complex_graph": complex_graph,
            "name": complex_inputs["name"],
            "ligand": complex_inputs["ligand"],
        }

    def featurize_ligand(self, complex_inputs):
        name = complex_inputs["name"]
        ligands = complex_inputs["ligand"]

        for lig_idx, lig in enumerate(ligands):
            if (
                self.cfg.max_lig_size is not None
                and lig.GetNumHeavyAtoms() > self.cfg.max_lig_size
            ):
                logging.info(
                    f"{name}: Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is "
                    f"larger than max_lig_size {self.cfg.max_lig_size}. Skipping"
                )
                continue

            try:
                complex_graph = ComplexData()
                complex_graph["name"] = name

                complex_graph.mol = copy.deepcopy(lig)

                get_lig_graph_with_matching(
                    lig,
                    complex_graph,
                    self.cfg.popsize,
                    self.cfg.maxiter,
                    self.cfg.matching,
                    self.cfg.keep_original,
                    self.cfg.num_conformers,
                    remove_hs=self.cfg.remove_hs,
                    new_version=True,
                )

            except Exception as e:
                logging.error(
                    f"Failed to featurize ligand at index {lig_idx} due to {e}"
                )
                complex_graph = None

        return complex_graph

    def featurize_protein(self, complex_graph, complex_inputs):
        apo_rec_struct = complex_inputs["apo_rec_struct"]
        holo_rec_struct = complex_inputs["holo_rec_struct"]
        lm_embedding_chains = complex_inputs["lm_embedding"]
        name = complex_inputs["name"]

        try:
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

            complex_graph["apo_rec_path"] = complex_inputs["apo_rec_path"]
            complex_graph["holo_rec_path"] = complex_inputs.get("holo_rec_path", None)

            if lm_embedding_chains is not None:
                lm_embeddings = torch.cat(lm_embedding_chains, dim=0)
            else:
                lm_embeddings = None

            # Get the atom level receptor graph for apo / holo structures
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

            complex_graph["atom", "atom_bond", "atom"].edge_index = torch.from_numpy(
                edge_index
            )
            complex_graph["atom", "atom_bond", "atom"].squeeze_mask = torch.from_numpy(
                squeeze_mask
            )
            complex_graph["atom", "atom_bond", "atom"].edge_mask = torch.from_numpy(
                edge_mask
            )
            complex_graph["atom", "atom_bond", "atom"].ring_sub_mask = torch.from_numpy(
                ring_sub_mask
            )
            complex_graph[
                "atom", "atom_bond", "atom"
            ].ring_flip_mask = torch.from_numpy(ring_flip_mask)
            complex_graph["atom_bond", "atom"].atom_fragment_index = torch.from_numpy(
                fragment_index
            )
            complex_graph["atom", "atom_bond", "atom"].res_to_rotate = torch.from_numpy(
                res_to_rotate
            )
            complex_graph[
                "atom_bond", "atom_angle", "atom_bond"
            ].angle_2_index = torch.from_numpy(angle_2_index)

            lens_receptors = np.asarray(
                [len([a for a in res.atoms]) for res in apo_rec_struct.residues]
            )
            complex_graph["receptor"].lens_receptors = torch.from_numpy(
                lens_receptors
            ).long()

            if holo_rec_struct is not None:
                if self.cfg.flexible_backbone or self.cfg.flexible_sidechains:
                    # Rigid body alignment of apo and holo structures
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
                    logging.info(
                        f"{name}: RMSD between aligned apo and holo full atomic structures is {rmsd}",
                    )

                    # Aligning backbone frames of apo and holo structures, and
                    # getting the corresponding rotation
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

                    rmsd = torch.sqrt(
                        torch.mean(
                            torch.sum((holo_rec_pos - aligned_apo_rec_pos) ** 2, dim=1)
                        )
                    )
                    logging.info(
                        f"{name}: RMSD between aligned apo and holo full atomic structures after sidechain alignment {rmsd.item()}",
                    )

                    # Sidechain conformer matching to get closest "holo" structure
                    # starting from local structures of apo
                    (
                        conformer_matched_apo_rec_pos,
                        sc_conformer_match_rotations,
                        sc_conformer_match_improvements,
                    ) = sidechain_conformer_matching(
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
                    logging.info(
                        f"{name}: RMSD between aligned apo and holo full atomic structures after sidechain conformer matching {rmsd.item()}",
                    )

        except Exception as e:
            logging.error(
                f"{name}: Failed to featurize protein due to {e}", exc_info=True
            )
            return None

        # Add edge index for posebusters, used in relaxation module
        complex_graph = get_posebusters_edge_index(complex_graph)
        return complex_graph
