import copy

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution


def scRMSD(atom_ids, atoms1, atoms2):
    if len(atom_ids) == 0:
        return 0.0
    # fix the alignment, so that the indices match
    try:
        coords_1 = atoms1[atom_ids]
        coords_2 = atoms2[atom_ids]
        return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / len(coords_1))
    except Exception as e:
        print("Cannot calculate RMSD. Maybe atoms1, and atoms2 do not mach?")
        print("atom_ids:", atom_ids)
        print("atoms1:", atoms1.shape)
        print("atoms2:", atoms2.shape)
        print(e)
        raise e


class SidechainOptimizer:
    def __init__(
        self,
        apo_pos: np.array,
        holo_pos: np.array,
        rotatable_bonds: torch.Tensor,
        current_subcomponents: list[torch.Tensor],
        ligand: torch.Tensor,
        seed=None,
    ):
        super().__init__()
        if seed:
            np.random.seed(seed)
        self.rotatable_bonds = rotatable_bonds
        self.apo_pos = copy.deepcopy(apo_pos)
        self.holo_pos = holo_pos
        # Convert to numpy for applying rotations
        self.current_subcomponents = [
            c.cpu().numpy() if isinstance(c, torch.Tensor) else c
            for c in current_subcomponents
        ]
        self.ligand = (
            ligand.cpu().numpy() if isinstance(ligand, torch.Tensor) else ligand
        )
        # These are the atoms which get a new position
        self.modified_atoms = np.unique(
            np.concatenate(self.current_subcomponents).ravel()
        )
        # Store the last calculated RMSD so that we can check if the optimization improved the score
        self.last_rmsd = None
        mask = np.ones(self.holo_pos.shape[0], dtype=bool)
        mask[self.modified_atoms] = False
        self.mask = mask

    def closest_pos(self, sc_pos, rest_pos):
        return np.min(
            np.linalg.norm(sc_pos[None, :, :] - rest_pos[:, None, :], axis=-1), axis=0
        )

    def penalty_with_weighted_exp_all_rmsd(self, values):
        new_pos = self.apply_rotations(values)
        ligand_pos = np.row_stack((new_pos.copy(), self.ligand))

        # mask now includes all non-modified atoms and the ligand positions
        mask = np.append(self.mask.copy(), np.ones(self.ligand.shape[0], dtype=bool))

        distance = np.linalg.norm(
            ligand_pos[None, mask, :] - new_pos[self.modified_atoms, None, :], axis=-1
        )
        weight = np.exp(-distance)
        distance_sum = np.sum(np.multiply(distance, weight), axis=1)
        weight_sum = np.sum(weight, axis=1)
        weight_all = np.multiply(
            weight_sum * (1 / np.sum(weight_sum)), np.sqrt(distance_sum)
        )
        self.last_rmsd = scRMSD(self.modified_atoms, new_pos, self.holo_pos)

        return (self.last_rmsd / np.sqrt(np.sum(weight_all))) * np.sqrt(
            np.sum(distance_sum)
        )

    def penalty_with_nearest_rmsd(self, values):
        new_pos = self.apply_rotations(values)
        new_atoms = new_pos[self.modified_atoms]

        closest_pair = self.closest_pos(new_atoms, new_pos[self.mask])

        np.row_stack((closest_pair, self.closest_pos(new_atoms, self.ligand)))

        closeness_rmsd = np.sqrt(np.mean(closest_pair))  # TODO use RMSD function?

        self.last_rmsd = scRMSD(self.modified_atoms, new_pos, self.holo_pos)

        return self.last_rmsd - closeness_rmsd

    def score_conformation(self, values):
        # 1. Apply rotations to the current sidechain
        # Note that indices in current_subcomponent are based on the whole protein (so self.rec_atoms)
        # The same holds for self.rotatable_bonds
        new_pos = self.apply_rotations(values)

        # 2. Calculate the RMSD between the current sidechain and the true sidechain
        self.last_rmsd = scRMSD(self.modified_atoms, new_pos, self.holo_pos)
        return self.last_rmsd

    def apply_rotations(self, values):
        # cannot use modify sidechain function for now, as this does only work for complex graphs

        pos = self.apo_pos.copy()

        for torsion_update, rot_bond, subcomponent in zip(
            values, self.rotatable_bonds, self.current_subcomponents
        ):
            if torsion_update != 0:
                u, v = rot_bond
                mask_rotate = subcomponent
                # get atom positions of current subcomponent
                try:
                    rot_vec = (
                        pos[u] - pos[v]
                    )  # convention: positive rotation if pointing inwards

                    rot_vec = (
                        rot_vec * torsion_update / np.linalg.norm(rot_vec)
                    )  # idx_edge!
                    rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()

                    # Note the rot_mat.T is important, since sidechain angles must be updated with the same convention
                    pos[mask_rotate] = (pos[mask_rotate] - pos[v]) @ rot_mat.T + pos[v]
                except Exception as e:
                    print("Skipping sidechain update because of the error:")
                    print(e)

        return pos


def sidechain_conformer_matching(
    apo_pos,
    holo_pos,
    edge_index,
    mask_rotate,
    atom_rec_index,
    fragment_index,
    res_to_rotate,
    ligand,
    score="dist",
    seed=0,
    popsize=15,
    maxiter=100,
    mutation=(0.5, 1),
    recombination=0.7,
):
    apo_pos = apo_pos.numpy()
    holo_pos = holo_pos.numpy()
    edge_index = edge_index.numpy()
    mask_rotate = mask_rotate.numpy()
    atom_rec_index = atom_rec_index.numpy()
    fragment_index = fragment_index.numpy()
    res_to_rotate = res_to_rotate.numpy()

    apo_pos_orig = copy.deepcopy(apo_pos)
    filterSCHs = []

    complete_rmsd_start = scRMSD(list(range(len(apo_pos))), apo_pos, holo_pos)

    improvements = 0
    optimal_rotations = []
    for res_idx in range(atom_rec_index.max() + 1):
        # These are all the rotable edges for the corresponding residue
        residue_edge_mask = np.zeros((mask_rotate.shape[0],), dtype=np.bool_)
        rotatable_bond_idxs = res_to_rotate[:, 1][
            np.flatnonzero(res_to_rotate[:, 0] == res_idx)
        ]
        residue_edge_mask[rotatable_bond_idxs] = True

        mask_to_use = residue_edge_mask & mask_rotate
        rotatable_bonds = edge_index[:, mask_to_use].T

        if len(rotatable_bonds) == 0:
            continue

        max_bound = [np.pi] * len(rotatable_bonds)
        min_bound = [-np.pi] * len(rotatable_bonds)
        bounds = list(zip(min_bound, max_bound))

        indices_to_check = np.arange(edge_index.shape[1])[mask_to_use]
        subcomponents = [
            fragment_index[1][np.flatnonzero(fragment_index[0] == index)]
            for index in indices_to_check
        ]

        opt = SidechainOptimizer(
            apo_pos, holo_pos, rotatable_bonds, subcomponents, ligand, seed=seed
        )
        if score == "dist":
            scoring = opt.score_conformation
        elif score == "nearest":
            scoring = opt.penalty_with_nearest_rmsd
        elif score == "exp":
            scoring = opt.penalty_with_weighted_exp_all_rmsd

        ## Optimize conformations
        result = differential_evolution(
            scoring,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            disp=False,
            seed=seed,
        )

        optimal_rotations.extend(result["x"])
        filterSCHs.extend(list(opt.modified_atoms))

        before = scRMSD(opt.modified_atoms, apo_pos, holo_pos)

        if before <= opt.last_rmsd:
            pass  # print("No improvement possible for this sidechain. Not applying any rotations.")
        else:
            # Apply and store the optimal rotations
            apo_pos = opt.apply_rotations(result["x"])
            after = scRMSD(opt.modified_atoms, apo_pos, holo_pos)
            improvements += before - after

    complete_rmsd_end = scRMSD(list(range(len(apo_pos))), apo_pos, holo_pos)
    assert (
        complete_rmsd_end <= complete_rmsd_start
    ), "RMSD should not increase after conformer matching."

    return (
        apo_pos,
        optimal_rotations,
        scRMSD(list(set(filterSCHs)), apo_pos_orig, holo_pos)
        - scRMSD(list(set(filterSCHs)), apo_pos, holo_pos),
    )
