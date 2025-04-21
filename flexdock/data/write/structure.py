import copy
import dataclasses


import numpy as np

from flexdock.data import constants


@dataclasses.dataclass
class Protein:
    # Chain identifiers for all residues in the protein
    chain_ids: np.ndarray  # [num_res_in_struct]

    # Atom coordinates (in angstroms)
    atom_positions: np.ndarray  # [num_res_in_struct, num_atom_type, 3]

    # Binary mask indicating presence (1.0) or absence of an atom (1.0)
    # Used in feature computation
    atom_mask: np.ndarray  # [num_res_in_struct]

    # Residue names
    residue_types: np.ndarray  # [num_res_in_struct]

    # Residue index
    residue_index: np.ndarray

    in_pocket: np.ndarray = None

    nearby_atoms: np.ndarray = None


def biopython_to_protein(structure, in_pocket=None):
    chain_ids = []
    atom_coords = []
    atom_mask = []
    residue_types = []
    residue_index = []

    res_count = 0

    for chain in structure:
        for idx, residue in enumerate(chain):
            chain_ids.append(chain.id)
            resname = residue.get_resname()
            residue_atom_pos = [[0, 0, 0]] * constants.max_atoms
            residue_atom_mask = [0] * constants.max_atoms

            for atom in residue:
                if atom.element == "H":
                    continue

                # This masking is currently dictated by ATOM_ORDER_DICT since we use this for ESMFold processing
                if atom.name in constants.ATOM_ORDER_DICT[resname]:
                    atom_idx = constants.ATOM_ORDER_DICT[resname].index(atom.name)
                    residue_atom_mask[atom_idx] = 1
                    residue_atom_pos[atom_idx] = list(atom.get_vector())

            atom_coords.append(residue_atom_pos)
            atom_mask.append(residue_atom_mask)
            residue_types.append(residue.get_resname())
            residue_index.append(res_count)
            res_count += 1

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n + 1 for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids], dtype=np.int32)

    return Protein(
        chain_ids=chain_index,
        atom_positions=np.array(atom_coords, dtype=np.float32),
        atom_mask=np.asarray(atom_mask, dtype="bool"),
        residue_types=np.array(residue_types),
        residue_index=np.array(residue_index, dtype=np.int32),
        in_pocket=in_pocket,
    )


def pyg_to_protein(pyg_data, protein=None):
    """
    Convert pyg_data to Protein object. If pyg_data corresponds to a pocket,
    the pocket positions are inpainted into a provided Protein.
    """
    # We assume the input pyg_data is a HeteroData object
    atom_coords_pocket = []
    atom_mask_pocket = []
    residue_types_pocket = []

    n_residues = pyg_data["receptor"].x.shape[0]
    atom_to_residue = (
        pyg_data["atom", "atom_rec_contact", "receptor"]["edge_index"][1]
        .detach()
        .cpu()
        .numpy()
    )
    atom_feats = pyg_data["atom"].x.detach().cpu().numpy()

    atom_pos = pyg_data["atom"].pos.detach().cpu().numpy() + pyg_data.original_center[0]
    res_feats = pyg_data["receptor"].x.long().detach().cpu().numpy()

    for residue_idx in range(n_residues):
        residue_type = constants.allowable_features["possible_amino_acids"][
            res_feats[residue_idx][0]
        ]
        residue_types_pocket.append(residue_type)

        residue_atom_mask = [0] * constants.max_atoms
        residue_atom_pos = [[0, 0, 0]] * constants.max_atoms

        atom_members = np.flatnonzero(atom_to_residue == residue_idx)

        for atom_idx in atom_members:
            atom_name = constants.allowable_features["possible_atom_type_3"][
                atom_feats[atom_idx, -1]
            ]

            if atom_name in constants.ATOM_ORDER_DICT[residue_type]:
                atom_index_in_residue = constants.ATOM_ORDER_DICT[residue_type].index(
                    atom_name
                )
                residue_atom_mask[atom_index_in_residue] = 1
                residue_atom_pos[atom_index_in_residue] = atom_pos[atom_idx].tolist()

        atom_coords_pocket.append(residue_atom_pos)
        atom_mask_pocket.append(residue_atom_mask)

    if protein is None:
        return Protein(
            chain_ids=np.zeros((n_residues,), dtype=np.int32),
            atom_positions=np.array(atom_coords_pocket, dtype=np.float32),
            atom_mask=np.array(atom_mask_pocket, dtype=np.bool_),
            residue_types=np.array(residue_types_pocket),
            residue_index=np.arange(n_residues, dtype=np.int32),
        )

    in_pocket = protein.in_pocket
    atom_positions = copy.deepcopy(protein.atom_positions)
    atom_mask = copy.deepcopy(protein.atom_mask)
    residue_types = protein.residue_types

    if in_pocket is not None:
        atom_mask[in_pocket] = np.array(atom_mask_pocket)
        atom_positions[in_pocket] = np.array(atom_coords_pocket)
        residue_types[in_pocket] = residue_types_pocket

    return Protein(
        chain_ids=protein.chain_ids,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_types=residue_types,
        residue_index=protein.residue_index,
        in_pocket=in_pocket,
    )


def protein_to_pdb(protein: Protein, filename) -> list[str]:
    """Converts a given structure to a PDB string.

    Args:
     structure: The protein/chain to convert to PDB.

    Returns:
     PDB string.
    """
    #   res_1to3 = lambda r: constants.restype_1to3.get(restypes[r], 'UNK')
    pdb_lines = []

    atom_mask = protein.atom_mask
    aatype = protein.residue_types.tolist()
    aatype_int = [
        constants.restype_order_3.get(elem, constants.restype_num) for elem in aatype
    ]
    aatype_int = np.array(aatype_int).astype(np.int32)

    atom_positions = protein.atom_positions
    # (Old version) We don't store residue index so assumed a continuous order for now!
    # (Old version) residue_index = np.arange(len(aatype)).astype(np.int32) + 1
    residue_index = protein.residue_index.astype(np.int32) + 1
    chain_index = protein.chain_ids.astype(np.int32)
    # Beta factors are not stored, so zeroed out.
    b_factors = np.zeros(protein.atom_mask.shape)

    if np.any(aatype_int > constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= constants.PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {constants.PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = constants.PDB_CHAIN_IDS[i - 1]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]

    # Add all atom sites.
    for i in range(aatype_int.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    aatype[i - 1],
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = aatype[i]
        atom_types = constants.ATOM_ORDER_DICT[res_name_3]
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index, aatype[-1], chain_ids[chain_index[-1]], residue_index[-1]
        )
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    pdb_lines = "\n".join(pdb_lines) + "\n"

    with open(filename, "w") as f:
        f.write(pdb_lines)


def _chain_end(
    atom_index: int, end_resname: str, chain_name: str, residue_index: int
) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def pyg_to_pdb(pyg_data, filename: str, protein=None):
    protein_to_pdb(pyg_to_protein(pyg_data, protein=protein), filename)
