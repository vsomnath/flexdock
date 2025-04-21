from tempfile import NamedTemporaryFile

import parmed as pmd

try:
    import pytraj as pt
except:
    pass

from flexdock.data.constants import ATOM_ORDER_DICT


def parse_pdb_from_path(path, remove_hs=True, reorder=True, as_pytraj=False):
    struct = pmd.load_file(path)
    if remove_hs:
        struct.strip("@/H")
    if reorder:
        struct.atoms.sort(
            key=lambda atom: (
                atom.residue.idx,
                float("inf")
                if atom.name == "OXT"
                else ATOM_ORDER_DICT[atom.residue.name].index(atom.name),
            )
        )
        struct.atoms.index_members()
        for res in struct.residues:
            res.atoms.sort(
                key=lambda atom: float("inf")
                if atom.name == "OXT"
                else ATOM_ORDER_DICT[res.name].index(atom.name)
            )
    if as_pytraj:
        with NamedTemporaryFile(suffix=".pdb") as file:
            struct.save(file.name, overwrite=True)
            return pt.load(file.name)
    return struct


def reorder_struct(struct):
    reorder = [
        (
            res.idx,
            float("inf")
            if atom.name == "OXT"
            else ATOM_ORDER_DICT[res.name].index(atom.name),
        )
        for res in struct.residues
        for atom in res.atoms
    ]
    for res in struct.residues:
        res.atoms.sort(key=lambda atom: reorder[atom.idx])
