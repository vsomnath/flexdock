import os
from argparse import ArgumentParser

from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from flexdock.data.constants import AVAILABLE_DATASETS, restype_3to1

biopython_parser = PDBParser()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data/PDBBIND_atomCorrected/")
    parser.add_argument("--out_file", type=str, default="data/prepared_for_esm.fasta")
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=None,
        help="List of pdb ids for which esm embeddings will be generated",
    )
    parser.add_argument(
        "--max_complexes", type=int, default=None, help="Maximum number of examples"
    )
    parser.add_argument(
        "--protein_ligand_csv",
        type=str,
        default="data/protein_ligand_example_csv.csv",
        help="Path to a .csv specifying the input as described in the main README",
    )
    args = parser.parse_args()

    return args


def get_sequence_from_file(file_path):
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequences = []
    for i, chain in enumerate(structure):
        seq = ""
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue

            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha is not None and n is not None and c is not None
            ):  # only append residue if it is an amino acid
                try:
                    seq += restype_3to1[residue.get_resname()]
                except Exception:
                    seq += "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex ",
                        file_path,
                        ". Replacing it with a dash - .",
                    )
        sequences.append(seq)
    return sequences


def prepare_files_for_embedding(args):
    data_dir = args.data_dir

    if args.dataset not in AVAILABLE_DATASETS:
        if args.protein_ligand_csv is None:
            print(
                f"Provided dataset {args.dataset} not in {AVAILABLE_DATASETS}"
                "and protein_ligand_csv is None. Please supply a valid file"
            )
            return

    else:
        sequences = []
        ids = []

        if args.dataset == "pdbbind":
            if args.ids is None:
                names = [
                    name for name in os.listdir(args.data_dir) if name != ".DS_Store"
                ]
            else:
                names = args.ids

            file_paths = [
                os.path.join(data_dir, name, f"{name}_protein_processed_fix.pdb")
                for name in names
            ]

    if args.max_complexes is not None:
        file_paths = file_paths[: args.max_complexes]
        names = names[: args.max_complexes]

    for file_path, name in zip(file_paths, names):
        try:
            l = get_sequence_from_file(file_path)

            for i, seq in enumerate(l):
                sequences.append(seq)
                ids.append(f"{name}_chain_{i}")
        except Exception as e:
            print(f"Failed to process {name} due to {e}")

    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)

    dirname = os.path.dirname(args.out_file)
    os.makedirs(dirname, exist_ok=True)
    SeqIO.write(records, args.out_file, "fasta")


if __name__ == "__main__":
    args = parse_args()
    prepare_files_for_embedding(args)
