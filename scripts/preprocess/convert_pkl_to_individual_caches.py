import os
import pickle
import torch
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl_dir", help="Directory where .pkl cache is stored")
    parser.add_argument("--require_ligand", action="store_true", default=True)
    parser.add_argument(
        "--limit_complexes", type=int, default=None, help="Number of complexes to use."
    )
    parser.add_argument("--save_dir", help="Directory to save files under")
    parser.add_argument(
        "--use_esmflow_update",
        action="store_true",
        help="Whether to use the pipeline corresponding to ESMFlow",
    )

    return parser.parse_args()


def convert_files(args):
    if args.use_esmflow_update:
        raise NotImplementedError(
            "Currently conversion from pkl files to individual cache not supported for esmflow update."
        )

    heterograph_file = f"{args.pkl_dir}/heterographs.pkl"
    with open(heterograph_file, "rb") as f:
        complex_graphs = pickle.load(f)

    if args.require_ligand:
        ligand_file = f"{args.pkl_dir}/rdkit_ligands.pkl"
        with open(ligand_file, "rb") as f:
            rdkit_ligands = pickle.load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    names = []

    if args.limit_complexes is not None and args.limit_complexes >= 1:
        complex_graphs = complex_graphs[: args.limit_complexes]

    for idx, complex_graph in tqdm(enumerate(complex_graphs)):
        name = complex_graph["name"]

        try:
            filename = f"{args.save_dir}/heterograph-{name}-0.pt"
            torch.save(complex_graph, filename)

            if args.require_ligand:
                ligand = rdkit_ligands[idx]
                lig_file = f"{args.save_dir}/rdkit_ligand-{name}-0.pkl"
                with open(lig_file, "wb") as f:
                    pickle.dump((ligand), f)
            names.append(name)

        except Exception as e:
            print(f"Complex={name}: Could not save because of {e}")

    with open(f"{args.save_dir}/complex_names.pkl", "wb") as f:
        pickle.dump(names, f)


def main():
    args = parse_args()
    convert_files(args)


if __name__ == "__main__":
    main()
