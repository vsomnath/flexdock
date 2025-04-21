import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--esm_embeddings_path", type=str, default="data/embeddings_output", help=""
    )
    parser.add_argument(
        "--output_path", type=str, default="data/esm2_3billion_embeddings.pt", help=""
    )
    parser.add_argument(
        "--cache_individual",
        action="store_true",
        help="Whether to save individual embeddings to disk",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.cache_individual:
        os.makedirs(args.output_path, exist_ok=True)
        filenames = tqdm(os.listdir(args.esm_embeddings_path))
        pdbs = defaultdict(list)

        for filename in filenames:
            pdb = filename.split("_")[0]
            pdbs[pdb].append(filename)

        for pdb in pdbs:
            files_for_pdb = pdbs[pdb]
            file_dict = {
                int(filename.split(".")[0].split("_")[2]): torch.load(
                    os.path.join(args.esm_embeddings_path, filename)
                )["representations"][33]
                for filename in files_for_pdb
            }
            output_filename = f"{args.output_path}/{pdb}.pt"
            print(f"{output_filename}")
            torch.save(file_dict, output_filename)

    else:
        dict = {}
        for filename in tqdm(os.listdir(args.esm_embeddings_path)):
            dict[filename.split(".")[0]] = torch.load(
                os.path.join(args.esm_embeddings_path, filename)
            )["representations"][33]
        torch.save(dict, args.output_path)


if __name__ == "__main__":
    main()
