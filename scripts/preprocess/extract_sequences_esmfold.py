import os
import argparse
from collections import defaultdict
from functools import partial

import prody as pr
import pandas as pd


def get_sequences_from_pdbfile(file_path, parser=None):
    parsed_protein = pr.parsePDB(file_path)

    sequence = None
    for i, chain in enumerate(parsed_protein.iterChains()):
        if chain.ca is not None:
            if sequence is None:
                sequence = chain.ca.getSequence()
            else:
                sequence += ":" + chain.ca.getSequence()
    return sequence


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="pdbbind", help="Dataset")
    parser.add_argument("--file_identifier", type=str)
    parser.add_argument("--max_complexes", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--quiet_parse", action="store_true")

    parser.add_argument("--use_prody", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.use_prody:
        prot_parser = None
    else:
        from Bio.PDB import PDBParser

        prot_parser = PDBParser(QUIET=args.quiet_parse)

    sequence_fn = partial(get_sequences_from_pdbfile, parser=prot_parser)

    pdb_ids = os.listdir(args.data_dir)
    if args.max_complexes is not None:
        pdb_ids = pdb_ids[: args.max_complexes]

    sharded_idxs = list(range(len(pdb_ids) // args.shard_size + 1))
    print(sharded_idxs)

    for shard_idx in sharded_idxs:
        shard_ids = pdb_ids[
            args.shard_size * shard_idx : args.shard_size * (shard_idx + 1)
        ]
        pdbid_seqs = defaultdict(list)

        for pdb_id in shard_ids:
            filename = f"{args.data_dir}/{pdb_id}/{pdb_id}_{args.file_identifier}"
            if os.path.exists(filename):
                sequence = sequence_fn(filename)
                pdbid_seqs["name"].append(pdb_id)
                pdbid_seqs["seqres"].append(sequence)

        print(f"Saving shard {shard_idx} to {args.dataset}_{shard_idx}.csv")
        df = pd.DataFrame.from_dict(pdbid_seqs)
        df.to_csv(f"{args.dataset}_{shard_idx}.csv", index=False)


if __name__ == "__main__":
    main()
