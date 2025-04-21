import os
import argparse
import torch
import esm
import pandas as pd

device = "cpu" if not torch.cuda.is_available() else "cuda"


def load_esmfold_model():
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_file", type=str, help="csv file with sequences")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save pdb strutures to"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Chunk size used for ESMFold"
    )

    args = parser.parse_args()
    return args


def run_esmfold():
    args = parse_args()

    df = pd.read_csv(args.csv_file)
    pdb_ids = df["name"].values.tolist()
    sequences = df["seqres"].values.tolist()

    print(f"Generating ESMFold outputs for {len(sequences)} sequences")

    model = load_esmfold_model()
    model.set_chunk_size(args.chunk_size)
    chunk_size = args.chunk_size

    for pdb_id, sequence in zip(pdb_ids, sequences):
        if os.path.exists(f"{args.output_dir}/{pdb_id}/{pdb_id}_esmfold.pdb"):
            print(f"Complex={pdb_id} already has ESMFold file. Skipping it.")
            continue

        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            os.makedirs(f"{args.output_dir}/{pdb_id}/", exist_ok=True)
            with open(f"{args.output_dir}/{pdb_id}/{pdb_id}_esmfold.pdb", "w") as f:
                f.write(output)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory on chunk_size", chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    continue
            else:
                print(f"Could not process {pdb_id} due to {e}")
                continue


if __name__ == "__main__":
    run_esmfold()
