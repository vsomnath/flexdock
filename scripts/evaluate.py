import argparse
import logging
import wandb

from flexdock.metrics.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="pdbbind",
        choices=["pdbbind", "posebusters", "moad"],
    )
    parser.add_argument("--output_dir", type=str, default="inf_results/dummy_test")

    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--project", type=str)
    parser.add_argument("--entity", type=str)

    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--task", default="docking", type=str)
    parser.add_argument("--only_relaxation", action="store_true", default=False)
    parser.add_argument("--use_symmetry_correction", default=False, action="store_true")
    parser.add_argument(
        "--only_nearby_residues_atomic", default=False, action="store_true"
    )
    parser.add_argument("--align_proteins_by", default="nearby_atoms")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = args.output_dir + f"_align{args.align_proteins_by}"

    if args.wandb:
        wandb.init(
            entity=args.entity,
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args,
        )

    evaluator = Evaluator(args=args)
    evaluator.evaluate(input_csv=args.input_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    main()
