import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Wrapper around the metrics pipeline with separate classical_ml/naive/improved outputs."
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="improved",
        choices=["classical_ml", "naive", "improved"],
        help="Retrieval mode whose judged outputs should be analyzed.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional judged input override.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional analysis directory override.",
    )
    args, remaining_args = parser.parse_known_args()

    input_path = (
        Path(args.input_path)
        if args.input_path
        else (
            project_root
            / "outputs"
            / "evaluation"
            / "local_smoke"
            / args.retrieval_mode
            / "evaluation_runs_judged.parquet"
        )
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            project_root
            / "outputs"
            / "evaluation"
            / "local_smoke"
            / args.retrieval_mode
            / "analysis"
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.metrics",
        "--input_path",
        str(input_path),
        "--output_dir",
        str(output_dir),
        "--verbose",
        *remaining_args,
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
