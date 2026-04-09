import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Wrapper around the evaluation pipeline with separate classical_ml/naive/improved outputs."
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="improved",
        choices=["classical_ml", "naive", "improved"],
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=None,
        help="Optional questions file override.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output parquet override.",
    )
    args, remaining_args = parser.parse_known_args()

    questions_path = (
        Path(args.questions_path)
        if args.questions_path
        else (
            project_root
            / "data"
            / "evaluation"
            / "financebench"
            / "financebench_subset_local_smoke.parquet"
        )
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else (
            project_root
            / "outputs"
            / "evaluation"
            / "local_smoke"
            / args.retrieval_mode
            / "evaluation_runs.parquet"
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluation_pipeline",
        "--questions_path",
        str(questions_path),
        "--output_path",
        str(output_path),
        "--retrieval_mode",
        args.retrieval_mode,
        "--verbose",
        *remaining_args,
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
