import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Wrapper around the security evaluation pipeline."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["rag", "agent", "both"],
        help="Which answer mode to benchmark.",
    )
    parser.add_argument(
        "--cases_path",
        type=str,
        default=None,
        help="Optional security benchmark cases file override.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    args, remaining_args = parser.parse_known_args()

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.security_eval",
        "--mode",
        args.mode,
    ]
    if args.cases_path:
        cmd.extend(["--cases_path", str(Path(args.cases_path))])
    if args.output_dir:
        cmd.extend(["--output_dir", str(Path(args.output_dir))])
    cmd.extend(remaining_args)

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
