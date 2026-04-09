import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing.chunking",
        "--input_path",
        str(project_root / "data" / "processed" / "processed_pages.parquet"),
        "--output_path",
        str(project_root / "data" / "processed" / "chunks.parquet"),
        "--method",
        "token",
        "--chunk_size",
        "500",
        "--overlap",
        "75",
        *sys.argv[1:],
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
