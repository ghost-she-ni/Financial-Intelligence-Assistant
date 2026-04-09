import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        "-m",
        "src.ingestion.collect_reports",
        "--input_dir",
        str(project_root / "data" / "raw_pdfs"),
        "--output_path",
        str(project_root / "data" / "metadata" / "documents_metadata.parquet"),
        *sys.argv[1:],
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
