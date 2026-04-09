from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    entities_path = project_root / "data" / "processed" / "entities.parquet"
    triplets_path = project_root / "data" / "processed" / "triplets.parquet"

    cmd = [
        sys.executable,
        "-m",
        "src.extraction.competitor_analysis",
        "--chunks_path",
        str(project_root / "data" / "processed" / "chunks.parquet"),
        "--entities_path",
        str(entities_path),
        "--triplets_path",
        str(triplets_path),
        "--output_dir",
        str(project_root / "outputs" / "analysis"),
        *sys.argv[1:],
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
