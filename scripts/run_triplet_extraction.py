from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        "-m",
        "src.extraction.triplet_extractor",
        "--chunks_path",
        str(project_root / "data" / "processed" / "chunks.parquet"),
        "--output_path",
        str(project_root / "data" / "processed" / "triplets.parquet"),
        "--llm_model",
        "gpt-4o-mini",
        "--llm_cache_path",
        str(project_root / "data" / "cache" / "llm_responses.jsonl"),
        "--mode",
        "all",
        "--save_every",
        "10",
        *sys.argv[1:],
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
