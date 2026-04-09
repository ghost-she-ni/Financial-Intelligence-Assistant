import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        "-m",
        "src.embeddings.embed_chunks",
        "--chunks_path",
        str(project_root / "data" / "processed" / "chunks.parquet"),
        "--cache_path",
        str(project_root / "data" / "embeddings" / "chunk_embeddings.parquet"),
        "--model_name",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--batch_size",
        "64",
        "--save_every_batches",
        "5",
        *sys.argv[1:],
    ]

    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
