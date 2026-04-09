from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.retrieval.retrieve import (
    build_persistent_chunk_index,
    get_default_persistent_index_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent retrieval index artifact.")
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to the chunk metadata file.",
    )
    parser.add_argument(
        "--chunk_embeddings_path",
        type=str,
        default="data/embeddings/chunk_embeddings.parquet",
        help="Path to the chunk embeddings cache.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to materialize.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="Optional artifact directory. Defaults to data/indexes/retrieval.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="native",
        choices=["native", "faiss"],
        help="Persistent retrieval backend to materialize.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the persistent index even if it is already current.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    chunks_path = (PROJECT_ROOT / args.chunks_path).resolve()
    chunk_embeddings_path = (PROJECT_ROOT / args.chunk_embeddings_path).resolve()
    index_dir = (
        (PROJECT_ROOT / args.index_dir).resolve()
        if args.index_dir is not None
        else get_default_persistent_index_dir(chunks_path)
    )

    try:
        paths = build_persistent_chunk_index(
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            embedding_model=args.embedding_model,
            index_dir=index_dir,
            backend=args.backend,
            force_rebuild=args.force,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("Persistent retrieval index build failed: %s", exc)
        sys.exit(1)

    print("Persistent retrieval index ready.")
    print(f"Backend    : {args.backend}")
    print(f"Index root : {paths.index_root}")
    print(f"Model dir  : {paths.model_dir}")
    print(f"Manifest   : {paths.manifest_path}")


if __name__ == "__main__":
    main()
