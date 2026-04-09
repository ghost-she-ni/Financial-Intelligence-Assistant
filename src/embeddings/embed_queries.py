from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.embeddings.cache import (
    append_records_to_cache,
    build_query_embedding_record,
    compute_text_hash,
    get_missing_query_rows,
    load_query_embedding_cache,
    make_query_id,
    resolve_text_hash,
    save_cache,
)

# Configure module logger
logger = logging.getLogger(__name__)

REQUIRED_QUERY_COLUMNS = {
    "query_text",
}


def load_queries(queries_path: Path) -> pd.DataFrame:
    """
    Load queries from a Parquet or CSV file.
    """
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file does not exist: {queries_path}")

    if queries_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(queries_path)
    elif queries_path.suffix.lower() == ".csv":
        df = pd.read_csv(queries_path)
    else:
        raise ValueError("Queries file must be .parquet or .csv")

    missing_cols = REQUIRED_QUERY_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required query columns: {sorted(missing_cols)}")

    df = df.copy()
    df["query_text"] = df["query_text"].fillna("").astype(str).str.strip()
    df = df[df["query_text"] != ""].reset_index(drop=True)

    if "query_id" not in df.columns:
        df["query_id"] = df["query_text"].apply(make_query_id)
    else:
        df["query_id"] = df["query_id"].fillna("").astype(str).str.strip()
        missing_id_mask = df["query_id"] == ""
        df.loc[missing_id_mask, "query_id"] = df.loc[missing_id_mask, "query_text"].apply(
            make_query_id
        )

    # Remove duplicate queries based on query_id
    df = df.drop_duplicates(subset=["query_id"], keep="first").reset_index(drop=True)

    return df


def load_embedding_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    """
    Load a sentence-transformers embedding model.
    """
    try:
        if device:
            return SentenceTransformer(model_name, device=device)
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """
    Compute normalized embeddings for a batch of texts.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def process_missing_queries(
    missing_queries_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    cache_path: Path,
    model: SentenceTransformer,
    embedding_model_name: str,
    batch_size: int,
    save_every_batches: int,
) -> pd.DataFrame:
    """
    Compute embeddings only for missing queries and update the cache incrementally.
    """
    if missing_queries_df.empty:
        logger.info("No missing query embeddings. Cache is already up to date.")
        return cache_df

    total_missing = len(missing_queries_df)
    logger.info(f"Missing query embeddings to compute: {total_missing}")

    new_records_buffer: list[dict] = []
    processed_count = 0
    batch_counter = 0

    for start_idx in tqdm(range(0, total_missing, batch_size), desc="Embedding queries"):
        end_idx = min(start_idx + batch_size, total_missing)
        batch_df = missing_queries_df.iloc[start_idx:end_idx].copy()

        texts = batch_df["query_text"].fillna("").tolist()
        embeddings = embed_texts(model=model, texts=texts, batch_size=batch_size)

        for row, embedding in zip(batch_df.itertuples(index=False), embeddings):
            text_hash = resolve_text_hash(
                text_hash=getattr(row, "text_hash", None),
                text=row.query_text,
            )
            new_records_buffer.append(
                build_query_embedding_record(
                    query_id=row.query_id,
                    query_text=row.query_text,
                    embedding_model=embedding_model_name,
                    text_hash=text_hash,
                    embedding=embedding,
                )
            )

        processed_count += len(batch_df)
        batch_counter += 1

        if batch_counter % save_every_batches == 0 or processed_count == total_missing:
            cache_df = append_records_to_cache(
                cache_df=cache_df,
                new_records=new_records_buffer,
                key_columns=["query_id", "embedding_model"],
            )
            save_cache(cache_df, cache_path)

            logger.info(
                f"Saved cache update: {processed_count}/{total_missing} missing query embeddings processed."
            )
            new_records_buffer = []

    return cache_df


def print_summary(
    queries_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    missing_queries_df: pd.DataFrame,
    embedding_model_name: str,
) -> None:
    """
    Print a short summary using logging.
    """
    total_queries = len(queries_df)
    total_cached_for_model = len(
        cache_df[cache_df["embedding_model"] == embedding_model_name]
    )

    logger.info("Query embedding completed.")
    logger.info(f"Embedding model: {embedding_model_name}")
    logger.info(f"Total queries in dataset: {total_queries}")
    logger.info(f"Missing queries before run: {len(missing_queries_df)}")
    logger.info(f"Cached query embeddings for model after run: {total_cached_for_model}")

    if total_queries > 0:
        coverage = 100.0 * total_cached_for_model / total_queries
        logger.info(f"Coverage for model: {coverage:.2f}%")

    preview_cols = ["query_id", "query_text", "embedding_model"]
    preview_df = cache_df[cache_df["embedding_model"] == embedding_model_name][
        preview_cols
    ].head(10)
    logger.info("Cache preview:")
    logger.info("\n" + preview_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and cache query embeddings.")
    parser.add_argument(
        "--queries_path",
        type=str,
        default="data/queries/queries.parquet",
        help="Path to queries file (.parquet or .csv).",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/embeddings/query_embeddings.parquet",
        help="Path to query embedding cache (.parquet).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation.",
    )
    parser.add_argument(
        "--save_every_batches",
        type=int,
        default=5,
        help="Save cache every N embedding batches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute missing query count, do not generate embeddings.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Process only the first N missing queries (by query_id).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and recompute all queries.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    queries_path = Path(args.queries_path)
    cache_path = Path(args.cache_path)

    # Ensure cache directory is writable
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading queries...")
    try:
        queries_df = load_queries(queries_path)
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        sys.exit(1)

    if queries_df.empty:
        logger.warning("Queries DataFrame is empty. Nothing to do.")
        sys.exit(0)

    logger.info("Loading existing query embedding cache...")
    try:
        cache_df = load_query_embedding_cache(cache_path)
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        sys.exit(1)

    logger.info("Checking which query embeddings are missing...")
    if args.force:
        logger.info("Force mode enabled: all queries will be recomputed.")
        missing_queries_df = queries_df.copy()
        missing_queries_df["text_hash"] = (
            missing_queries_df["query_text"].fillna("").apply(compute_text_hash)
        )
    else:
        missing_queries_df = get_missing_query_rows(
            queries_df=queries_df,
            cache_df=cache_df,
            embedding_model=args.model_name,
            query_text_column="query_text",
            query_id_column="query_id",
        )

    if missing_queries_df.empty:
        logger.info("No missing queries. Exiting.")
        sys.exit(0)

    # Apply limit if provided
    if args.limit_queries is not None and args.limit_queries > 0:
        missing_queries_df = missing_queries_df.head(args.limit_queries)
        logger.info(f"Limited to first {len(missing_queries_df)} missing queries.")

    if args.dry_run:
        logger.info(
            f"Dry run: would compute embeddings for {len(missing_queries_df)} queries."
        )
        logger.info("Exiting without computation.")
        sys.exit(0)

    logger.info("Loading embedding model...")
    try:
        model = load_embedding_model(args.model_name, device=args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    cache_df = process_missing_queries(
        missing_queries_df=missing_queries_df,
        cache_df=cache_df,
        cache_path=cache_path,
        model=model,
        embedding_model_name=args.model_name,
        batch_size=args.batch_size,
        save_every_batches=args.save_every_batches,
    )

    print_summary(
        queries_df=queries_df,
        cache_df=cache_df,
        missing_queries_df=missing_queries_df,
        embedding_model_name=args.model_name,
    )

    logger.info(f"Saved to: {cache_path.resolve()}")


if __name__ == "__main__":
    main()
