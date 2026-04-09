from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

CHUNK_EMBEDDING_COLUMNS = [
    "chunk_id",
    "doc_id",
    "company",
    "fiscal_year",
    "embedding_model",
    "text_hash",
    "embedding",
    "created_at",
]

QUERY_EMBEDDING_COLUMNS = [
    "query_id",
    "query_text",
    "embedding_model",
    "text_hash",
    "embedding",
    "created_at",
]

__all__ = [
    "now_utc_iso",
    "compute_text_hash",
    "resolve_text_hash",
    "make_query_id",
    "load_parquet_cache",
    "load_chunk_embedding_cache",
    "load_query_embedding_cache",
    "save_cache",
    "find_cached_chunk_embedding",
    "find_cached_query_embedding",
    "build_chunk_embedding_record",
    "build_query_embedding_record",
    "append_records_to_cache",
    "get_missing_chunk_rows",
    "get_missing_query_rows",
]


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def compute_text_hash(text: str | None) -> str:
    """
    Compute a stable SHA256 hash for a text string.

    Handles None and non‑string inputs by converting to string.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        # Convert to string (e.g., pandas NaN becomes "nan")
        text = str(text)

    normalized_text = text.strip()
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def resolve_text_hash(text_hash: object, text: str | None) -> str:
    """
    Return an existing non-empty text hash, otherwise recompute it from the text.
    """
    if isinstance(text_hash, str) and text_hash.strip():
        return text_hash.strip()
    return compute_text_hash(text)


def make_query_id(query_text: str) -> str:
    """Build a stable query identifier from the query text."""
    return compute_text_hash(query_text)


def load_parquet_cache(cache_path: Path, expected_columns: List[str]) -> pd.DataFrame:
    """
    Load a cache parquet file if it exists, otherwise return an empty DataFrame
    with the expected schema.
    """
    if not cache_path.exists():
        logger.debug(f"Cache file not found: {cache_path}, starting empty.")
        return pd.DataFrame(columns=expected_columns)

    try:
        df = pd.read_parquet(cache_path)
    except Exception as e:
        raise ValueError(f"Failed to read cache file {cache_path}: {e}") from e

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in cache file {cache_path}: {sorted(missing_cols)}"
        )

    logger.info(f"Loaded cache from {cache_path} with {len(df)} rows.")
    return df[expected_columns].copy()


def load_chunk_embedding_cache(cache_path: Path) -> pd.DataFrame:
    """Load the chunk embedding cache."""
    return load_parquet_cache(cache_path, CHUNK_EMBEDDING_COLUMNS)


def load_query_embedding_cache(cache_path: Path) -> pd.DataFrame:
    """Load the query embedding cache."""
    return load_parquet_cache(cache_path, QUERY_EMBEDDING_COLUMNS)


def save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save a cache DataFrame to Parquet."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved cache to {cache_path} with {len(df)} rows.")
    except Exception as e:
        raise ValueError(f"Failed to save cache to {cache_path}: {e}") from e


def find_cached_chunk_embedding(
    cache_df: pd.DataFrame,
    chunk_id: str,
    embedding_model: str,
    text_hash: str,
) -> pd.Series | None:
    """
    Return the cached chunk embedding row if it exists.
    """
    matches = cache_df[
        (cache_df["chunk_id"] == chunk_id)
        & (cache_df["embedding_model"] == embedding_model)
        & (cache_df["text_hash"] == text_hash)
    ]

    if matches.empty:
        return None
    return matches.iloc[0]


def find_cached_query_embedding(
    cache_df: pd.DataFrame,
    query_id: str,
    embedding_model: str,
    text_hash: str,
) -> pd.Series | None:
    """
    Return the cached query embedding row if it exists.
    """
    matches = cache_df[
        (cache_df["query_id"] == query_id)
        & (cache_df["embedding_model"] == embedding_model)
        & (cache_df["text_hash"] == text_hash)
    ]

    if matches.empty:
        return None
    return matches.iloc[0]


def build_chunk_embedding_record(
    chunk_id: str,
    doc_id: str,
    company: str,
    fiscal_year: int,
    embedding_model: str,
    text_hash: str,
    embedding: Iterable[float],
) -> dict:
    """Build a row for the chunk embedding cache."""
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "company": company,
        "fiscal_year": int(fiscal_year),
        "embedding_model": embedding_model,
        "text_hash": text_hash,
        "embedding": list(embedding),
        "created_at": now_utc_iso(),
    }


def build_query_embedding_record(
    query_id: str,
    query_text: str,
    embedding_model: str,
    text_hash: str,
    embedding: Iterable[float],
) -> dict:
    """Build a row for the query embedding cache."""
    return {
        "query_id": query_id,
        "query_text": query_text,
        "embedding_model": embedding_model,
        "text_hash": text_hash,
        "embedding": list(embedding),
        "created_at": now_utc_iso(),
    }


def append_records_to_cache(
    cache_df: pd.DataFrame,
    new_records: List[dict],
    key_columns: List[str],
) -> pd.DataFrame:
    """
    Append new records and remove duplicates on key columns, keeping the latest one.

    This function assumes that `new_records` contain the most up‑to‑date information
    for each key. Duplicates (same key) in the cache are replaced by the new entries.
    """
    if not new_records:
        logger.debug("No new records to append.")
        return cache_df.copy()

    new_df = pd.DataFrame(new_records)
    logger.info(f"Appending {len(new_df)} new records to cache.")

    merged_df = pd.concat([cache_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=key_columns, keep="last")

    logger.debug(f"Cache after deduplication: {len(merged_df)} rows.")
    return merged_df.reset_index(drop=True)


def get_missing_chunk_rows(
    chunks_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    embedding_model: str,
    text_column: str = "chunk_text",
) -> pd.DataFrame:
    """
    Return only chunk rows that still need embeddings for the given model and text hash.
    """
    working_df = chunks_df.copy()
    working_df["text_hash"] = working_df[text_column].fillna("").apply(compute_text_hash)
    working_df["embedding_model"] = embedding_model

    cache_keys = cache_df[["chunk_id", "embedding_model", "text_hash"]].drop_duplicates()
    cache_keys["is_cached"] = True

    merged = working_df.merge(
        cache_keys,
        on=["chunk_id", "embedding_model", "text_hash"],
        how="left",
    )

    missing_df = merged[merged["is_cached"].ne(True)].drop(columns=["is_cached"])
    logger.info(f"Found {len(missing_df)} chunks missing embeddings (out of {len(chunks_df)} total).")
    return missing_df.reset_index(drop=True)


def get_missing_query_rows(
    queries_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    embedding_model: str,
    query_text_column: str = "query_text",
    query_id_column: str = "query_id",
) -> pd.DataFrame:
    """
    Return only query rows that still need embeddings for the given model and text hash.
    """
    working_df = queries_df.copy()

    if query_id_column not in working_df.columns:
        working_df[query_id_column] = working_df[query_text_column].apply(make_query_id)

    working_df["text_hash"] = working_df[query_text_column].fillna("").apply(compute_text_hash)
    working_df["embedding_model"] = embedding_model

    cache_keys = cache_df[["query_id", "embedding_model", "text_hash"]].drop_duplicates()
    cache_keys["is_cached"] = True

    merged = working_df.merge(
        cache_keys,
        on=["query_id", "embedding_model", "text_hash"],
        how="left",
    )

    missing_df = merged[merged["is_cached"].ne(True)].drop(columns=["is_cached"])
    logger.info(f"Found {len(missing_df)} queries missing embeddings (out of {len(queries_df)} total).")
    return missing_df.reset_index(drop=True)
