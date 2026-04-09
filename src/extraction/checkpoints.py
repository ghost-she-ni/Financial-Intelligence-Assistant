from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CHECKPOINT_COLUMNS = [
    "chunk_id",
    "status",
    "record_count",
    "updated_at",
    "error_message",
]

SUCCESS_CHECKPOINT_STATUSES = {"success"}


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_checkpoint_path(output_path: Path) -> Path:
    """Store extractor checkpoints next to the main output artifact."""
    suffix = output_path.suffix or ".parquet"
    return output_path.with_name(f"{output_path.stem}_checkpoint{suffix}")


def empty_checkpoint_df() -> pd.DataFrame:
    """Return an empty extraction checkpoint table."""
    return pd.DataFrame(columns=CHECKPOINT_COLUMNS)


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Load a checkpoint table if it exists."""
    if not checkpoint_path.exists():
        return empty_checkpoint_df()

    if checkpoint_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(checkpoint_path)
    elif checkpoint_path.suffix.lower() == ".csv":
        df = pd.read_csv(checkpoint_path)
    else:
        raise ValueError("Checkpoint file must be .parquet or .csv")

    for column in CHECKPOINT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[CHECKPOINT_COLUMNS].copy()


def bootstrap_checkpoint_from_output(existing_output_df: pd.DataFrame) -> pd.DataFrame:
    """Create success checkpoint rows from previously saved extractor outputs."""
    if existing_output_df.empty or "chunk_id" not in existing_output_df.columns:
        return empty_checkpoint_df()

    counts_df = (
        existing_output_df["chunk_id"]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("chunk_id")
        .reset_index(name="record_count")
    )
    if counts_df.empty:
        return empty_checkpoint_df()

    counts_df["status"] = "success"
    counts_df["updated_at"] = ""
    counts_df["error_message"] = ""
    return counts_df[CHECKPOINT_COLUMNS].copy()


def load_or_bootstrap_checkpoint(
    checkpoint_path: Path,
    existing_output_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load a checkpoint and backfill success rows from existing extractor outputs."""
    checkpoint_df = load_checkpoint(checkpoint_path)
    bootstrap_df = bootstrap_checkpoint_from_output(existing_output_df)

    if bootstrap_df.empty:
        return checkpoint_df

    merged_df = pd.concat([checkpoint_df, bootstrap_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["chunk_id"], keep="last")
    return merged_df.sort_values("chunk_id").reset_index(drop=True)


def build_checkpoint_record(
    chunk_id: str,
    status: str,
    record_count: int,
    error_message: str = "",
) -> dict[str, Any]:
    """Build one extractor checkpoint record."""
    return {
        "chunk_id": str(chunk_id),
        "status": status,
        "record_count": int(record_count),
        "updated_at": now_utc_iso(),
        "error_message": str(error_message or ""),
    }


def upsert_checkpoint_records(
    checkpoint_df: pd.DataFrame,
    new_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Insert or replace checkpoint rows by chunk_id."""
    if not new_records:
        return checkpoint_df.copy()

    new_df = pd.DataFrame(new_records)
    for column in CHECKPOINT_COLUMNS:
        if column not in new_df.columns:
            new_df[column] = None

    merged_df = pd.concat([checkpoint_df, new_df[CHECKPOINT_COLUMNS]], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["chunk_id"], keep="last")
    return merged_df.sort_values("chunk_id").reset_index(drop=True)


def save_checkpoint(checkpoint_df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Persist a checkpoint table to parquet or csv."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.suffix.lower() == ".parquet":
        checkpoint_df.to_parquet(checkpoint_path, index=False)
    elif checkpoint_path.suffix.lower() == ".csv":
        checkpoint_df.to_csv(checkpoint_path, index=False)
    else:
        raise ValueError("Checkpoint file must be .parquet or .csv")


def get_successfully_processed_chunk_ids(checkpoint_df: pd.DataFrame) -> set[str]:
    """Return chunk ids that completed successfully, even with zero extracted rows."""
    if checkpoint_df.empty:
        return set()

    return set(
        checkpoint_df.loc[
            checkpoint_df["status"].fillna("").astype(str).isin(SUCCESS_CHECKPOINT_STATUSES),
            "chunk_id",
        ].astype(str)
    )
