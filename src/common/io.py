from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def read_table(path: Path) -> pd.DataFrame:
    """Load a parquet or CSV table."""
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")
