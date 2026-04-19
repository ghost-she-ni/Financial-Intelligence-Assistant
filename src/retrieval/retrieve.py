from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.embeddings.cache import make_query_id
from src.extraction.knowledge_base import get_knowledge_artifacts
from src.preprocessing.sections import (
    compute_section_intent_score,
    compute_section_mismatch_penalty,
    infer_chunk_section_metadata,
)

# Configure module logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants & pre‑compiled patterns
# ----------------------------------------------------------------------

REQUIRED_CHUNK_COLUMNS = {
    "chunk_id",
    "doc_id",
    "company",
    "fiscal_year",
    "page_start",
    "page_end",
    "chunk_text",
}

OPTIONAL_CHUNK_COLUMNS = {
    "document_source",
    "document_type",
    "file_name",
    "section_id",
    "section_code",
    "section_title",
    "section_group",
}

REQUIRED_CHUNK_EMBEDDING_COLUMNS = {
    "chunk_id",
    "embedding_model",
    "embedding",
}

REQUIRED_QUERY_EMBEDDING_COLUMNS = {
    "query_id",
    "query_text",
    "embedding_model",
    "embedding",
}

COMPANY_ALIASES = {
    "adobe": "adobe",
    "lockheed martin": "lockheedmartin",
    "lockheedmartin": "lockheedmartin",
    "pfizer": "pfizer",
}

# Line-level noise patterns. These are intentionally conservative because
# section headers such as "Item 1A. Risk Factors" are useful retrieval signals.
NOISE_PATTERNS = [
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*exhibits?\s*$", re.IGNORECASE),
    re.compile(r"^\s*signatures?\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*report of independent registered public accounting firm\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*incorporated by reference\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*report on internal control over financial reporting\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*form\s+10[-]?k\s*$", re.IGNORECASE),
    re.compile(r"^\s*securities and exchange commission\s*$", re.IGNORECASE),
    re.compile(r"^\s*washington,\s*d\.c\.\s*20549\s*$", re.IGNORECASE),
    re.compile(r"^\s*item\s+\d+[ab]?\.\s*$", re.IGNORECASE),
]

# Exact lines that are pure noise
EXACT_NOISE_LINES = {
    "",
    "Table of Contents",
    "Page",
    "Part I",
    "Part II",
    "Part III",
    "Part IV",
    "Item 1.",
    "Item 1A.",
    "Item 1B.",
    "Item 2.",
    "Item 3.",
    "Item 4.",
    "Item 5.",
    "Item 6.",
    "Item 7.",
    "Item 7A.",
    "Item 8.",
    "Item 9.",
    "Item 9A.",
    "Item 9B.",
    "Item 10.",
    "Item 11.",
    "Item 12.",
    "Item 13.",
    "Item 14.",
    "Item 15.",
}

STOPWORDS = {
    "what",
    "are",
    "the",
    "in",
    "did",
    "how",
    "main",
    "of",
    "or",
    "and",
    "to",
    "by",
    "for",
    "with",
    "on",
    "from",
    "a",
    "an",
    "their",
    "its",
    "annual",
    "report",
}

IMPORTANT_SHORT_TOKENS = {
    "ai",
    "ar",
    "vr",
    "ml",
    "ui",
    "ux",
    "eps",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

BM25_K1 = 1.5
BM25_B = 0.75
PERSISTENT_INDEX_FORMAT_VERSION = 3
NATIVE_DENSE_BACKEND = "native_dense"
FAISS_BACKEND = "faiss_flat_ip"
CLASSICAL_ML_RETRIEVAL_MODE = "classical_ml"
NAIVE_RETRIEVAL_MODE = "naive"
IMPROVED_RETRIEVAL_MODE = "improved"
RETRIEVAL_MODE_ALIASES = {
    "classical_ml": CLASSICAL_ML_RETRIEVAL_MODE,
    "classic_ml": CLASSICAL_ML_RETRIEVAL_MODE,
    "ml": CLASSICAL_ML_RETRIEVAL_MODE,
    "tfidf": CLASSICAL_ML_RETRIEVAL_MODE,
    "naive": NAIVE_RETRIEVAL_MODE,
    "baseline": NAIVE_RETRIEVAL_MODE,
    "simple": NAIVE_RETRIEVAL_MODE,
    "improved": IMPROVED_RETRIEVAL_MODE,
    "advanced": IMPROVED_RETRIEVAL_MODE,
    "hybrid": IMPROVED_RETRIEVAL_MODE,
    "dense_hybrid": IMPROVED_RETRIEVAL_MODE,
}


@dataclass(frozen=True)
class CachedChunkIndex:
    vectors: np.ndarray
    metadata_df: pd.DataFrame
    noise_mask: np.ndarray
    backend: str = NATIVE_DENSE_BACKEND
    ann_index: object | None = None


@dataclass(frozen=True)
class CachedQueryEmbeddings:
    embeddings_by_query_id: dict[str, tuple[np.ndarray, str]]
    embeddings_by_query_text: dict[str, tuple[np.ndarray, str]]


@dataclass(frozen=True)
class CachedClassicalChunkIndex:
    metadata_df: pd.DataFrame
    noise_mask: np.ndarray
    document_term_counters: tuple[Counter[str], ...]
    idf_by_term: dict[str, float]
    document_norms: np.ndarray


@dataclass(frozen=True)
class PersistentIndexPaths:
    index_root: Path
    model_dir: Path
    manifest_path: Path
    metadata_path: Path
    vectors_path: Path
    noise_mask_path: Path
    ann_index_path: Path
    lock_path: Path

# Pre‑compile section keyword patterns
SECTION_KEYWORD_PATTERNS = {
    "risk_factors": [
        re.compile(r"risk factors", re.IGNORECASE),
        re.compile(r"\brisk\b", re.IGNORECASE),
        re.compile(r"\buncertaint", re.IGNORECASE),
        re.compile(r"\bcybersecurity\b", re.IGNORECASE),
        re.compile(r"\bthreat", re.IGNORECASE),
        re.compile(r"\badverse\b", re.IGNORECASE),
    ],
    "competition": [
        re.compile(r"\bcompetition\b", re.IGNORECASE),
        re.compile(r"\bcompetitive\b", re.IGNORECASE),
        re.compile(r"\bcompetitor", re.IGNORECASE),
        re.compile(r"\bcompete\b", re.IGNORECASE),
        re.compile(r"\bbid", re.IGNORECASE),
        re.compile(r"\bbidders\b", re.IGNORECASE),
        re.compile(r"\bmarket position\b", re.IGNORECASE),
    ],
    "business_segments": [
        re.compile(r"\bbusiness segment", re.IGNORECASE),
        re.compile(r"\bsegment\b", re.IGNORECASE),
        re.compile(r"\boperating segment\b", re.IGNORECASE),
        re.compile(r"\bcommercial\b", re.IGNORECASE),
        re.compile(r"\bportfolio\b", re.IGNORECASE),
        re.compile(r"\bproducts?\b", re.IGNORECASE),
        re.compile(r"\bservices?\b", re.IGNORECASE),
    ],
    "financial_metrics": [
        re.compile(r"\brevenue\b", re.IGNORECASE),
        re.compile(r"\bincome\b", re.IGNORECASE),
        re.compile(r"\bnet income\b", re.IGNORECASE),
        re.compile(r"\boperating income\b", re.IGNORECASE),
        re.compile(r"\bps\b", re.IGNORECASE),
        re.compile(r"\bdiluted eps\b", re.IGNORECASE),
        re.compile(r"\bcash flow\b", re.IGNORECASE),
        re.compile(r"\bmargin\b", re.IGNORECASE),
        re.compile(r"\bprofit\b", re.IGNORECASE),
    ],
    "ai": [
        re.compile(r"\bartificial intelligence\b", re.IGNORECASE),
        re.compile(r"\bgenerative ai\b", re.IGNORECASE),
        re.compile(r"\bgenerative artificial intelligence\b", re.IGNORECASE),
        re.compile(r"\bmachine learning\b", re.IGNORECASE),
        re.compile(r"\bai\b", re.IGNORECASE),
    ],
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet or CSV."""
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def validate_columns(df: pd.DataFrame, required_columns: set[str], df_name: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {sorted(missing)}")


def ensure_chunk_optional_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add optional retrieval metadata columns."""
    enriched_df = metadata_df.copy()

    for column in OPTIONAL_CHUNK_COLUMNS:
        if column not in enriched_df.columns:
            enriched_df[column] = None

    return enriched_df


def infer_missing_chunk_sections(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Infer section metadata only for candidate chunks missing explicit section fields."""
    enriched_df = ensure_chunk_optional_columns(metadata_df)
    missing_mask = (
        enriched_df["section_code"].isna()
        | enriched_df["section_title"].isna()
        | enriched_df["section_group"].isna()
    )

    if not missing_mask.any():
        return enriched_df

    inferred_records = (
        enriched_df.loc[missing_mask, "chunk_text"]
        .fillna("")
        .apply(infer_chunk_section_metadata)
        .tolist()
    )
    inferred_df = pd.DataFrame(inferred_records, index=enriched_df.index[missing_mask])

    for column in ["section_code", "section_title", "section_group"]:
        enriched_df.loc[missing_mask, column] = (
            enriched_df.loc[missing_mask, column]
            .fillna(inferred_df[column])
        )

    return enriched_df


def get_file_signature(path: Path) -> tuple[str, int, int]:
    """Return a cache key that changes whenever a file path or file contents likely changed."""
    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"File does not exist: {resolved_path}")

    stat = resolved_path.stat()
    return str(resolved_path), stat.st_mtime_ns, stat.st_size


def find_project_root(start_path: Path) -> Path | None:
    """Find the nearest parent directory that looks like the project root."""
    resolved_start = start_path.resolve()
    candidates = [resolved_start] if resolved_start.is_dir() else [resolved_start.parent]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def get_portable_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise a POSIX path string."""
    resolved_path = path.resolve()
    project_root = find_project_root(resolved_path)
    if project_root is None:
        return resolved_path.as_posix()

    try:
        return resolved_path.relative_to(project_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def compute_file_sha256(path: Path) -> str:
    """Compute a SHA256 digest for a file without relying on platform metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_portable_file_signature(path: Path) -> dict[str, int | str]:
    """Return a portable content signature for persisted reproducibility manifests."""
    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"File does not exist: {resolved_path}")

    return {
        "path": get_portable_path(resolved_path),
        "sha256": compute_file_sha256(resolved_path),
        "size": int(resolved_path.stat().st_size),
    }


def sanitize_artifact_component(value: str) -> str:
    """Convert an identifier such as an embedding model name into a safe path component."""
    sanitized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return sanitized or "default"


@lru_cache(maxsize=1)
def get_faiss_module() -> object | None:
    """Import faiss lazily so the project still works without the optional dependency."""
    try:
        import faiss  # type: ignore
    except ImportError:
        return None
    return faiss


def resolve_persistent_backend(backend: str) -> str:
    """Normalize persistent index backend aliases."""
    normalized_backend = backend.lower().strip()
    backend_aliases = {
        "native": NATIVE_DENSE_BACKEND,
        NATIVE_DENSE_BACKEND: NATIVE_DENSE_BACKEND,
        "faiss": FAISS_BACKEND,
        FAISS_BACKEND: FAISS_BACKEND,
    }

    if normalized_backend not in backend_aliases:
        raise ValueError("persistent index backend must be one of: native, faiss.")

    return backend_aliases[normalized_backend]


def get_default_persistent_index_dir(chunks_path: Path) -> Path:
    """Store retrieval index artifacts under data/indexes/retrieval by default."""
    return chunks_path.resolve().parents[1] / "indexes" / "retrieval"


def resolve_persistent_index_paths(
    chunks_path: Path,
    embedding_model: str,
    backend: str,
    index_dir: Path | None = None,
) -> PersistentIndexPaths:
    """Resolve artifact paths for one embedding model."""
    index_root = (index_dir or get_default_persistent_index_dir(chunks_path)).resolve()
    model_root = index_root / sanitize_artifact_component(embedding_model)
    model_dir = model_root / sanitize_artifact_component(backend)

    return PersistentIndexPaths(
        index_root=index_root,
        model_dir=model_dir,
        manifest_path=model_dir / "manifest.json",
        metadata_path=model_dir / "metadata.parquet",
        vectors_path=model_dir / "vectors.npy",
        noise_mask_path=model_dir / "noise_mask.npy",
        ann_index_path=model_dir / "index.faiss",
        lock_path=model_dir / "build.lock",
    )


def build_source_signature_payload(
    chunks_path: Path,
    chunk_embeddings_path: Path,
) -> dict[str, dict[str, int | str]]:
    """Capture the source files that define a chunk index build."""
    return {
        "chunks": get_portable_file_signature(chunks_path),
        "chunk_embeddings": get_portable_file_signature(chunk_embeddings_path),
    }


def load_json_file(path: Path) -> dict | None:
    """Load a JSON file, returning None if it is missing or invalid."""
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON atomically to reduce the chance of partial index artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary_path.replace(path)


def write_faiss_index_atomic(index: object, path: Path) -> None:
    """Write a FAISS index atomically."""
    faiss = get_faiss_module()
    if faiss is None:
        raise ImportError("faiss is not installed. Install faiss-cpu to use the faiss backend.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    faiss.write_index(index, str(temporary_path))
    temporary_path.replace(path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def persistent_index_artifacts_exist(paths: PersistentIndexPaths, backend: str) -> bool:
    """Check whether all required persistent artifacts exist for one backend."""
    required_paths = [
        paths.manifest_path,
        paths.metadata_path,
        paths.vectors_path,
        paths.noise_mask_path,
    ]
    if backend == FAISS_BACKEND:
        required_paths.append(paths.ann_index_path)

    return all(path.exists() for path in required_paths)


def prepare_chunk_index(
    chunks_df: pd.DataFrame,
    chunk_embeddings_df: pd.DataFrame,
    embedding_model: str,
) -> CachedChunkIndex:
    """Merge chunk metadata with embeddings and precompute retrieval helpers."""
    validate_columns(chunks_df, REQUIRED_CHUNK_COLUMNS, "chunks_df")
    validate_columns(chunk_embeddings_df, REQUIRED_CHUNK_EMBEDDING_COLUMNS, "chunk_embeddings_df")

    chunk_embeddings_df = chunk_embeddings_df[
        chunk_embeddings_df["embedding_model"] == embedding_model
    ].copy()

    if chunk_embeddings_df.empty:
        raise ValueError(f"No chunk embeddings found for model: {embedding_model}")

    merged_df = chunks_df.merge(
        chunk_embeddings_df[["chunk_id", "embedding_model", "embedding"]],
        on="chunk_id",
        how="inner",
    )

    if merged_df.empty:
        raise ValueError("No merged chunk data found after joining chunks with embeddings.")

    vectors = np.vstack(
        merged_df["embedding"].apply(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
    )
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vector_norms = np.where(vector_norms == 0.0, 1.0, vector_norms)
    normalized_vectors = np.ascontiguousarray(vectors / vector_norms, dtype=np.float32)
    normalized_vectors.setflags(write=False)

    metadata_columns = [
        "chunk_id",
        "doc_id",
        "company",
        "fiscal_year",
        "page_start",
        "page_end",
        "chunk_text",
    ]
    metadata_columns.extend(
        column for column in sorted(OPTIONAL_CHUNK_COLUMNS) if column in merged_df.columns
    )

    metadata_df = infer_missing_chunk_sections(merged_df[metadata_columns].copy())
    metadata_df = metadata_df.reset_index(drop=True)

    noise_mask = metadata_df["chunk_text"].fillna("").astype(str).apply(is_noisy_chunk).to_numpy()
    noise_mask = np.asarray(noise_mask, dtype=bool)
    noise_mask.setflags(write=False)

    return CachedChunkIndex(
        vectors=normalized_vectors,
        metadata_df=metadata_df,
        noise_mask=noise_mask,
        backend=NATIVE_DENSE_BACKEND,
        ann_index=None,
    )


def build_persistent_index_manifest(
    embedding_model: str,
    backend: str,
    source_signatures: dict[str, dict[str, int | str]],
    prepared_index: CachedChunkIndex,
) -> dict:
    """Build the manifest stored next to persistent retrieval artifacts."""
    vector_dim = int(prepared_index.vectors.shape[1]) if prepared_index.vectors.ndim == 2 else 0

    return {
        "format_version": PERSISTENT_INDEX_FORMAT_VERSION,
        "backend": backend,
        "embedding_model": embedding_model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vector_count": int(len(prepared_index.metadata_df)),
        "vector_dim": vector_dim,
        "source_signatures": source_signatures,
    }


def is_persistent_index_current(
    manifest_payload: dict | None,
    embedding_model: str,
    source_signatures: dict[str, dict[str, int | str]],
    expected_backend: str | None = None,
) -> bool:
    """Check whether a persistent index artifact is still valid for the current sources."""
    if manifest_payload is None:
        return False

    backend = manifest_payload.get("backend")
    if backend not in {NATIVE_DENSE_BACKEND, FAISS_BACKEND}:
        return False

    return (
        manifest_payload.get("format_version") == PERSISTENT_INDEX_FORMAT_VERSION
        and (expected_backend is None or backend == expected_backend)
        and manifest_payload.get("embedding_model") == embedding_model
        and manifest_payload.get("source_signatures") == source_signatures
    )


def acquire_persistent_index_lock(
    lock_path: Path,
    timeout_seconds: float = 15.0,
    poll_interval_seconds: float = 0.2,
) -> bool:
    """Acquire a cooperative file lock for index builds."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.write(datetime.now(timezone.utc).isoformat())
            return True
        except FileExistsError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_seconds)


def release_persistent_index_lock(lock_path: Path) -> None:
    """Release a cooperative file lock if this process still owns it."""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove persistent index lock: %s", lock_path)


def build_persistent_chunk_index(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    embedding_model: str,
    index_dir: Path | None = None,
    backend: str = "native",
    force_rebuild: bool = False,
) -> PersistentIndexPaths:
    """Build or refresh a persistent retrieval index artifact on disk."""
    backend_id = resolve_persistent_backend(backend)
    paths = resolve_persistent_index_paths(
        chunks_path=chunks_path,
        embedding_model=embedding_model,
        backend=backend_id,
        index_dir=index_dir,
    )
    source_signatures = build_source_signature_payload(chunks_path, chunk_embeddings_path)
    manifest_payload = load_json_file(paths.manifest_path)

    if (
        not force_rebuild
        and persistent_index_artifacts_exist(paths, backend_id)
        and is_persistent_index_current(
            manifest_payload=manifest_payload,
            embedding_model=embedding_model,
            source_signatures=source_signatures,
            expected_backend=backend_id,
        )
    ):
        return paths

    lock_acquired = acquire_persistent_index_lock(paths.lock_path)
    if not lock_acquired:
        manifest_payload = load_json_file(paths.manifest_path)
        if (
            persistent_index_artifacts_exist(paths, backend_id)
            and is_persistent_index_current(
                manifest_payload=manifest_payload,
                embedding_model=embedding_model,
                source_signatures=source_signatures,
                expected_backend=backend_id,
            )
        ):
            return paths
        raise TimeoutError(f"Timed out waiting for persistent index lock: {paths.lock_path}")

    try:
        manifest_payload = load_json_file(paths.manifest_path)
        if (
            not force_rebuild
            and persistent_index_artifacts_exist(paths, backend_id)
            and is_persistent_index_current(
                manifest_payload=manifest_payload,
                embedding_model=embedding_model,
                source_signatures=source_signatures,
                expected_backend=backend_id,
            )
        ):
            return paths

        chunks_df = load_dataframe(chunks_path)
        chunk_embeddings_df = load_dataframe(chunk_embeddings_path)
        prepared_index = prepare_chunk_index(
            chunks_df=chunks_df,
            chunk_embeddings_df=chunk_embeddings_df,
            embedding_model=embedding_model,
        )

        manifest_payload = build_persistent_index_manifest(
            embedding_model=embedding_model,
            backend=backend_id,
            source_signatures=source_signatures,
            prepared_index=prepared_index,
        )

        paths.model_dir.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(prepared_index.metadata_df, paths.metadata_path)
        write_numpy_atomic(paths.vectors_path, np.asarray(prepared_index.vectors, dtype=np.float32))
        write_numpy_atomic(paths.noise_mask_path, np.asarray(prepared_index.noise_mask, dtype=bool))
        if backend_id == FAISS_BACKEND:
            faiss = get_faiss_module()
            if faiss is None:
                raise ImportError(
                    "faiss is not installed. Install faiss-cpu to build the faiss backend."
                )
            faiss_index = faiss.IndexFlatIP(int(prepared_index.vectors.shape[1]))
            faiss_index.add(np.asarray(prepared_index.vectors, dtype=np.float32))
            write_faiss_index_atomic(faiss_index, paths.ann_index_path)
        write_json_atomic(paths.manifest_path, manifest_payload)
    finally:
        release_persistent_index_lock(paths.lock_path)

    return paths


@lru_cache(maxsize=8)
def _load_persistent_chunk_index_cached(
    backend: str,
    manifest_path_str: str,
    manifest_mtime_ns: int,
    manifest_size: int,
    metadata_path_str: str,
    metadata_mtime_ns: int,
    metadata_size: int,
    vectors_path_str: str,
    vectors_mtime_ns: int,
    vectors_size: int,
    noise_mask_path_str: str,
    noise_mask_mtime_ns: int,
    noise_mask_size: int,
    ann_index_path_str: str,
    ann_index_mtime_ns: int,
    ann_index_size: int,
) -> CachedChunkIndex:
    """Load a persistent retrieval artifact into an in-process cache."""
    _ = (
        backend,
        manifest_mtime_ns,
        manifest_size,
        metadata_mtime_ns,
        metadata_size,
        vectors_mtime_ns,
        vectors_size,
        noise_mask_mtime_ns,
        noise_mask_size,
        ann_index_mtime_ns,
        ann_index_size,
    )

    metadata_df = pd.read_parquet(metadata_path_str)
    metadata_df = ensure_chunk_optional_columns(metadata_df)
    validate_columns(metadata_df, REQUIRED_CHUNK_COLUMNS, "persistent_metadata_df")

    vectors = np.load(vectors_path_str, mmap_mode="r")
    noise_mask = np.asarray(np.load(noise_mask_path_str, mmap_mode="r"), dtype=bool)

    if vectors.ndim != 2:
        raise ValueError("Persistent retrieval vectors must be a 2D array.")
    if len(metadata_df) != int(vectors.shape[0]):
        raise ValueError("Persistent retrieval metadata length does not match vector count.")
    if len(noise_mask) != len(metadata_df):
        raise ValueError("Persistent retrieval noise mask length does not match metadata length.")

    ann_index = None
    if backend == FAISS_BACKEND:
        faiss = get_faiss_module()
        if faiss is None:
            raise ImportError(
                "faiss is not installed. Install faiss-cpu to load the faiss retrieval backend."
            )
        ann_index = faiss.read_index(ann_index_path_str)

    return CachedChunkIndex(
        vectors=vectors,
        metadata_df=metadata_df,
        noise_mask=noise_mask,
        backend=backend,
        ann_index=ann_index,
    )


def load_current_persistent_chunk_index(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    embedding_model: str,
    index_dir: Path | None = None,
    preferred_backend: str | None = None,
) -> CachedChunkIndex | None:
    """Load a current persistent retrieval artifact if one is available."""
    source_signatures = build_source_signature_payload(chunks_path, chunk_embeddings_path)

    candidate_backends = (
        [preferred_backend]
        if preferred_backend is not None
        else [NATIVE_DENSE_BACKEND, FAISS_BACKEND]
    )

    for backend in candidate_backends:
        paths = resolve_persistent_index_paths(
            chunks_path=chunks_path,
            embedding_model=embedding_model,
            backend=backend,
            index_dir=index_dir,
        )

        if not paths.manifest_path.exists():
            continue

        manifest_payload = load_json_file(paths.manifest_path)
        if manifest_payload is None:
            continue

        manifest_backend = manifest_payload.get("backend")
        if manifest_backend != backend:
            continue

        if not persistent_index_artifacts_exist(paths, backend):
            continue

        if not is_persistent_index_current(
            manifest_payload=manifest_payload,
            embedding_model=embedding_model,
            source_signatures=source_signatures,
            expected_backend=backend,
        ):
            continue

        ann_index_signature = ("", 0, 0)
        if backend == FAISS_BACKEND:
            ann_index_signature = get_file_signature(paths.ann_index_path)

        return _load_persistent_chunk_index_cached(
            backend,
            *get_file_signature(paths.manifest_path),
            *get_file_signature(paths.metadata_path),
            *get_file_signature(paths.vectors_path),
            *get_file_signature(paths.noise_mask_path),
            *ann_index_signature,
        )

    return None


@lru_cache(maxsize=8)
def _load_chunk_index_cached(
    chunks_path_str: str,
    chunks_mtime_ns: int,
    chunks_size: int,
    chunk_embeddings_path_str: str,
    chunk_embeddings_mtime_ns: int,
    chunk_embeddings_size: int,
    embedding_model: str,
) -> CachedChunkIndex:
    """Load and prepare the chunk index once per file signature + embedding model."""
    _ = (
        chunks_mtime_ns,
        chunks_size,
        chunk_embeddings_mtime_ns,
        chunk_embeddings_size,
    )

    chunks_df = load_dataframe(Path(chunks_path_str))
    chunk_embeddings_df = load_dataframe(Path(chunk_embeddings_path_str))
    return prepare_chunk_index(
        chunks_df=chunks_df,
        chunk_embeddings_df=chunk_embeddings_df,
        embedding_model=embedding_model,
    )


def get_cached_chunk_index(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    embedding_model: str,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
) -> CachedChunkIndex:
    """Return the best available chunk index for the current file versions."""
    normalized_mode = persistent_index_mode.lower().strip()
    if normalized_mode not in {"auto", "persistent", "source"}:
        raise ValueError(
            "persistent_index_mode must be one of: auto, persistent, source."
        )

    normalized_backend = persistent_index_backend.lower().strip()
    if normalized_backend not in {"auto", "native", "faiss", NATIVE_DENSE_BACKEND, FAISS_BACKEND}:
        raise ValueError(
            "persistent_index_backend must be one of: auto, native, faiss."
        )

    preferred_backend = None if normalized_backend == "auto" else resolve_persistent_backend(
        normalized_backend
    )
    build_backend = preferred_backend or NATIVE_DENSE_BACKEND

    if normalized_mode in {"auto", "persistent"}:
        persistent_index = None
        try:
            if rebuild_persistent_index:
                build_persistent_chunk_index(
                    chunks_path=chunks_path,
                    chunk_embeddings_path=chunk_embeddings_path,
                    embedding_model=embedding_model,
                    index_dir=persistent_index_dir,
                    backend=build_backend,
                    force_rebuild=True,
                )

            persistent_index = load_current_persistent_chunk_index(
                chunks_path=chunks_path,
                chunk_embeddings_path=chunk_embeddings_path,
                embedding_model=embedding_model,
                index_dir=persistent_index_dir,
                preferred_backend=preferred_backend,
            )
        except Exception as exc:
            if normalized_mode == "auto":
                logger.warning(
                    "Persistent retrieval index unavailable, falling back to source data: %s",
                    exc,
                )
            else:
                raise

        if persistent_index is not None:
            return persistent_index

        if normalized_mode == "auto":
            try:
                build_persistent_chunk_index(
                    chunks_path=chunks_path,
                    chunk_embeddings_path=chunk_embeddings_path,
                    embedding_model=embedding_model,
                    index_dir=persistent_index_dir,
                    backend=build_backend,
                    force_rebuild=False,
                )
                persistent_index = load_current_persistent_chunk_index(
                    chunks_path=chunks_path,
                    chunk_embeddings_path=chunk_embeddings_path,
                    embedding_model=embedding_model,
                    index_dir=persistent_index_dir,
                    preferred_backend=preferred_backend,
                )
                if persistent_index is not None:
                    return persistent_index
            except Exception as exc:
                logger.warning("Persistent retrieval index unavailable, falling back to source data: %s", exc)

        if normalized_mode == "persistent":
            raise ValueError("Persistent retrieval index could not be loaded or built.")

    return _load_chunk_index_cached(
        *get_file_signature(chunks_path),
        *get_file_signature(chunk_embeddings_path),
        embedding_model,
    )


@lru_cache(maxsize=8)
def _load_query_embeddings_cached(
    query_embeddings_path_str: str,
    query_embeddings_mtime_ns: int,
    query_embeddings_size: int,
    embedding_model: str,
) -> CachedQueryEmbeddings:
    """Load query embeddings once per file signature + embedding model."""
    _ = (query_embeddings_mtime_ns, query_embeddings_size)

    query_embeddings_df = load_dataframe(Path(query_embeddings_path_str))
    validate_columns(query_embeddings_df, REQUIRED_QUERY_EMBEDDING_COLUMNS, "query_embeddings_df")

    query_embeddings_df = query_embeddings_df[
        query_embeddings_df["embedding_model"] == embedding_model
    ].copy()

    if query_embeddings_df.empty:
        raise ValueError(f"No query embeddings found for model: {embedding_model}")

    query_embeddings_df = query_embeddings_df.drop_duplicates(
        subset=["query_id"],
        keep="last",
    )

    embeddings_by_query_id: dict[str, tuple[np.ndarray, str]] = {}
    embeddings_by_query_text: dict[str, tuple[np.ndarray, str]] = {}
    for row in query_embeddings_df.itertuples(index=False):
        embedding = np.asarray(row.embedding, dtype=np.float32)
        embedding.setflags(write=False)
        embeddings_by_query_id[row.query_id] = (embedding, row.query_text)
        embeddings_by_query_text[str(row.query_text)] = (embedding, row.query_id)

    return CachedQueryEmbeddings(
        embeddings_by_query_id=embeddings_by_query_id,
        embeddings_by_query_text=embeddings_by_query_text,
    )


def get_cached_query_embeddings(
    query_embeddings_path: Path,
    embedding_model: str,
) -> CachedQueryEmbeddings:
    """Return cached query embeddings for the current cache file version."""
    return _load_query_embeddings_cached(
        *get_file_signature(query_embeddings_path),
        embedding_model,
    )


def clear_retrieval_caches() -> None:
    """Clear in-memory retrieval caches for long-lived processes."""
    _load_persistent_chunk_index_cached.cache_clear()
    _load_chunk_index_cached.cache_clear()
    _load_query_embeddings_cached.cache_clear()
    _load_classical_chunk_index_cached.cache_clear()


def normalize_term(token: str) -> str:
    """Lightweight normalization to improve lexical matching across singular/plural forms."""
    normalized = token.strip().lower().replace("’", "'")
    if normalized.endswith("'s"):
        normalized = normalized[:-2]

    if len(normalized) > 4:
        if normalized.endswith("ies"):
            normalized = normalized[:-3] + "y"
        elif normalized.endswith("s") and not normalized.endswith("ss"):
            normalized = normalized[:-1]

    return normalized


def tokenize_text(text: str, drop_stopwords: bool = False) -> list[str]:
    """Tokenize text for lexical retrieval and reranking."""
    if not isinstance(text, str) or not text.strip():
        return []

    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = normalize_term(raw_token)
        if not token:
            continue
        if len(token) < 3 and token not in IMPORTANT_SHORT_TOKENS:
            continue
        if drop_stopwords and token in STOPWORDS and token not in IMPORTANT_SHORT_TOKENS:
            continue
        tokens.append(token)

    return tokens


def normalize_score_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a score series while handling flat or empty inputs."""
    if series.empty:
        return series.copy()

    min_value = float(series.min())
    max_value = float(series.max())

    if np.isclose(max_value, min_value):
        return series.apply(lambda value: 1.0 if value > 0 else 0.0)

    return (series - min_value) / (max_value - min_value)


def compute_cosine_scores(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity scores for normalized candidate vectors."""
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    query_norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_norms = np.where(query_norms == 0.0, 1.0, query_norms)
    normalized_query = query_vector / query_norms

    return np.dot(vectors, normalized_query[0])


def compute_bm25_scores(query_text: str, documents: pd.Series) -> pd.Series:
    """Compute BM25 scores over a set of candidate chunk texts."""
    query_terms = tokenize_text(query_text, drop_stopwords=True)
    if not query_terms:
        return pd.Series(np.zeros(len(documents), dtype=np.float32), index=documents.index)

    document_counters: list[Counter[str]] = []
    document_lengths: list[int] = []
    document_frequency: Counter[str] = Counter()

    for text in documents.fillna("").astype(str):
        terms = tokenize_text(text, drop_stopwords=False)
        counter = Counter(terms)
        document_counters.append(counter)
        document_lengths.append(len(terms))

        for term in set(query_terms):
            if counter.get(term, 0) > 0:
                document_frequency[term] += 1

    avgdl = float(np.mean(document_lengths)) if document_lengths else 1.0
    total_documents = max(1, len(documents))
    scores: list[float] = []

    for counter, doc_len in zip(document_counters, document_lengths):
        score = 0.0
        for term in query_terms:
            term_frequency = counter.get(term, 0)
            if term_frequency == 0:
                continue

            df = document_frequency.get(term, 0)
            idf = np.log(1.0 + ((total_documents - df + 0.5) / (df + 0.5)))
            numerator = term_frequency * (BM25_K1 + 1.0)
            denominator = term_frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * (doc_len / max(avgdl, 1.0))
            )
            score += idf * numerator / denominator

        scores.append(score)

    return pd.Series(scores, index=documents.index, dtype=np.float32)


def keyword_coverage_score(query_text: str, chunk_text: str) -> float:
    """Compute the fraction of salient query terms covered by the chunk."""
    query_terms = set(tokenize_text(query_text, drop_stopwords=True))
    if not query_terms:
        return 0.0

    chunk_terms = set(tokenize_text(chunk_text, drop_stopwords=False))
    return len(query_terms & chunk_terms) / len(query_terms)


def header_keyword_score(
    query_text: str,
    chunk_text: str,
    section_title: str | None = None,
) -> float:
    """Give a stronger bonus when the header or section title aligns with the query."""
    header_lines = [line.strip() for line in str(chunk_text).splitlines() if line.strip()][:6]
    header_text = "\n".join(header_lines)

    if section_title:
        header_text = f"{section_title}\n{header_text}"

    return keyword_coverage_score(query_text, header_text)


def numeric_density_penalty(
    query_text: str,
    chunk_text: str,
    section_group: str | None = None,
) -> float:
    """Penalize table-like chunks for non-financial questions."""
    intents = detect_query_intents(query_text)
    if "financial_metrics" in intents:
        return 0.0

    text = str(chunk_text)
    digits = sum(char.isdigit() for char in text)
    letters = sum(char.isalpha() for char in text)

    if digits == 0:
        return 0.0

    ratio = digits / max(1, digits + letters)
    penalty = 0.0

    if ratio > 0.18:
        penalty += min(0.35, (ratio - 0.18) * 1.5)

    if section_group == "financial_statements":
        penalty = max(penalty, 0.35)

    return min(0.60, penalty)


def relation_intent_score(query_text: str, relations: list[str] | None) -> float:
    """Boost chunks whose extracted relation types align with the question intent."""
    if not relations:
        return 0.0

    relation_set = {str(relation).strip().upper() for relation in relations if str(relation).strip()}
    if not relation_set:
        return 0.0

    query_lower = query_text.lower()
    intents = detect_query_intents(query_text)
    scores: list[float] = []

    if "competition" in intents and "COMPETES_WITH" in relation_set:
        scores.append(1.0)
    if "financial_metrics" in intents and "REPORTS" in relation_set:
        scores.append(1.0)
    if "risk_factors" in intents and "FACES_RISK_FROM" in relation_set:
        scores.append(1.0)
    if (
        any(term in query_lower for term in ["offer", "product", "products", "platform", "service"])
        and "OFFERS" in relation_set
    ):
        scores.append(0.85)
    if (
        any(term in query_lower for term in ["ceo", "chief executive", "lead", "led by", "executive"])
        and "LEADS_BY" in relation_set
    ):
        scores.append(0.85)
    if "MENTIONS" in relation_set and any(term in query_lower for term in ["mention", "mentions", "mentioned"]):
        scores.append(0.50)

    return max(scores, default=0.0)


def load_chunk_knowledge(
    chunks_path: Path,
) -> pd.DataFrame:
    """Load per-chunk extracted knowledge if entities/triplets artifacts exist."""
    knowledge_artifacts = get_knowledge_artifacts(chunks_path=chunks_path)
    return knowledge_artifacts.chunk_facts_df.copy()


def load_chunk_index_data(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    embedding_model: str,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load prepared chunk vectors + metadata from the in-memory cache."""
    cached_index = get_cached_chunk_index(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        embedding_model=embedding_model,
        persistent_index_mode=persistent_index_mode,
        persistent_index_backend=persistent_index_backend,
        persistent_index_dir=persistent_index_dir,
        rebuild_persistent_index=rebuild_persistent_index,
    )
    return cached_index.vectors, cached_index.metadata_df.copy()


def load_query_embedding(
    query_embeddings_path: Path,
    embedding_model: str,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
) -> tuple[np.ndarray, str, str]:
    """Load a query embedding from the in-memory cache."""
    if query_text is None and query_id is None:
        raise ValueError("Either query_text or query_id must be provided.")

    cached_query_embeddings = get_cached_query_embeddings(
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
    )

    if query_text is not None:
        resolved_query_id = make_query_id(query_text)
        match = cached_query_embeddings.embeddings_by_query_id.get(resolved_query_id)
        if match is None:
            match_by_text = cached_query_embeddings.embeddings_by_query_text.get(query_text)
            if match_by_text is None:
                raise ValueError(
                    "Query embedding not found for the provided query_text. "
                    "Run embed_queries.py first."
                )
            embedding, resolved_query_id = match_by_text
            resolved_query_text = query_text
            return embedding, resolved_query_id, resolved_query_text

        embedding, resolved_query_text = match
        return embedding, resolved_query_id, resolved_query_text

    if query_id is not None:
        match = cached_query_embeddings.embeddings_by_query_id.get(query_id)
        if match is None:
            raise ValueError(
                f"Query embedding not found for query_id={query_id}. "
                "Run embed_queries.py first."
            )
        embedding, resolved_query_text = match
        return embedding, query_id, resolved_query_text

    # Should never reach here
    raise ValueError("Invalid query arguments.")


def resolve_query_reference(
    query_embeddings_path: Path,
    embedding_model: str,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve stable query metadata even when a vector is not needed."""
    if query_text is None and query_id is None:
        raise ValueError("Either query_text or query_id must be provided.")

    cached_query_embeddings = get_cached_query_embeddings(
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
    )

    if query_text is not None:
        resolved_query_id = make_query_id(query_text)
        match = cached_query_embeddings.embeddings_by_query_id.get(resolved_query_id)
        if match is not None:
            _, resolved_query_text = match
            return resolved_query_id, resolved_query_text

        match_by_text = cached_query_embeddings.embeddings_by_query_text.get(query_text)
        if match_by_text is not None:
            _, resolved_query_id = match_by_text
            return resolved_query_id, query_text

        return resolved_query_id, query_text

    match = cached_query_embeddings.embeddings_by_query_id.get(query_id)
    if match is None:
        raise ValueError(
            f"Query embedding not found for query_id={query_id}. "
            "Run embed_queries.py first."
        )
    _, resolved_query_text = match
    return query_id, resolved_query_text


def normalize_retrieval_mode(retrieval_mode: str) -> str:
    """Resolve retrieval mode aliases into the canonical retrieval values."""
    normalized_mode = retrieval_mode.strip().lower()
    resolved_mode = RETRIEVAL_MODE_ALIASES.get(normalized_mode)
    if resolved_mode is None:
        raise ValueError(
            "retrieval_mode must be one of: "
            f"{', '.join(sorted(RETRIEVAL_MODE_ALIASES))}."
        )
    return resolved_mode


def _finalize_baseline_results(
    results_df: pd.DataFrame,
    resolved_query_id: str,
    resolved_query_text: str,
    retrieval_mode: str,
) -> pd.DataFrame:
    """Add a stable result schema for the comparable baseline retrievers."""
    finalized_df = results_df.copy()

    for column_name in [
        "lexical_score",
        "coverage_score",
        "section_score",
        "section_penalty",
        "header_score",
        "numeric_penalty",
        "knowledge_score",
        "knowledge_coverage_score",
        "knowledge_relation_score",
        "bm25_score",
        "preliminary_score",
        "hybrid_score",
        "rerank_score",
    ]:
        finalized_df[column_name] = 0.0

    finalized_df["final_score"] = finalized_df["score"].astype(np.float32)
    finalized_df["retrieval_mode"] = retrieval_mode

    finalized_df.insert(0, "query_id", resolved_query_id)
    finalized_df.insert(1, "query_text", resolved_query_text)
    return finalized_df


def retrieve_top_k_naive(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    top_k: int,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Retrieve the top-k chunks with the pedagogical naive dense baseline.

    This mode intentionally keeps only dense cosine similarity on top of the
    cached embeddings. It reuses the current corpus and embeddings so the
    comparison against the improved retriever stays fair.
    """
    if query_text is None and query_id is None:
        raise ValueError("Either query_text or query_id must be provided.")

    cached_index = get_cached_chunk_index(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        embedding_model=embedding_model,
        persistent_index_mode=persistent_index_mode,
        persistent_index_backend=persistent_index_backend,
        persistent_index_dir=persistent_index_dir,
        rebuild_persistent_index=rebuild_persistent_index,
    )

    query_vector, resolved_query_id, resolved_query_text = load_query_embedding(
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
        query_text=query_text,
        query_id=query_id,
    )

    filter_mask = build_retrieval_mask(
        metadata_df=cached_index.metadata_df,
        query_text=resolved_query_text,
        enable_metadata_filters=enable_metadata_filters,
        enable_noise_filter=enable_noise_filter,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        noise_mask=cached_index.noise_mask,
    )

    if not filter_mask.any():
        if verbose:
            logger.warning(
                "Naive retrieval filters removed all candidates; falling back to the full cached index."
            )
        filter_mask = build_retrieval_mask(
            metadata_df=cached_index.metadata_df,
            query_text=resolved_query_text,
            enable_metadata_filters=False,
            enable_noise_filter=enable_noise_filter,
            company_filter=company_filter,
            fiscal_year_filter=fiscal_year_filter,
            noise_mask=cached_index.noise_mask,
        )

    if not filter_mask.any():
        filter_mask = np.ones(len(cached_index.metadata_df), dtype=bool)

    candidate_indices = np.flatnonzero(filter_mask)
    candidate_metadata_df = cached_index.metadata_df.iloc[candidate_indices].reset_index(drop=True)
    candidate_vectors = cached_index.vectors[candidate_indices]

    if candidate_metadata_df.empty:
        raise ValueError("No candidates available after naive retrieval filtering.")

    candidate_scores = compute_cosine_scores(candidate_vectors, query_vector).astype(np.float32)
    results_df = candidate_metadata_df.copy()
    results_df["score"] = candidate_scores
    results_df = results_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    if verbose:
        logger.info("Naive retrieval summary:")
        logger.info("Candidate chunks kept: %s", len(candidate_metadata_df))
        logger.info("Top-k returned      : %s", len(results_df))
        logger.info("Noise filter enabled: %s", enable_noise_filter)
        logger.info("Metadata filters    : %s", enable_metadata_filters)

    return _finalize_baseline_results(
        results_df=results_df,
        resolved_query_id=resolved_query_id,
        resolved_query_text=resolved_query_text,
        retrieval_mode=NAIVE_RETRIEVAL_MODE,
    )


def _compute_tfidf_weights(term_counter: Counter[str], idf_by_term: dict[str, float]) -> dict[str, float]:
    """Build log-scaled TF-IDF weights for a sparse token counter."""
    weights: dict[str, float] = {}
    for term, frequency in term_counter.items():
        if frequency <= 0:
            continue
        idf = idf_by_term.get(term)
        if idf is None:
            continue
        weights[term] = (1.0 + float(np.log(frequency))) * idf
    return weights


@lru_cache(maxsize=8)
def _load_classical_chunk_index_cached(
    chunks_path_str: str,
    chunks_mtime_ns: int,
    chunks_size: int,
    chunk_embeddings_path_str: str,
    chunk_embeddings_mtime_ns: int,
    chunk_embeddings_size: int,
    embedding_model: str,
) -> CachedClassicalChunkIndex:
    """Prepare token-level TF-IDF statistics for the classical ML baseline."""
    cached_index = _load_chunk_index_cached(
        chunks_path_str,
        chunks_mtime_ns,
        chunks_size,
        chunk_embeddings_path_str,
        chunk_embeddings_mtime_ns,
        chunk_embeddings_size,
        embedding_model,
    )

    metadata_df = cached_index.metadata_df.copy()
    document_term_counters: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()

    for text in metadata_df["chunk_text"].fillna("").astype(str):
        term_counter = Counter(tokenize_text(text, drop_stopwords=True))
        document_term_counters.append(term_counter)
        for term in term_counter:
            document_frequency[term] += 1

    document_count = max(len(document_term_counters), 1)
    idf_by_term = {
        term: float(np.log((1.0 + document_count) / (1.0 + frequency)) + 1.0)
        for term, frequency in document_frequency.items()
    }

    document_norms = np.asarray(
        [
            max(
                np.sqrt(
                    sum(weight * weight for weight in _compute_tfidf_weights(counter, idf_by_term).values())
                ),
                1e-12,
            )
            for counter in document_term_counters
        ],
        dtype=np.float32,
    )
    document_norms.setflags(write=False)

    return CachedClassicalChunkIndex(
        metadata_df=metadata_df,
        noise_mask=cached_index.noise_mask,
        document_term_counters=tuple(document_term_counters),
        idf_by_term=idf_by_term,
        document_norms=document_norms,
    )


def get_cached_classical_chunk_index(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    embedding_model: str,
) -> CachedClassicalChunkIndex:
    """Return cached TF-IDF statistics aligned with the current chunk corpus."""
    return _load_classical_chunk_index_cached(
        *get_file_signature(chunks_path),
        *get_file_signature(chunk_embeddings_path),
        embedding_model,
    )


def retrieve_top_k_classical_ml(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    top_k: int,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Retrieve the top-k chunks with a classical TF-IDF + cosine baseline.

    This baseline avoids embedding similarity and uses sparse term statistics
    directly on the chunk text so the final report can compare a classical
    vector-space model, the naive dense RAG, and the improved retriever.
    """
    if query_text is None and query_id is None:
        raise ValueError("Either query_text or query_id must be provided.")

    cached_index = get_cached_classical_chunk_index(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        embedding_model=embedding_model,
    )
    resolved_query_id, resolved_query_text = resolve_query_reference(
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
        query_text=query_text,
        query_id=query_id,
    )

    filter_mask = build_retrieval_mask(
        metadata_df=cached_index.metadata_df,
        query_text=resolved_query_text,
        enable_metadata_filters=enable_metadata_filters,
        enable_noise_filter=enable_noise_filter,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        noise_mask=cached_index.noise_mask,
    )

    if not filter_mask.any():
        if verbose:
            logger.warning(
                "Classical ML retrieval filters removed all candidates; falling back to the full cached index."
            )
        filter_mask = build_retrieval_mask(
            metadata_df=cached_index.metadata_df,
            query_text=resolved_query_text,
            enable_metadata_filters=False,
            enable_noise_filter=enable_noise_filter,
            company_filter=company_filter,
            fiscal_year_filter=fiscal_year_filter,
            noise_mask=cached_index.noise_mask,
        )

    if not filter_mask.any():
        filter_mask = np.ones(len(cached_index.metadata_df), dtype=bool)

    query_counter = Counter(tokenize_text(resolved_query_text, drop_stopwords=True))
    query_weights = _compute_tfidf_weights(query_counter, cached_index.idf_by_term)
    query_norm = max(
        np.sqrt(sum(weight * weight for weight in query_weights.values())),
        1e-12,
    )

    candidate_indices = np.flatnonzero(filter_mask)
    candidate_metadata_df = cached_index.metadata_df.iloc[candidate_indices].reset_index(drop=True)
    candidate_counters = [cached_index.document_term_counters[index] for index in candidate_indices]
    candidate_norms = cached_index.document_norms[candidate_indices]

    if candidate_metadata_df.empty:
        raise ValueError("No candidates available after classical ML retrieval filtering.")

    candidate_scores: list[float] = []
    for document_counter, document_norm in zip(candidate_counters, candidate_norms):
        dot_product = 0.0
        for term, query_weight in query_weights.items():
            term_frequency = document_counter.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_weight = (1.0 + float(np.log(term_frequency))) * cached_index.idf_by_term[term]
            dot_product += query_weight * doc_weight

        candidate_scores.append(dot_product / (query_norm * max(float(document_norm), 1e-12)))

    results_df = candidate_metadata_df.copy()
    results_df["score"] = np.asarray(candidate_scores, dtype=np.float32)
    results_df = results_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    if verbose:
        logger.info("Classical ML retrieval summary:")
        logger.info("Candidate chunks kept: %s", len(candidate_metadata_df))
        logger.info("Top-k returned      : %s", len(results_df))
        logger.info("Noise filter enabled: %s", enable_noise_filter)
        logger.info("Metadata filters    : %s", enable_metadata_filters)

    return _finalize_baseline_results(
        results_df=results_df,
        resolved_query_id=resolved_query_id,
        resolved_query_text=resolved_query_text,
        retrieval_mode=CLASSICAL_ML_RETRIEVAL_MODE,
    )


def extract_query_filters(query_text: str) -> dict:
    """Extract simple metadata filters from the query text."""
    query_lower = query_text.lower()

    detected_company = None
    for alias, canonical_name in COMPANY_ALIASES.items():
        if alias in query_lower:
            detected_company = canonical_name
            break

    year_match = re.search(r"\b(2022|2023|2024)\b", query_text)
    detected_year = int(year_match.group(1)) if year_match else None

    return {
        "company": detected_company,
        "fiscal_year": detected_year,
    }


def is_noisy_chunk(text: str) -> bool:
    """Detect chunks that are mostly navigational boilerplate."""
    if not isinstance(text, str) or not text.strip():
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    total_words = len(text.split())
    noise_lines = 0

    for line in lines[:20]:
        if line in EXACT_NOISE_LINES:
            noise_lines += 1
            continue

        if any(pattern.match(line) for pattern in NOISE_PATTERNS):
            noise_lines += 1

    if total_words < 40 and noise_lines == len(lines[:20]):
        return True

    return total_words < 80 and noise_lines / max(1, min(len(lines), 20)) >= 0.7


def lexical_overlap_score(query_text: str, chunk_text: str) -> float:
    """Compute a very simple lexical overlap score."""
    query_tokens = set(tokenize_text(query_text, drop_stopwords=True))
    chunk_tokens = set(tokenize_text(chunk_text, drop_stopwords=False))

    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & chunk_tokens)
    return overlap / len(query_tokens)


def detect_query_intents(query_text: str) -> list[str]:
    """Detect coarse query intents to apply a small section-aware bonus."""
    query_lower = query_text.lower()
    intents = []

    if "risk factor" in query_lower or "risks" in query_lower or "risk" in query_lower:
        intents.append("risk_factors")

    if (
        "competition" in query_lower
        or "competitive" in query_lower
        or "competitor" in query_lower
        or "compete" in query_lower
    ):
        intents.append("competition")

    if (
        "business segment" in query_lower
        or "segments" in query_lower
        or "segment" in query_lower
    ):
        intents.append("business_segments")

    if (
        "financial metric" in query_lower
        or "financial metrics" in query_lower
        or "eps" in query_lower
        or "revenue" in query_lower
        or "income" in query_lower
        or "margin" in query_lower
        or "cash flow" in query_lower
    ):
        intents.append("financial_metrics")

    if (
        "artificial intelligence" in query_lower
        or "generative ai" in query_lower
        or "ai-related" in query_lower
        or " ai " in f" {query_lower} "
    ):
        intents.append("ai")

    return list(dict.fromkeys(intents))


def section_prior_score(query_text: str, chunk_text: str) -> float:
    """Compute a lightweight section-aware bonus based on detected query intent."""
    return section_prior_score_with_metadata(
        query_text=query_text,
        chunk_text=chunk_text,
        section_group=None,
    )


def section_prior_score_with_metadata(
    query_text: str,
    chunk_text: str,
    section_group: str | None = None,
) -> float:
    """Combine text-based intent hints with explicit section metadata when available."""
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return 0.0

    intents = detect_query_intents(query_text)
    if not intents:
        return 0.0

    text_lower = chunk_text.lower()
    scores = []

    for intent in intents:
        patterns = SECTION_KEYWORD_PATTERNS.get(intent, [])
        if not patterns:
            continue

        matches = sum(1 for pattern in patterns if pattern.search(text_lower))
        if matches == 0:
            scores.append(0.0)
        else:
            scores.append(matches / len(patterns))

    if not scores:
        text_score = 0.0
    else:
        text_score = max(scores)

    metadata_score = compute_section_intent_score(
        query_intents=intents,
        section_group=section_group,
    )

    return max(text_score, metadata_score)


def build_retrieval_mask(
    metadata_df: pd.DataFrame,
    query_text: str,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    noise_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a boolean mask for metadata and noise filtering."""
    mask = np.ones(len(metadata_df), dtype=bool)
    filters = {"company": None, "fiscal_year": None}
    uploaded_mask = np.zeros(len(metadata_df), dtype=bool)

    if "document_source" in metadata_df.columns:
        uploaded_mask = metadata_df["document_source"].fillna("").eq("uploaded").to_numpy()

    if enable_metadata_filters:
        filters = extract_query_filters(query_text)

    if company_filter is not None:
        filters["company"] = company_filter
    if fiscal_year_filter is not None:
        filters["fiscal_year"] = fiscal_year_filter

    if filters["company"] is not None:
        company_mask = metadata_df["company"].eq(filters["company"]).to_numpy()
        mask &= (company_mask | uploaded_mask)

    if filters["fiscal_year"] is not None and (
        enable_metadata_filters or fiscal_year_filter is not None
    ):
        fiscal_year_mask = metadata_df["fiscal_year"].eq(filters["fiscal_year"]).to_numpy()
        mask &= (fiscal_year_mask | uploaded_mask)

    if enable_noise_filter:
        if noise_mask is None:
            noise_mask = metadata_df["chunk_text"].fillna("").astype(str).apply(is_noisy_chunk).to_numpy()
        noise_mask = np.asarray(noise_mask, dtype=bool)
        if len(noise_mask) != len(metadata_df):
            raise ValueError("noise_mask length does not match metadata_df length.")
        mask &= ~noise_mask

    return mask


def apply_retrieval_filters(
    metadata_df: pd.DataFrame,
    query_text: str,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    noise_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Filter retrieval candidates using query metadata and noise removal."""
    mask = build_retrieval_mask(
        metadata_df=metadata_df,
        query_text=query_text,
        enable_metadata_filters=enable_metadata_filters,
        enable_noise_filter=enable_noise_filter,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        noise_mask=noise_mask,
    )
    return metadata_df.loc[mask].reset_index(drop=True)


def combine_hybrid_scores(
    dense_scores: pd.Series,
    bm25_scores: pd.Series,
    coverage_scores: pd.Series,
    section_scores: pd.Series,
    section_penalties: pd.Series,
    enable_lexical_rerank: bool,
) -> pd.Series:
    """Combine dense similarity with lexical retrieval and section-aware priors."""
    dense_norm = normalize_score_series(dense_scores)
    bm25_norm = normalize_score_series(bm25_scores)

    if enable_lexical_rerank:
        return (
            0.48 * dense_norm
            + 0.27 * bm25_norm
            + 0.15 * coverage_scores
            + 0.20 * section_scores
            - 0.15 * section_penalties
        )

    return 0.75 * dense_norm + 0.25 * section_scores - 0.15 * section_penalties


def combine_rerank_scores(
    candidate_df: pd.DataFrame,
    enable_lexical_rerank: bool,
) -> pd.Series:
    """Final reranking pass over hybrid candidates."""
    lexical_norm = normalize_score_series(candidate_df["lexical_score"])
    bm25_norm = normalize_score_series(candidate_df["bm25_score"])

    if enable_lexical_rerank:
        return (
            0.30 * candidate_df["hybrid_score"]
            + 0.18 * candidate_df["coverage_score"]
            + 0.24 * candidate_df["section_score"]
            + 0.12 * candidate_df["header_score"]
            + 0.10 * lexical_norm
            + 0.10 * bm25_norm
            - 0.10 * candidate_df["numeric_penalty"]
            - 0.15 * candidate_df["section_penalty"]
        )

    return (
        0.60 * candidate_df["hybrid_score"]
        + 0.25 * candidate_df["section_score"]
        + 0.10 * candidate_df["header_score"]
        - 0.10 * candidate_df["numeric_penalty"]
        - 0.15 * candidate_df["section_penalty"]
    )


def safe_console_text(value: object) -> str:
    """Best-effort console-safe rendering for Windows terminals with legacy encodings."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def unique_int_indices_preserving_order(indices: np.ndarray) -> np.ndarray:
    """Deduplicate integer indices while preserving their first-seen order."""
    ordered: list[int] = []
    seen: set[int] = set()

    for value in indices.tolist():
        int_value = int(value)
        if int_value in seen:
            continue
        seen.add(int_value)
        ordered.append(int_value)

    return np.asarray(ordered, dtype=np.int64)


def select_dense_candidate_indices(
    cached_index: CachedChunkIndex,
    query_vector: np.ndarray,
    filter_mask: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Use ANN when available to reduce the candidate set before lexical reranking."""
    filtered_indices = np.flatnonzero(filter_mask)
    filtered_count = len(filtered_indices)

    if filtered_count <= target_size:
        return filtered_indices

    if cached_index.backend != FAISS_BACKEND or cached_index.ann_index is None:
        return filtered_indices

    # For moderately sized filtered sets, exact dense scoring is still simpler and stable.
    if filtered_count <= max(target_size * 2, 400):
        return filtered_indices

    normalized_query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    query_norm = np.linalg.norm(normalized_query, axis=1, keepdims=True)
    query_norm = np.where(query_norm == 0.0, 1.0, query_norm)
    normalized_query = normalized_query / query_norm

    total_vectors = len(filter_mask)
    search_k = min(total_vectors, max(target_size * 4, 128))

    while True:
        _, raw_indices = cached_index.ann_index.search(normalized_query, int(search_k))
        ann_indices = raw_indices[0]
        ann_indices = ann_indices[ann_indices >= 0]
        ann_indices = ann_indices[filter_mask[ann_indices]]
        ann_indices = unique_int_indices_preserving_order(ann_indices)

        if len(ann_indices) >= target_size:
            return ann_indices[:target_size]

        if search_k >= total_vectors:
            break

        search_k = min(total_vectors, search_k * 2)

    return filtered_indices


def retrieve_top_k(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    top_k: int,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    enable_lexical_rerank: bool = True,
    enable_bm25: bool = True,
    enable_reranker: bool = True,
    candidate_pool_size: int | None = None,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        chunks_path: Path to chunks file.
        chunk_embeddings_path: Path to chunk embeddings cache.
        query_embeddings_path: Path to query embeddings cache.
        embedding_model: Name of the embedding model to use.
        top_k: Number of chunks to return.
        query_text: Query text (must be in the query embeddings cache).
        query_id: Optional query ID (alternative to query_text).
        company_filter: Optional explicit company filter.
        fiscal_year_filter: Optional explicit fiscal year filter.
        enable_metadata_filters: Whether to filter by company/year.
        enable_noise_filter: Whether to exclude noisy chunks.
        enable_lexical_rerank: Whether to use lexical scoring features in ranking.
        enable_bm25: Whether to include BM25 scoring in the hybrid ranking stage.
        enable_reranker: Whether to apply a second reranking pass on top candidates.
        candidate_pool_size: Candidate pool size used before final reranking.
        persistent_index_mode: Retrieval index backend selection (auto, persistent, source).
        persistent_index_backend: Persistent artifact backend selection (auto, native, faiss).
        persistent_index_dir: Optional directory where prepared retrieval artifacts are stored.
        rebuild_persistent_index: Force rebuilding the prepared retrieval artifact before search.
        verbose: If True, log detailed progress.

    Returns:
        DataFrame with one row per chunk, containing scores and metadata.
    """
    if query_text is None and query_id is None:
        raise ValueError("Either query_text or query_id must be provided.")

    # Load data
    try:
        cached_index = get_cached_chunk_index(
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            embedding_model=embedding_model,
            persistent_index_mode=persistent_index_mode,
            persistent_index_backend=persistent_index_backend,
            persistent_index_dir=persistent_index_dir,
            rebuild_persistent_index=rebuild_persistent_index,
        )
        vectors = cached_index.vectors
        metadata_df = cached_index.metadata_df
        noise_mask = cached_index.noise_mask
    except Exception as e:
        logger.error(f"Failed to load chunk index data: {e}")
        raise

    try:
        query_vector, resolved_query_id, resolved_query_text = load_query_embedding(
            query_embeddings_path=query_embeddings_path,
            embedding_model=embedding_model,
            query_text=query_text,
            query_id=query_id,
        )
    except Exception as e:
        logger.error(f"Failed to load query embedding: {e}")
        raise

    original_candidate_count = len(metadata_df)

    # Apply filters
    filter_mask = build_retrieval_mask(
        metadata_df=metadata_df,
        query_text=resolved_query_text,
        enable_metadata_filters=enable_metadata_filters,
        enable_noise_filter=enable_noise_filter,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        noise_mask=noise_mask,
    )

    if not filter_mask.any():
        if verbose:
            logger.warning("Filters removed all candidates; falling back to the full cached index.")
        filter_mask = build_retrieval_mask(
            metadata_df=metadata_df,
            query_text=resolved_query_text,
            enable_metadata_filters=False,
            enable_noise_filter=enable_noise_filter,
            company_filter=company_filter,
            fiscal_year_filter=fiscal_year_filter,
            noise_mask=noise_mask,
        )

    if not filter_mask.any():
        filter_mask = np.ones(len(metadata_df), dtype=bool)

    filtered_candidate_count = int(filter_mask.sum())
    if candidate_pool_size is None:
        candidate_pool_size = max(top_k * 8, 30)
    candidate_pool_size = min(candidate_pool_size, filtered_candidate_count)

    dense_candidate_pool_size = min(
        filtered_candidate_count,
        max(candidate_pool_size * 4, top_k * 24, 120),
    )
    selected_indices = select_dense_candidate_indices(
        cached_index=cached_index,
        query_vector=query_vector,
        filter_mask=filter_mask,
        target_size=dense_candidate_pool_size,
    )
    if len(selected_indices) == 0:
        selected_indices = np.flatnonzero(filter_mask)

    filtered_vectors = vectors[selected_indices]
    filtered_metadata_df = metadata_df.iloc[selected_indices].reset_index(drop=True)

    if filtered_metadata_df.empty:
        raise ValueError("No candidates available after retrieval filtering.")

    if verbose:
        filters = extract_query_filters(resolved_query_text)
        logger.info("Retrieval filtering summary:")
        logger.info(f"Original candidate chunks: {original_candidate_count}")
        logger.info(f"Filtered candidate chunks: {filtered_candidate_count}")
        logger.info(f"Dense backend             : {cached_index.backend}")
        logger.info(f"Dense preselected chunks  : {len(filtered_metadata_df)}")
        logger.info(f"Detected company filter   : {company_filter or filters['company']}")
        logger.info(f"Detected fiscal year      : {fiscal_year_filter or filters['fiscal_year']}")
        logger.info(f"Noise filter enabled      : {enable_noise_filter}")
        logger.info(f"Lexical features enabled  : {enable_lexical_rerank}")
        logger.info(f"BM25 enabled              : {enable_bm25}")
        logger.info(f"Reranker enabled          : {enable_reranker}")

    all_scores_df = filtered_metadata_df.copy()
    all_scores_df["score"] = compute_cosine_scores(filtered_vectors, query_vector).astype(np.float32)
    all_scores_df["lexical_score"] = all_scores_df["chunk_text"].apply(
        lambda text: lexical_overlap_score(resolved_query_text, text)
    )
    all_scores_df["coverage_score"] = all_scores_df["chunk_text"].apply(
        lambda text: keyword_coverage_score(resolved_query_text, text)
    )
    all_scores_df["section_score"] = 0.0
    all_scores_df["section_penalty"] = 0.0
    all_scores_df["header_score"] = 0.0
    all_scores_df["numeric_penalty"] = 0.0
    all_scores_df["knowledge_score"] = 0.0
    all_scores_df["knowledge_coverage_score"] = 0.0
    all_scores_df["knowledge_relation_score"] = 0.0

    chunk_knowledge_df = load_chunk_knowledge(chunks_path)
    if not chunk_knowledge_df.empty:
        all_scores_df = all_scores_df.merge(chunk_knowledge_df, on="chunk_id", how="left")
    else:
        all_scores_df["entity_count"] = 0
        all_scores_df["triplet_count"] = 0
        all_scores_df["knowledge_entities"] = [[] for _ in range(len(all_scores_df))]
        all_scores_df["knowledge_entity_types"] = [[] for _ in range(len(all_scores_df))]
        all_scores_df["knowledge_triplets"] = [[] for _ in range(len(all_scores_df))]
        all_scores_df["knowledge_relations"] = [[] for _ in range(len(all_scores_df))]
        all_scores_df["knowledge_competitors"] = [[] for _ in range(len(all_scores_df))]
        all_scores_df["knowledge_text"] = ""
        all_scores_df["knowledge_entities_preview"] = ""
        all_scores_df["knowledge_triplets_preview"] = ""

    for column in [
        "entity_count",
        "triplet_count",
        "knowledge_entities",
        "knowledge_entity_types",
        "knowledge_triplets",
        "knowledge_relations",
        "knowledge_competitors",
        "knowledge_text",
        "knowledge_entities_preview",
        "knowledge_triplets_preview",
    ]:
        if column not in all_scores_df.columns:
            all_scores_df[column] = None

    all_scores_df["entity_count"] = all_scores_df["entity_count"].fillna(0).astype(int)
    all_scores_df["triplet_count"] = all_scores_df["triplet_count"].fillna(0).astype(int)
    all_scores_df["knowledge_text"] = all_scores_df["knowledge_text"].fillna("").astype(str)
    all_scores_df["knowledge_entities_preview"] = (
        all_scores_df["knowledge_entities_preview"].fillna("").astype(str)
    )
    all_scores_df["knowledge_triplets_preview"] = (
        all_scores_df["knowledge_triplets_preview"].fillna("").astype(str)
    )
    all_scores_df["knowledge_entities"] = all_scores_df["knowledge_entities"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    all_scores_df["knowledge_triplets"] = all_scores_df["knowledge_triplets"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    all_scores_df["knowledge_relations"] = all_scores_df["knowledge_relations"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    all_scores_df["knowledge_competitors"] = all_scores_df["knowledge_competitors"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    all_scores_df["knowledge_coverage_score"] = all_scores_df["knowledge_text"].apply(
        lambda text: keyword_coverage_score(resolved_query_text, text)
    )
    all_scores_df["knowledge_relation_score"] = all_scores_df["knowledge_relations"].apply(
        lambda relations: relation_intent_score(resolved_query_text, relations)
    )
    all_scores_df["knowledge_score"] = (
        0.70 * all_scores_df["knowledge_coverage_score"]
        + 0.30 * all_scores_df["knowledge_relation_score"]
    ).clip(upper=1.0)

    if enable_bm25 and enable_lexical_rerank:
        bm25_documents = (
            all_scores_df["section_title"].fillna("").astype(str)
            + "\n"
            + all_scores_df["chunk_text"].fillna("").astype(str)
        )
        all_scores_df["bm25_score"] = compute_bm25_scores(resolved_query_text, bm25_documents)
    else:
        all_scores_df["bm25_score"] = 0.0

    candidate_pool_size = min(candidate_pool_size, len(all_scores_df))

    all_scores_df["preliminary_score"] = (
        0.55 * normalize_score_series(all_scores_df["score"])
        + 0.30 * normalize_score_series(all_scores_df["bm25_score"])
        + 0.15 * all_scores_df["coverage_score"]
        + 0.08 * all_scores_df["knowledge_score"]
    )

    preliminary_pool_size = min(
        len(all_scores_df),
        max(candidate_pool_size * 2, top_k * 12, 60),
    )

    candidate_df = (
        all_scores_df.sort_values("preliminary_score", ascending=False)
        .head(preliminary_pool_size)
        .copy()
    )

    query_intents = detect_query_intents(resolved_query_text)
    candidate_df["section_score"] = candidate_df.apply(
        lambda row: section_prior_score_with_metadata(
            query_text=resolved_query_text,
            chunk_text=row["chunk_text"],
            section_group=row.get("section_group"),
        ),
        axis=1,
    )
    candidate_df["section_penalty"] = candidate_df["section_group"].apply(
        lambda value: compute_section_mismatch_penalty(
            query_intents=query_intents,
            section_group=value,
        )
    )
    candidate_df["header_score"] = candidate_df.apply(
        lambda row: header_keyword_score(
            query_text=resolved_query_text,
            chunk_text=row["chunk_text"],
            section_title=row.get("section_title"),
        ),
        axis=1,
    )
    candidate_df["numeric_penalty"] = candidate_df.apply(
        lambda row: numeric_density_penalty(
            query_text=resolved_query_text,
            chunk_text=row["chunk_text"],
            section_group=row.get("section_group"),
        ),
        axis=1,
    )
    candidate_df["hybrid_score"] = combine_hybrid_scores(
        dense_scores=candidate_df["score"],
        bm25_scores=candidate_df["bm25_score"],
        coverage_scores=candidate_df["coverage_score"],
        section_scores=candidate_df["section_score"],
        section_penalties=candidate_df["section_penalty"],
        enable_lexical_rerank=enable_lexical_rerank and enable_bm25,
    )
    candidate_df["hybrid_score"] = candidate_df["hybrid_score"] + 0.08 * candidate_df["knowledge_score"]

    if enable_reranker:
        candidate_df["rerank_score"] = combine_rerank_scores(
            candidate_df=candidate_df,
            enable_lexical_rerank=enable_lexical_rerank and enable_bm25,
        )
        candidate_df["rerank_score"] = candidate_df["rerank_score"] + 0.10 * candidate_df["knowledge_score"]
        results_df = (
            candidate_df.sort_values("rerank_score", ascending=False)
            .head(top_k)
            .copy()
        )
    else:
        candidate_df["rerank_score"] = candidate_df["hybrid_score"]
        results_df = (
            candidate_df.sort_values("hybrid_score", ascending=False)
            .head(top_k)
            .copy()
        )

    results_df["final_score"] = results_df["rerank_score"]
    results_df = results_df.reset_index(drop=True)

    # Add query metadata
    results_df.insert(0, "query_id", resolved_query_id)
    results_df.insert(1, "query_text", resolved_query_text)

    return results_df


def retrieve_top_k_with_mode(
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    top_k: int,
    retrieval_mode: str = IMPROVED_RETRIEVAL_MODE,
    query_text: Optional[str] = None,
    query_id: Optional[str] = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    enable_metadata_filters: bool = True,
    enable_noise_filter: bool = True,
    enable_lexical_rerank: bool = True,
    enable_bm25: bool = True,
    enable_reranker: bool = True,
    candidate_pool_size: int | None = None,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Dispatch retrieval to the requested pedagogical mode."""
    normalized_retrieval_mode = normalize_retrieval_mode(retrieval_mode)

    if normalized_retrieval_mode == CLASSICAL_ML_RETRIEVAL_MODE:
        return retrieve_top_k_classical_ml(
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            query_embeddings_path=query_embeddings_path,
            embedding_model=embedding_model,
            top_k=top_k,
            query_text=query_text,
            query_id=query_id,
            company_filter=company_filter,
            fiscal_year_filter=fiscal_year_filter,
            enable_metadata_filters=enable_metadata_filters,
            enable_noise_filter=enable_noise_filter,
            verbose=verbose,
        )

    if normalized_retrieval_mode == NAIVE_RETRIEVAL_MODE:
        return retrieve_top_k_naive(
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            query_embeddings_path=query_embeddings_path,
            embedding_model=embedding_model,
            top_k=top_k,
            query_text=query_text,
            query_id=query_id,
            company_filter=company_filter,
            fiscal_year_filter=fiscal_year_filter,
            enable_metadata_filters=enable_metadata_filters,
            enable_noise_filter=enable_noise_filter,
            persistent_index_mode=persistent_index_mode,
            persistent_index_backend=persistent_index_backend,
            persistent_index_dir=persistent_index_dir,
            rebuild_persistent_index=rebuild_persistent_index,
            verbose=verbose,
        )

    results_df = retrieve_top_k(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
        top_k=top_k,
        query_text=query_text,
        query_id=query_id,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        enable_metadata_filters=enable_metadata_filters,
        enable_noise_filter=enable_noise_filter,
        enable_lexical_rerank=enable_lexical_rerank,
        enable_bm25=enable_bm25,
        enable_reranker=enable_reranker,
        candidate_pool_size=candidate_pool_size,
        persistent_index_mode=persistent_index_mode,
        persistent_index_backend=persistent_index_backend,
        persistent_index_dir=persistent_index_dir,
        rebuild_persistent_index=rebuild_persistent_index,
        verbose=verbose,
    )
    results_df = results_df.copy()
    results_df["retrieval_mode"] = IMPROVED_RETRIEVAL_MODE
    return results_df


def print_results(results_df: pd.DataFrame) -> None:
    """Print retrieval results in a readable format."""
    if results_df.empty:
        print("No retrieval results found.")
        return

    query_text = results_df.iloc[0]["query_text"]
    print("\nRetrieval completed.")
    print(f"Query: {safe_console_text(query_text)}")
    print(f"Top-k results: {len(results_df)}\n")

    for rank, row in enumerate(results_df.itertuples(index=False), start=1):
        print(f"[Rank {rank}]")
        print(f"Cosine score : {row.score:.4f}")
        print(f"BM25 score   : {getattr(row, 'bm25_score', 0.0):.4f}")
        print(f"Lexical score: {row.lexical_score:.4f}")
        print(f"Coverage     : {getattr(row, 'coverage_score', 0.0):.4f}")
        print(f"Section score: {row.section_score:.4f}")
        print(f"Knowledge    : {getattr(row, 'knowledge_score', 0.0):.4f}")
        print(f"Header score : {getattr(row, 'header_score', 0.0):.4f}")
        print(f"Hybrid score : {getattr(row, 'hybrid_score', 0.0):.4f}")
        print(f"Rerank score : {getattr(row, 'rerank_score', 0.0):.4f}")
        print(f"Final score  : {row.final_score:.4f}")
        print(f"Chunk ID     : {row.chunk_id}")
        print(f"Company      : {row.company}")
        print(f"Fiscal year  : {row.fiscal_year}")
        print(f"Pages        : {row.page_start}-{row.page_end}")
        print(
            "Section      : "
            f"{getattr(row, 'section_group', None) or 'unknown'}"
            f" | {getattr(row, 'section_title', None) or 'unknown'}"
        )
        if getattr(row, "knowledge_entities_preview", ""):
            print(f"Entities     : {safe_console_text(row.knowledge_entities_preview)}")
        if getattr(row, "knowledge_triplets_preview", ""):
            print(f"Triplets     : {safe_console_text(row.knowledge_triplets_preview)}")
        print("Text preview:")
        preview = row.chunk_text[:300].replace("\n", " ")
        print(safe_console_text(preview + ("..." if len(row.chunk_text) > 300 else "")))
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a query.")
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to chunks file.",
    )
    parser.add_argument(
        "--chunk_embeddings_path",
        type=str,
        default="data/embeddings/chunk_embeddings.parquet",
        help="Path to chunk embeddings cache.",
    )
    parser.add_argument(
        "--query_embeddings_path",
        type=str,
        default="data/embeddings/query_embeddings.parquet",
        help="Path to query embeddings cache.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve.",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default=IMPROVED_RETRIEVAL_MODE,
        choices=[
            CLASSICAL_ML_RETRIEVAL_MODE,
            NAIVE_RETRIEVAL_MODE,
            IMPROVED_RETRIEVAL_MODE,
        ],
        help="Retrieval mode to run: classical ML baseline, naive dense baseline, or improved retriever.",
    )
    parser.add_argument(
        "--query_text",
        type=str,
        default=None,
        help="Query text. Must already exist in query_embeddings cache.",
    )
    parser.add_argument(
        "--query_id",
        type=str,
        default=None,
        help="Optional query_id if you want to retrieve by id.",
    )
    parser.add_argument(
        "--company_filter",
        type=str,
        default=None,
        help="Optional explicit company filter overriding query inference.",
    )
    parser.add_argument(
        "--fiscal_year_filter",
        type=int,
        default=None,
        help="Optional explicit fiscal year filter overriding query inference.",
    )
    parser.add_argument(
        "--disable_metadata_filters",
        action="store_true",
        help="Disable company/year filtering inferred from the query text.",
    )
    parser.add_argument(
        "--disable_noise_filter",
        action="store_true",
        help="Disable removal of noisy chunks such as exhibits or signatures.",
    )
    parser.add_argument(
        "--disable_lexical_rerank",
        action="store_true",
        help="Disable lexical scoring features in hybrid retrieval.",
    )
    parser.add_argument(
        "--disable_bm25",
        action="store_true",
        help="Disable BM25 scoring in the hybrid retrieval stage.",
    )
    parser.add_argument(
        "--disable_reranker",
        action="store_true",
        help="Disable the final reranking pass and return top hybrid candidates directly.",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=None,
        help="Optional number of hybrid candidates kept before final reranking.",
    )
    parser.add_argument(
        "--persistent_index_mode",
        type=str,
        default="auto",
        choices=["auto", "persistent", "source"],
        help="Use a prepared retrieval artifact when available.",
    )
    parser.add_argument(
        "--persistent_index_backend",
        type=str,
        default="auto",
        choices=["auto", "native", "faiss"],
        help="Backend used by the prepared retrieval artifact.",
    )
    parser.add_argument(
        "--persistent_index_dir",
        type=str,
        default=None,
        help="Optional directory for prepared retrieval artifacts.",
    )
    parser.add_argument(
        "--rebuild_persistent_index",
        action="store_true",
        help="Force rebuilding the prepared retrieval artifact before retrieval.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for disabling BM25 and the final reranking pass.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save results (.parquet or .csv).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and report counts without running retrieval.",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Validate arguments
    if args.query_text is None and args.query_id is None:
        logger.error("Either --query_text or --query_id must be provided.")
        sys.exit(1)

    # Convert paths
    chunks_path = Path(args.chunks_path)
    chunk_embeddings_path = Path(args.chunk_embeddings_path)
    query_embeddings_path = Path(args.query_embeddings_path)
    persistent_index_dir = Path(args.persistent_index_dir) if args.persistent_index_dir else None

    # Dry-run: only load data and print stats
    if args.dry_run:
        try:
            vectors, metadata_df = load_chunk_index_data(
                chunks_path=chunks_path,
                chunk_embeddings_path=chunk_embeddings_path,
                embedding_model=args.embedding_model,
                persistent_index_mode=args.persistent_index_mode,
                persistent_index_backend=args.persistent_index_backend,
                persistent_index_dir=persistent_index_dir,
                rebuild_persistent_index=args.rebuild_persistent_index,
            )
            _, resolved_query_id, resolved_query_text = load_query_embedding(
                query_embeddings_path=query_embeddings_path,
                embedding_model=args.embedding_model,
                query_text=args.query_text,
                query_id=args.query_id,
            )
            logger.info("Dry run completed.")
            logger.info(f"Total chunks: {len(metadata_df)}")
            logger.info(f"Persistent index mode: {args.persistent_index_mode}")
            logger.info(f"Persistent index backend: {args.persistent_index_backend}")
            logger.info(f"Query: {resolved_query_text} (ID: {resolved_query_id})")
        except Exception as e:
            logger.error(f"Dry run failed: {e}")
            sys.exit(1)
        sys.exit(0)

    # Run retrieval
    try:
        results_df = retrieve_top_k_with_mode(
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            query_embeddings_path=query_embeddings_path,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            query_text=args.query_text,
            query_id=args.query_id,
            company_filter=args.company_filter,
            fiscal_year_filter=args.fiscal_year_filter,
            enable_metadata_filters=not args.disable_metadata_filters,
            enable_noise_filter=not args.disable_noise_filter,
            enable_lexical_rerank=not args.disable_lexical_rerank,
            enable_bm25=not (args.disable_bm25 or args.fast),
            enable_reranker=not (args.disable_reranker or args.fast),
            candidate_pool_size=args.candidate_pool_size,
            persistent_index_mode=args.persistent_index_mode,
            persistent_index_backend=args.persistent_index_backend,
            persistent_index_dir=persistent_index_dir,
            rebuild_persistent_index=args.rebuild_persistent_index,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)

    print_results(results_df)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".parquet":
            results_df.to_parquet(output_path, index=False)
        elif output_path.suffix.lower() == ".csv":
            results_df.to_csv(output_path, index=False)
        else:
            logger.error("output_path must end with .parquet or .csv")
            sys.exit(1)
        logger.info(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
