from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.extraction.competitor_analysis import (
    build_clean_competitor_summary,
    build_competitor_mentions,
    build_competitor_summary,
    build_new_competitors_by_year,
)

ENTITY_COLUMNS = [
    "chunk_id",
    "entity_text",
    "entity_type",
    "confidence",
    "source_doc_id",
    "year",
    "company",
    "page_start",
    "page_end",
    "created_at",
]

TRIPLET_COLUMNS = [
    "chunk_id",
    "entity_a",
    "relation",
    "entity_b",
    "year",
    "company",
    "doc_id",
    "page_start",
    "page_end",
    "created_at",
]

CHUNK_FACT_COLUMNS = [
    "chunk_id",
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
]

COMPETITOR_MENTION_COLUMNS = [
    "chunk_id",
    "source_doc_id",
    "source_company",
    "year",
    "competitor_name",
    "page_start",
    "page_end",
    "has_competition_risk_signal",
    "explicit_competes_with",
    "mention_source",
    "chunk_text",
    "chunk_text_preview",
]

COMPETITOR_SUMMARY_COLUMNS = [
    "source_company",
    "year",
    "competitor_name",
    "mention_count",
    "competition_risk_mentions",
    "explicit_competes_with_count",
    "first_page_seen",
    "last_page_seen",
]

NEW_COMPETITOR_COLUMNS = ["source_company", "year", "new_competitor_name"]


@dataclass(frozen=True)
class KnowledgeArtifacts:
    entities_df: pd.DataFrame
    triplets_df: pd.DataFrame
    chunk_facts_df: pd.DataFrame
    competitor_mentions_df: pd.DataFrame
    competitor_summary_df: pd.DataFrame
    competitor_summary_clean_df: pd.DataFrame
    new_competitors_df: pd.DataFrame
    new_competitors_clean_df: pd.DataFrame


def get_default_entities_path(chunks_path: Path) -> Path:
    """Infer the standard entities artifact path from the chunks path."""
    return chunks_path.resolve().parent / "entities.parquet"


def get_default_triplets_path(chunks_path: Path) -> Path:
    """Infer the standard triplets artifact path from the chunks path."""
    return chunks_path.resolve().parent / "triplets.parquet"


def empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ENTITY_COLUMNS)


def empty_triplets_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRIPLET_COLUMNS)


def empty_chunk_facts_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CHUNK_FACT_COLUMNS)


def empty_competitor_mentions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPETITOR_MENTION_COLUMNS)


def empty_competitor_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPETITOR_SUMMARY_COLUMNS)


def empty_new_competitors_df() -> pd.DataFrame:
    return pd.DataFrame(columns=NEW_COMPETITOR_COLUMNS)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from parquet or csv."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_optional_dataframe(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    """Load an optional dataframe and normalize missing columns."""
    if not path.exists():
        return pd.DataFrame(columns=expected_columns)

    df = load_dataframe(path).copy()
    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    return df[expected_columns]


def get_optional_file_signature(path: Path) -> tuple[str, int, int, int]:
    """Return a stable cache key for present or missing optional artifacts."""
    resolved_path = path.resolve()
    if not resolved_path.exists():
        return str(resolved_path), 0, 0, 0

    stat = resolved_path.stat()
    return str(resolved_path), 1, stat.st_mtime_ns, stat.st_size


def dedupe_text_values(values: list[object]) -> list[str]:
    """Deduplicate string values while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        deduped.append(text)
        seen.add(key)

    return deduped


def build_triplet_preview(entity_a: object, relation: object, entity_b: object) -> str:
    """Format one relation triplet for display and retrieval hints."""
    a = str(entity_a).strip()
    r = str(relation).strip()
    b = str(entity_b).strip()

    if not a or not r or not b:
        return ""

    return f"{a} [{r}] {b}"


def build_chunk_facts(
    entities_df: pd.DataFrame,
    triplets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate extracted knowledge per chunk."""
    if entities_df.empty and triplets_df.empty:
        return empty_chunk_facts_df()

    rows_by_chunk: dict[str, dict[str, object]] = {}

    if not entities_df.empty:
        for chunk_id, group_df in entities_df.groupby("chunk_id"):
            entity_texts = dedupe_text_values(group_df["entity_text"].tolist())
            entity_types = dedupe_text_values(group_df["entity_type"].tolist())

            row = rows_by_chunk.setdefault(
                str(chunk_id),
                {
                    "chunk_id": str(chunk_id),
                    "entity_count": 0,
                    "triplet_count": 0,
                    "knowledge_entities": [],
                    "knowledge_entity_types": [],
                    "knowledge_triplets": [],
                    "knowledge_relations": [],
                    "knowledge_competitors": [],
                    "knowledge_text": "",
                    "knowledge_entities_preview": "",
                    "knowledge_triplets_preview": "",
                },
            )
            row["entity_count"] = len(entity_texts)
            row["knowledge_entities"] = entity_texts
            row["knowledge_entity_types"] = entity_types
            row["knowledge_entities_preview"] = " | ".join(entity_texts[:5])

    if not triplets_df.empty:
        for chunk_id, group_df in triplets_df.groupby("chunk_id"):
            triplet_previews = dedupe_text_values(
                [
                    build_triplet_preview(row.entity_a, row.relation, row.entity_b)
                    for row in group_df.itertuples(index=False)
                ]
            )
            relation_labels = dedupe_text_values(group_df["relation"].tolist())

            competitor_names = dedupe_text_values(
                group_df.loc[group_df["relation"] == "COMPETES_WITH", "entity_b"].tolist()
            )

            row = rows_by_chunk.setdefault(
                str(chunk_id),
                {
                    "chunk_id": str(chunk_id),
                    "entity_count": 0,
                    "triplet_count": 0,
                    "knowledge_entities": [],
                    "knowledge_entity_types": [],
                    "knowledge_triplets": [],
                    "knowledge_relations": [],
                    "knowledge_competitors": [],
                    "knowledge_text": "",
                    "knowledge_entities_preview": "",
                    "knowledge_triplets_preview": "",
                },
            )
            row["triplet_count"] = len(triplet_previews)
            row["knowledge_triplets"] = triplet_previews
            row["knowledge_relations"] = relation_labels
            row["knowledge_competitors"] = competitor_names
            row["knowledge_triplets_preview"] = " | ".join(triplet_previews[:4])

    rows: list[dict[str, object]] = []
    for row in rows_by_chunk.values():
        knowledge_text_parts = (
            list(row["knowledge_entities"])
            + list(row["knowledge_triplets"])
            + list(row["knowledge_competitors"])
        )
        row["knowledge_text"] = " ".join(knowledge_text_parts)
        rows.append(row)

    return pd.DataFrame(rows, columns=CHUNK_FACT_COLUMNS).sort_values("chunk_id").reset_index(
        drop=True
    )


def enrich_competitor_mentions_with_chunk_text(
    mentions_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach chunk text evidence to competitor mentions."""
    if mentions_df.empty:
        return empty_competitor_mentions_df()

    chunk_context_df = chunks_df[["chunk_id", "chunk_text"]].copy()
    enriched_df = mentions_df.merge(chunk_context_df, on="chunk_id", how="left")
    enriched_df["chunk_text"] = enriched_df["chunk_text"].fillna("").astype(str)
    enriched_df["chunk_text_preview"] = enriched_df["chunk_text"].apply(
        lambda text: text[:280].replace("\n", " ") + ("..." if len(text) > 280 else "")
    )

    return enriched_df[COMPETITOR_MENTION_COLUMNS].reset_index(drop=True)


@lru_cache(maxsize=8)
def _load_knowledge_artifacts_cached(
    chunks_path_str: str,
    chunks_exists: int,
    chunks_mtime_ns: int,
    chunks_size: int,
    entities_path_str: str,
    entities_exists: int,
    entities_mtime_ns: int,
    entities_size: int,
    triplets_path_str: str,
    triplets_exists: int,
    triplets_mtime_ns: int,
    triplets_size: int,
) -> KnowledgeArtifacts:
    """Load extracted knowledge artifacts once per source signature."""
    _ = (
        chunks_exists,
        chunks_mtime_ns,
        chunks_size,
        entities_exists,
        entities_mtime_ns,
        entities_size,
        triplets_exists,
        triplets_mtime_ns,
        triplets_size,
    )

    chunks_df = load_dataframe(Path(chunks_path_str))
    entities_df = load_optional_dataframe(Path(entities_path_str), ENTITY_COLUMNS)
    triplets_df = load_optional_dataframe(Path(triplets_path_str), TRIPLET_COLUMNS)

    chunk_facts_df = build_chunk_facts(entities_df=entities_df, triplets_df=triplets_df)

    if entities_df.empty and triplets_df.empty:
        competitor_mentions_df = empty_competitor_mentions_df()
        competitor_summary_df = empty_competitor_summary_df()
        competitor_summary_clean_df = empty_competitor_summary_df()
        new_competitors_df = empty_new_competitors_df()
        new_competitors_clean_df = empty_new_competitors_df()
    else:
        competitor_mentions_df = build_competitor_mentions(
            chunks_df=chunks_df,
            entities_df=entities_df,
            triplets_df=triplets_df,
        )
        competitor_mentions_df = enrich_competitor_mentions_with_chunk_text(
            mentions_df=competitor_mentions_df,
            chunks_df=chunks_df,
        )
        competitor_summary_df = build_competitor_summary(competitor_mentions_df)
        competitor_summary_clean_df = build_clean_competitor_summary(competitor_summary_df)
        new_competitors_df = build_new_competitors_by_year(competitor_summary_df)
        new_competitors_clean_df = build_new_competitors_by_year(competitor_summary_clean_df)

    return KnowledgeArtifacts(
        entities_df=entities_df,
        triplets_df=triplets_df,
        chunk_facts_df=chunk_facts_df,
        competitor_mentions_df=competitor_mentions_df,
        competitor_summary_df=competitor_summary_df,
        competitor_summary_clean_df=competitor_summary_clean_df,
        new_competitors_df=new_competitors_df,
        new_competitors_clean_df=new_competitors_clean_df,
    )


def get_knowledge_artifacts(
    chunks_path: Path,
    entities_path: Path | None = None,
    triplets_path: Path | None = None,
) -> KnowledgeArtifacts:
    """Load extracted entities/triplets plus derived knowledge views."""
    resolved_entities_path = entities_path or get_default_entities_path(chunks_path)
    resolved_triplets_path = triplets_path or get_default_triplets_path(chunks_path)

    return _load_knowledge_artifacts_cached(
        *get_optional_file_signature(chunks_path),
        *get_optional_file_signature(resolved_entities_path),
        *get_optional_file_signature(resolved_triplets_path),
    )
