from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.grounded_qa import build_context_block
from src.extraction.knowledge_base import get_knowledge_artifacts
from src.generation.rag_answer import DEFAULT_EMBEDDING_MODEL, ensure_query_embeddings_cached
from src.retrieval.retrieve import (
    IMPROVED_RETRIEVAL_MODE,
    normalize_retrieval_mode,
    retrieve_top_k_with_mode,
)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
    chunk_embeddings_path: Path = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
    query_embeddings_path: Path = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"
    knowledge_chunks_path: Path | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 5
    retrieval_mode: str = IMPROVED_RETRIEVAL_MODE
    persistent_index_mode: str = "auto"


def normalize_company_key(value: str | None) -> str | None:
    """Normalize company text for tolerant matching across UI, eval, and tool inputs."""
    if value is None:
        return None
    normalized = "".join(ch.lower() for ch in str(value) if ch.isalnum())
    return normalized or None


def _serialize_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    safe_df = df.copy()
    keep_columns = [column for column in columns if column in safe_df.columns]
    if not keep_columns:
        return []
    safe_df = safe_df[keep_columns].copy()
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return safe_df.to_dict(orient="records")


def search_financial_corpus(
    runtime_config: AgentRuntimeConfig,
    question: str | None = None,
    query: str | None = None,
    top_k: int | None = None,
    retrieval_mode: str | None = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
) -> dict[str, Any]:
    """Search the local corpus with the existing retrieval stack and return grounded chunks."""
    resolved_question = (question or query or "").strip()
    if resolved_question == "":
        raise ValueError("search_financial_corpus requires a non-empty question or query.")

    resolved_top_k = max(1, min(int(top_k or runtime_config.top_k), 8))
    resolved_retrieval_mode = normalize_retrieval_mode(
        retrieval_mode or runtime_config.retrieval_mode
    )

    ensure_query_embeddings_cached(
        query_texts=[resolved_question],
        query_embeddings_path=runtime_config.query_embeddings_path,
        embedding_model_name=runtime_config.embedding_model,
    )

    retrieval_results_df = retrieve_top_k_with_mode(
        chunks_path=runtime_config.chunks_path,
        chunk_embeddings_path=runtime_config.chunk_embeddings_path,
        query_embeddings_path=runtime_config.query_embeddings_path,
        embedding_model=runtime_config.embedding_model,
        top_k=resolved_top_k,
        retrieval_mode=resolved_retrieval_mode,
        query_text=resolved_question,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        enable_metadata_filters=True,
        enable_noise_filter=True,
        enable_lexical_rerank=True,
        enable_bm25=True,
        enable_reranker=True,
        persistent_index_mode=runtime_config.persistent_index_mode,
        verbose=False,
    )

    serialized_results = _serialize_records(
        retrieval_results_df,
        columns=[
            "chunk_id",
            "doc_id",
            "company",
            "fiscal_year",
            "file_name",
            "document_source",
            "page_start",
            "page_end",
            "section_group",
            "section_title",
            "score",
            "final_score",
            "knowledge_score",
            "chunk_text",
        ],
    )

    return {
        "question": resolved_question,
        "top_k": resolved_top_k,
        "retrieval_mode": resolved_retrieval_mode,
        "company_filter": company_filter,
        "fiscal_year_filter": fiscal_year_filter,
        "n_results": len(serialized_results),
        "retrieved_context": build_context_block(retrieval_results_df),
        "retrieval_results": serialized_results,
        "suggested_citations": [
            {"doc_id": row["doc_id"], "page": row["page_start"]}
            for row in serialized_results
            if row.get("doc_id") is not None and row.get("page_start") is not None
        ],
    }


def lookup_knowledge_graph(
    runtime_config: AgentRuntimeConfig,
    company: str | None = None,
    fiscal_year: int | None = None,
    entity_type: str | None = None,
    max_rows: int = 10,
) -> dict[str, Any]:
    """Inspect extracted entities and relationships for a company/year slice."""
    knowledge_chunks_path = runtime_config.knowledge_chunks_path or runtime_config.chunks_path
    knowledge_artifacts = get_knowledge_artifacts(chunks_path=knowledge_chunks_path)
    entities_df = knowledge_artifacts.entities_df.copy()
    triplets_df = knowledge_artifacts.triplets_df.copy()

    company_key = normalize_company_key(company)
    if company_key is not None and "company" in entities_df.columns:
        entities_df = entities_df[
            entities_df["company"].fillna("").astype(str).apply(normalize_company_key) == company_key
        ].copy()
    if company_key is not None and "company" in triplets_df.columns:
        triplets_df = triplets_df[
            triplets_df["company"].fillna("").astype(str).apply(normalize_company_key) == company_key
        ].copy()

    if fiscal_year is not None and "year" in entities_df.columns:
        entities_df = entities_df[pd.to_numeric(entities_df["year"], errors="coerce") == fiscal_year]
    if fiscal_year is not None and "year" in triplets_df.columns:
        triplets_df = triplets_df[pd.to_numeric(triplets_df["year"], errors="coerce") == fiscal_year]

    if entity_type is not None and entity_type.strip() and "entity_type" in entities_df.columns:
        normalized_entity_type = entity_type.strip().lower()
        entities_df = entities_df[
            entities_df["entity_type"].fillna("").astype(str).str.lower() == normalized_entity_type
        ].copy()

    top_entities_df = (
        entities_df.groupby(["entity_text", "entity_type"], as_index=False)
        .agg(
            mention_count=("chunk_id", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .sort_values(["mention_count", "entity_text"], ascending=[False, True])
        .head(max_rows)
    )
    top_triplets_df = (
        triplets_df.groupby(["entity_a", "relation", "entity_b"], as_index=False)
        .agg(
            mention_count=("chunk_id", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .sort_values(["mention_count", "relation"], ascending=[False, True])
        .head(max_rows)
    )

    return {
        "company": company,
        "fiscal_year": fiscal_year,
        "entity_type": entity_type,
        "entity_rows": int(len(entities_df)),
        "triplet_rows": int(len(triplets_df)),
        "top_entities": _serialize_records(
            top_entities_df,
            ["entity_text", "entity_type", "mention_count", "first_year", "last_year"],
        ),
        "top_triplets": _serialize_records(
            top_triplets_df,
            ["entity_a", "relation", "entity_b", "mention_count", "first_year", "last_year"],
        ),
    }


def get_competitor_evidence(
    runtime_config: AgentRuntimeConfig,
    company: str | None = None,
    fiscal_year: int | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Return competitor summaries and evidence rows derived from extracted knowledge."""
    knowledge_chunks_path = runtime_config.knowledge_chunks_path or runtime_config.chunks_path
    knowledge_artifacts = get_knowledge_artifacts(chunks_path=knowledge_chunks_path)
    summary_df = knowledge_artifacts.competitor_summary_clean_df.copy()
    mentions_df = knowledge_artifacts.competitor_mentions_df.copy()

    company_key = normalize_company_key(company)
    if company_key is not None and "source_company" in summary_df.columns:
        summary_df = summary_df[
            summary_df["source_company"].fillna("").astype(str).apply(normalize_company_key) == company_key
        ].copy()
    if company_key is not None and "source_company" in mentions_df.columns:
        mentions_df = mentions_df[
            mentions_df["source_company"].fillna("").astype(str).apply(normalize_company_key) == company_key
        ].copy()

    if fiscal_year is not None and "year" in summary_df.columns:
        summary_df = summary_df[pd.to_numeric(summary_df["year"], errors="coerce") == fiscal_year]
    if fiscal_year is not None and "year" in mentions_df.columns:
        mentions_df = mentions_df[pd.to_numeric(mentions_df["year"], errors="coerce") == fiscal_year]

    summary_df = summary_df.sort_values(
        ["mention_count", "competitor_name"], ascending=[False, True]
    ).head(top_n)
    mentions_df = mentions_df.sort_values(
        ["year", "competitor_name", "page_start"], ascending=[False, True, True]
    ).head(top_n)

    return {
        "company": company,
        "fiscal_year": fiscal_year,
        "summary_rows": _serialize_records(
            summary_df,
            [
                "source_company",
                "year",
                "competitor_name",
                "mention_count",
                "competition_risk_mentions",
                "explicit_competes_with_count",
                "first_page_seen",
                "last_page_seen",
            ],
        ),
        "evidence_rows": _serialize_records(
            mentions_df,
            [
                "source_company",
                "year",
                "competitor_name",
                "source_doc_id",
                "page_start",
                "page_end",
                "mention_source",
                "has_competition_risk_signal",
                "explicit_competes_with",
                "chunk_text_preview",
            ],
        ),
    }


def build_local_tool_specs() -> list[dict[str, Any]]:
    """Return OpenAI function-calling schemas for the local analyst tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_financial_corpus",
                "description": "Search the local financial filing corpus, including session uploads when available, and return the most relevant chunks with source metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "retrieval_mode": {
                            "type": "string",
                            "enum": ["classical_ml", "naive", "improved"],
                        },
                        "company_filter": {"type": "string"},
                        "fiscal_year_filter": {"type": "integer"},
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_knowledge_graph",
                "description": "Inspect extracted entities and triplets from the local knowledge graph for a company or year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "fiscal_year": {"type": "integer"},
                        "entity_type": {"type": "string"},
                        "max_rows": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_competitor_evidence",
                "description": "Return competitor summaries and source evidence from extracted local filings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "fiscal_year": {"type": "integer"},
                        "top_n": {"type": "integer"},
                    },
                },
            },
        },
    ]


def build_local_tool_registry(
    runtime_config: AgentRuntimeConfig,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Bind the local tool names to runtime-aware Python callables."""
    return {
        "search_financial_corpus": lambda **kwargs: search_financial_corpus(
            runtime_config=runtime_config,
            **kwargs,
        ),
        "lookup_knowledge_graph": lambda **kwargs: lookup_knowledge_graph(
            runtime_config=runtime_config,
            **kwargs,
        ),
        "get_competitor_evidence": lambda **kwargs: get_competitor_evidence(
            runtime_config=runtime_config,
            **kwargs,
        ),
    }


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse function-call arguments defensively."""
    text = raw_arguments.strip()
    if text == "":
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Tool-call arguments must decode to a JSON object.")
    return parsed
