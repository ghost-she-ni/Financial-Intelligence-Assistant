from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.prompting import build_direct_qa_system_prompt


def build_context_block(retrieval_results_df: pd.DataFrame) -> str:
    """Build a structured context block for grounded financial QA."""
    blocks: list[str] = []

    for rank, row in enumerate(retrieval_results_df.itertuples(index=False), start=1):
        section_lines = ""
        if hasattr(row, "document_source") and getattr(row, "document_source", None):
            section_lines += f"document_source: {row.document_source}\n"
        if hasattr(row, "file_name") and getattr(row, "file_name", None):
            section_lines += f"file_name: {row.file_name}\n"
        if hasattr(row, "document_type") and getattr(row, "document_type", None):
            section_lines += f"document_type: {row.document_type}\n"
        if hasattr(row, "section_group") and getattr(row, "section_group", None):
            section_lines += f"section_group: {row.section_group}\n"
        if hasattr(row, "section_title") and getattr(row, "section_title", None):
            section_lines += f"section_title: {row.section_title}\n"
        if hasattr(row, "knowledge_entities_preview") and getattr(
            row, "knowledge_entities_preview", None
        ):
            section_lines += f"extracted_entities: {row.knowledge_entities_preview}\n"
        if hasattr(row, "knowledge_triplets_preview") and getattr(
            row, "knowledge_triplets_preview", None
        ):
            section_lines += f"extracted_triplets: {row.knowledge_triplets_preview}\n"

        block = (
            f"[SOURCE {rank}]\n"
            f"doc_id: {row.doc_id}\n"
            f"company: {row.company}\n"
            f"fiscal_year: {row.fiscal_year}\n"
            f"pages: {row.page_start}-{row.page_end}\n"
            f"{section_lines}"
            f"chunk_id: {row.chunk_id}\n"
            f"text:\n{row.chunk_text}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """Default system prompt for grounded financial QA."""
    return build_direct_qa_system_prompt()


def build_user_prompt(question: str, retrieval_results_df: pd.DataFrame) -> str:
    """Build the user prompt containing the question and retrieved context."""
    context_block = build_context_block(retrieval_results_df)
    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        "Answer the question strictly from the retrieved context."
    )


def normalize_citations(
    citations: Any,
    retrieval_results_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Normalize and validate citations returned by the LLM."""
    if not isinstance(citations, list):
        return []

    valid_pairs: set[tuple[str, int]] = set()
    valid_doc_ids: set[str] = set()

    for row in retrieval_results_df.itertuples(index=False):
        valid_doc_ids.add(row.doc_id)
        for page in range(int(row.page_start), int(row.page_end) + 1):
            valid_pairs.add((row.doc_id, page))

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for item in citations:
        if not isinstance(item, dict):
            continue

        doc_id = item.get("doc_id")
        page = item.get("page")

        if not isinstance(doc_id, str):
            continue

        try:
            page = int(page)
        except Exception:
            continue

        pair = (doc_id, page)
        if doc_id in valid_doc_ids and pair in valid_pairs and pair not in seen:
            normalized.append({"doc_id": doc_id, "page": page})
            seen.add(pair)

    return normalized
