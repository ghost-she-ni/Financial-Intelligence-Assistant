from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.grounded_qa import (
    build_context_block,
    build_system_prompt,
    build_user_prompt,
    normalize_citations,
)
from src.common.io import now_utc_iso, read_table


def test_build_context_block_and_user_prompt_include_optional_metadata() -> None:
    retrieval_results_df = pd.DataFrame(
        [
            {
                "doc_id": "adobe_2022_10k",
                "company": "Adobe",
                "fiscal_year": 2022,
                "page_start": 10,
                "page_end": 11,
                "section_group": "risk_factors",
                "section_title": "Risk Factors",
                "knowledge_entities_preview": "Adobe; subscription revenue",
                "knowledge_triplets_preview": "Adobe|REPORTS|subscription revenue",
                "chunk_id": "chunk_001",
                "chunk_text": "Revenue increased year over year.",
            }
        ]
    )

    context_block = build_context_block(retrieval_results_df)
    user_prompt = build_user_prompt("What changed?", retrieval_results_df)

    assert "[SOURCE 1]" in context_block
    assert "section_group: risk_factors" in context_block
    assert "extracted_entities: Adobe; subscription revenue" in context_block
    assert "QUESTION:\nWhat changed?" in user_prompt
    assert context_block in user_prompt


def test_build_system_prompt_mentions_json_citations_contract() -> None:
    prompt = build_system_prompt()

    assert "You must answer ONLY using the provided retrieved context." in prompt
    assert '"citations": [{"doc_id": "...", "page": 17}]' in prompt


def test_normalize_citations_filters_invalid_and_duplicate_pairs() -> None:
    retrieval_results_df = pd.DataFrame(
        [
            {
                "doc_id": "adobe_2022_10k",
                "page_start": 10,
                "page_end": 11,
            }
        ]
    )

    citations = normalize_citations(
        [
            {"doc_id": "adobe_2022_10k", "page": 10},
            {"doc_id": "adobe_2022_10k", "page": "10"},
            {"doc_id": "adobe_2022_10k", "page": 12},
            {"doc_id": "pfizer_2024_10k", "page": 10},
            {"doc_id": "adobe_2022_10k", "page": 11},
        ],
        retrieval_results_df,
    )

    assert citations == [
        {"doc_id": "adobe_2022_10k", "page": 10},
        {"doc_id": "adobe_2022_10k", "page": 11},
    ]


def test_read_table_supports_csv_and_parquet(tmp_path) -> None:
    expected_df = pd.DataFrame([{"question_id": "q_1", "score": 1.0}])
    csv_path = tmp_path / "runs.csv"
    parquet_path = tmp_path / "runs.parquet"

    expected_df.to_csv(csv_path, index=False)
    expected_df.to_parquet(parquet_path, index=False)

    pd.testing.assert_frame_equal(read_table(csv_path), expected_df)
    pd.testing.assert_frame_equal(read_table(parquet_path), expected_df)


def test_now_utc_iso_returns_timezone_aware_iso_timestamp() -> None:
    parsed = datetime.fromisoformat(now_utc_iso())

    assert parsed.tzinfo is not None
