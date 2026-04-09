from __future__ import annotations

import json

import pandas as pd

from src.evaluation.evaluation_pipeline import (
    build_run_record,
    lexical_fallback_retrieve_top_k,
    load_questions,
    upsert_run_record,
)


def test_load_questions_uses_financebench_metadata(tmp_path) -> None:
    questions_path = tmp_path / "questions.parquet"
    pd.DataFrame(
        [
            {
                "financebench_id": "financebench_id_1",
                "question": "What changed?",
                "company": "Adobe",
                "doc_period": 2022,
                "expected_answer": "Revenue increased.",
            }
        ]
    ).to_parquet(questions_path, index=False)

    questions_df = load_questions(questions_path)

    assert questions_df.loc[0, "question_id"] == "financebench_id_1"
    assert questions_df.loc[0, "company_filter"] == "Adobe"
    assert questions_df.loc[0, "fiscal_year_filter"] == 2022
    assert questions_df.loc[0, "reference_answer"] == "Revenue increased."


def test_build_run_record_serializes_required_fields(tmp_path) -> None:
    question_row = pd.Series(
        {
            "question_id": "q_1",
            "question": "What changed?",
            "question_order": 1,
            "company_filter": "Adobe",
            "fiscal_year_filter": 2022,
            "reference_answer": "Revenue increased.",
        }
    )
    retrieval_results_df = pd.DataFrame(
        [
            {
                "doc_id": "adobe_2022_10k",
                "company": "adobe",
                "fiscal_year": 2022,
                "page_start": 10,
                "page_end": 10,
                "chunk_id": "chunk_001",
                "chunk_text": "Revenue increased year over year.",
            }
        ]
    )

    record = build_run_record(
        question_row=question_row,
        questions_path=tmp_path / "questions.parquet",
        top_k=5,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        skip_llm=False,
        retrieval_results_df=retrieval_results_df,
        generated_answer="Revenue increased.",
        citations=[{"doc_id": "adobe_2022_10k", "page": 10}],
        llm_metadata={"from_cache": True, "request_hash": "abc123", "created_at": "ts"},
        status="success",
    )

    assert record["question_id"] == "q_1"
    assert json.loads(record["retrieved_chunk_ids"]) == ["chunk_001"]
    assert "Revenue increased year over year." in record["retrieved_context"]
    assert record["generated_answer"] == "Revenue increased."
    assert json.loads(record["citations"]) == [{"doc_id": "adobe_2022_10k", "page": 10}]
    assert record["llm_from_cache"] is True


def test_upsert_run_record_replaces_existing_question_id() -> None:
    existing_runs_df = pd.DataFrame(
        [
            {"question_id": "q_1", "question_order": 1, "generated_answer": "old"},
            {"question_id": "q_2", "question_order": 2, "generated_answer": "keep"},
        ]
    )

    updated_df = upsert_run_record(
        existing_runs_df=existing_runs_df,
        record={"question_id": "q_1", "question_order": 1, "generated_answer": "new"},
    )

    assert updated_df["question_id"].tolist() == ["q_1", "q_2"]
    assert updated_df.loc[updated_df["question_id"] == "q_1", "generated_answer"].item() == "new"


def test_lexical_fallback_retrieve_top_k_applies_filters(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "adobe_2022_10k",
                "company": "adobe",
                "fiscal_year": 2022,
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe operating margin improved in 2022.",
            },
            {
                "chunk_id": "chunk_2",
                "doc_id": "pfizer_2021_10k",
                "company": "pfizer",
                "fiscal_year": 2021,
                "page_start": 12,
                "page_end": 12,
                "chunk_text": "Pfizer acquired several companies in 2021.",
            },
        ]
    ).to_parquet(chunks_path, index=False)

    results_df = lexical_fallback_retrieve_top_k(
        chunks_path=chunks_path,
        query_text="Did Adobe improve operating margin in 2022?",
        top_k=1,
        company_filter="Adobe",
        fiscal_year_filter=2022,
    )

    assert results_df["chunk_id"].tolist() == ["chunk_1"]
