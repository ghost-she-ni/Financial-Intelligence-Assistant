from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    build_results_table,
    classify_probable_error_cause,
    compute_summary_metrics,
    load_judged_runs,
    parse_json_list,
)


def test_parse_json_list_handles_json_and_invalid_values() -> None:
    assert parse_json_list('[{"doc_id":"a","page":1}]') == [{"doc_id": "a", "page": 1}]
    assert parse_json_list("not json") == []
    assert parse_json_list(None) == []


def test_load_judged_runs_enriches_analysis_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "judged.parquet"
    pd.DataFrame(
        [
            {
                "question_id": "q_1",
                "question": "What changed?",
                "generated_answer": "Revenue increased.",
                "citations": '[{"doc_id":"doc_1","page":12}]',
                "retrieved_context": "Revenue increased year over year.",
                "n_retrieved_chunks": 2,
                "judge_verdict": "Yes",
                "judge_justification": "Supported by the context.",
            }
        ]
    ).to_parquet(input_path, index=False)

    runs_df = load_judged_runs(input_path)

    assert runs_df.loc[0, "n_citations"] == 1
    assert bool(runs_df.loc[0, "has_citations"]) is True
    assert bool(runs_df.loc[0, "is_yes"]) is True
    assert bool(runs_df.loc[0, "no_citation"]) is False


def test_classify_probable_error_cause_covers_main_branches() -> None:
    assert classify_probable_error_cause(
        pd.Series(
            {
                "is_no": True,
                "generated_answer_empty": True,
                "status": "retrieval_only",
            }
        )
    ) == "generation imprecise"

    assert classify_probable_error_cause(
        pd.Series(
            {
                "is_no": True,
                "generated_answer_empty": False,
                "retrieved_context_empty": True,
                "n_retrieved_chunks": 0,
            }
        )
    ) == "mauvais retrieval"

    assert classify_probable_error_cause(
        pd.Series(
            {
                "is_no": True,
                "generated_answer_empty": False,
                "retrieved_context_empty": False,
                "n_retrieved_chunks": 2,
                "context_length_chars": 800,
                "has_citations": False,
                "retrieval_mode": "dense_hybrid",
            }
        )
    ) == "contexte incomplet"

    assert classify_probable_error_cause(
        pd.Series(
            {
                "is_no": True,
                "generated_answer_empty": False,
                "retrieved_context_empty": False,
                "n_retrieved_chunks": 6,
                "context_length_chars": 7000,
                "has_citations": False,
                "retrieval_mode": "dense_hybrid",
            }
        )
    ) == "chunking inadequat"


def test_compute_summary_metrics_returns_accuracy_and_no_citation_rate() -> None:
    results_df = pd.DataFrame(
        [
            {"is_yes": True, "is_no": False, "no_citation": False, "judge_verdict": "Yes", "probable_error_cause": ""},
            {"is_yes": False, "is_no": True, "no_citation": True, "judge_verdict": "No", "probable_error_cause": "mauvais retrieval"},
            {"is_yes": False, "is_no": True, "no_citation": True, "judge_verdict": "No", "probable_error_cause": "generation imprecise"},
        ]
    )

    summary = compute_summary_metrics(results_df)

    assert summary["total_questions"] == 3
    assert summary["number_of_yes"] == 1
    assert abs(summary["accuracy"] - (1 / 3)) < 1e-9
    assert summary["number_without_citation"] == 2
    assert abs(summary["no_citation_rate"] - (2 / 3)) < 1e-9


def test_build_results_table_adds_probable_error_cause() -> None:
    runs_df = pd.DataFrame(
        [
            {
                "question_id": "q_1",
                "company": "Adobe",
                "question_type": "domain-relevant",
                "question": "What changed?",
                "judge_verdict": "No",
                "judge_justification": "Not supported.",
                "is_yes": False,
                "is_no": True,
                "has_citations": False,
                "no_citation": True,
                "n_citations": 0,
                "n_retrieved_chunks": 0,
                "retrieval_mode": "lexical_fallback",
                "generated_answer": "",
                "reference_answer": "Revenue increased.",
                "status": "retrieval_only",
                "judge_status": "shortcut_empty_answer",
                "generated_answer_empty": True,
                "retrieved_context_empty": True,
                "context_length_chars": 0,
                "answer_length_chars": 0,
            }
        ]
    )

    results_df = build_results_table(runs_df)

    assert "probable_error_cause" in results_df.columns
    assert results_df.loc[0, "probable_error_cause"] == "generation imprecise"
