from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.judge import (
    build_judge_system_prompt,
    build_judge_user_prompt,
    build_judged_record,
    judge_answer_support,
    load_evaluation_runs,
    normalize_verdict,
    upsert_judged_record,
)


def test_build_judge_system_prompt_mentions_context_only() -> None:
    prompt = build_judge_system_prompt()

    assert "ONLY on the basis of the provided context".lower() in prompt.lower()
    assert '"verdict": "Yes" or "No"' in prompt
    assert '"justification": "short explanation"' in prompt


def test_build_judge_user_prompt_contains_required_inputs() -> None:
    prompt = build_judge_user_prompt(
        query="What changed?",
        context="Revenue increased.",
        generated_answer="Revenue increased in 2022.",
    )

    assert "QUERY:" in prompt
    assert "CONTEXT:" in prompt
    assert "GENERATED ANSWER:" in prompt
    assert "supported by the provided context only" in prompt


def test_normalize_verdict_accepts_yes_no_variants() -> None:
    assert normalize_verdict("Yes") == "Yes"
    assert normalize_verdict("yes") == "Yes"
    assert normalize_verdict("No") == "No"
    assert normalize_verdict("n") == "No"


def test_judge_answer_support_shortcuts_empty_answer() -> None:
    result = judge_answer_support(
        query="What changed?",
        context="Revenue increased.",
        generated_answer="",
    )

    assert result["verdict"] == "No"
    assert result["status"] == "shortcut_empty_answer"


def test_load_evaluation_runs_generates_question_ids_when_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "evaluation_runs.parquet"
    pd.DataFrame(
        [
            {
                "question": "What changed?",
                "retrieved_context": "Revenue increased.",
                "generated_answer": "Revenue increased.",
            }
        ]
    ).to_parquet(input_path, index=False)

    runs_df = load_evaluation_runs(input_path)

    assert "question_id" in runs_df.columns
    assert runs_df.loc[0, "question"] == "What changed?"


def test_build_judged_record_appends_judge_columns(tmp_path: Path) -> None:
    run_row = pd.Series(
        {
            "question_id": "q_1",
            "question": "What changed?",
            "retrieved_context": "Revenue increased.",
            "generated_answer": "Revenue increased.",
            "question_order": 1,
        }
    )
    judgment = {
        "verdict": "Yes",
        "justification": "The answer is explicitly stated in the context.",
        "status": "success",
        "request_hash": "abc123",
        "from_cache": True,
        "created_at": "2026-03-24T17:00:00+00:00",
    }

    record = build_judged_record(
        run_row=run_row,
        judgment=judgment,
        source_input_path=tmp_path / "evaluation_runs.parquet",
        llm_model="gpt-4o-mini",
    )

    assert record["judge_verdict"] == "Yes"
    assert record["judge_justification"].startswith("The answer is")
    assert record["judge_model"] == "gpt-4o-mini"
    assert record["source_evaluation_runs_path"].endswith("evaluation_runs.parquet")


def test_upsert_judged_record_replaces_existing_question_id() -> None:
    existing_df = pd.DataFrame(
        [
            {"question_id": "q_1", "question_order": 1, "judge_verdict": "No"},
            {"question_id": "q_2", "question_order": 2, "judge_verdict": "Yes"},
        ]
    )

    updated_df = upsert_judged_record(
        existing_df=existing_df,
        record={"question_id": "q_1", "question_order": 1, "judge_verdict": "Yes"},
    )

    assert updated_df["question_id"].tolist() == ["q_1", "q_2"]
    assert updated_df.loc[updated_df["question_id"] == "q_1", "judge_verdict"].item() == "Yes"
