from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.io import now_utc_iso, read_table
from src.embeddings.cache import make_query_id
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "outputs" / "evaluation" / "local_smoke" / "improved" / "evaluation_runs.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "local_smoke"
    / "improved"
    / "evaluation_runs_judged.parquet"
)
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_responses.jsonl"

REQUIRED_RUN_COLUMNS = {
    "question",
    "retrieved_context",
    "generated_answer",
}

SUCCESS_JUDGE_STATUSES = {
    "success",
    "shortcut_empty_answer",
    "shortcut_no_context",
}


def clean_text(value: Any) -> str:
    """Normalize arbitrary text input."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_judge_system_prompt() -> str:
    """System prompt for support-based answer judgment."""
    return (
        "You are an answer-support judge for financial question answering.\n"
        "Evaluate ONLY on the basis of the provided context.\n"
        "Do not use outside knowledge.\n"
        "Your task is to decide whether the generated answer is supported by the provided context.\n"
        "If the answer contains claims that are missing from the context, overstated, or contradicted, return No.\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "verdict": "Yes" or "No",\n'
        '  "justification": "short explanation"\n'
        "}\n"
        "Rules:\n"
        "- Judge support, not writing style.\n"
        "- Use Yes only if the answer is supported by the context.\n"
        "- Keep the justification short and specific.\n"
    )


def build_judge_user_prompt(
    query: str,
    context: str,
    generated_answer: str,
) -> str:
    """Build the user prompt for one judgment."""
    return (
        f"QUERY:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        "Evaluate whether the generated answer is supported by the provided context only."
    )


def normalize_verdict(value: Any) -> str:
    """Normalize raw model output into canonical Yes/No values."""
    text = clean_text(value).lower()
    if text in {"yes", "y"}:
        return "Yes"
    if text in {"no", "n"}:
        return "No"
    raise ValueError(f"Invalid judge verdict: {value!r}")


def judge_answer_support(
    query: str,
    context: str,
    generated_answer: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_cache_path: Path | str = DEFAULT_LLM_CACHE_PATH,
) -> dict[str, Any]:
    """Judge whether an answer is supported by the given retrieved context."""
    clean_query = clean_text(query)
    clean_context = clean_text(context)
    clean_answer = clean_text(generated_answer)

    if clean_answer == "":
        return {
            "verdict": "No",
            "justification": "The generated answer is empty.",
            "status": "shortcut_empty_answer",
            "request_hash": None,
            "from_cache": False,
            "created_at": now_utc_iso(),
        }

    if clean_context == "":
        return {
            "verdict": "No",
            "justification": "No retrieved context was provided.",
            "status": "shortcut_no_context",
            "request_hash": None,
            "from_cache": False,
            "created_at": now_utc_iso(),
        }

    llm_client = LLMClient(
        model=llm_model,
        temperature=0.0,
        max_output_tokens=250,
        cache_path=llm_cache_path,
    )

    llm_result = llm_client.generate_json(
        system_prompt=build_judge_system_prompt(),
        user_prompt=build_judge_user_prompt(
            query=clean_query,
            context=clean_context,
            generated_answer=clean_answer,
        ),
        task_name="judge_answer_support",
        metadata={
            "query": clean_query,
            "context_length_chars": len(clean_context),
            "generated_answer_length_chars": len(clean_answer),
        },
    )

    parsed_json = llm_result.get("parsed_json", {})
    verdict = normalize_verdict(parsed_json.get("verdict"))
    justification = clean_text(parsed_json.get("justification"))
    if justification == "":
        raise ValueError("Judge returned an empty justification.")

    return {
        "verdict": verdict,
        "justification": justification,
        "status": "success",
        "request_hash": llm_result["request_hash"],
        "from_cache": llm_result["from_cache"],
        "created_at": llm_result["created_at"],
    }


def load_evaluation_runs(input_path: Path) -> pd.DataFrame:
    """Load evaluation runs and normalize the fields consumed by the judge."""
    df = read_table(input_path)
    if df.empty:
        raise ValueError(f"Evaluation runs file is empty: {input_path}")

    missing_columns = REQUIRED_RUN_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Evaluation runs file is missing columns: {sorted(missing_columns)}")

    normalized_df = df.copy()
    normalized_df["question"] = normalized_df["question"].fillna("").astype(str).str.strip()
    normalized_df["retrieved_context"] = (
        normalized_df["retrieved_context"].fillna("").astype(str).str.strip()
    )
    normalized_df["generated_answer"] = (
        normalized_df["generated_answer"].fillna("").astype(str).str.strip()
    )

    if "question_id" not in normalized_df.columns:
        normalized_df["question_id"] = normalized_df["question"].apply(make_query_id)
    else:
        normalized_df["question_id"] = normalized_df["question_id"].fillna("").astype(str).str.strip()
        missing_id_mask = normalized_df["question_id"] == ""
        normalized_df.loc[missing_id_mask, "question_id"] = normalized_df.loc[
            missing_id_mask, "question"
        ].apply(make_query_id)

    if normalized_df["question_id"].duplicated().any():
        duplicated_ids = normalized_df.loc[
            normalized_df["question_id"].duplicated(), "question_id"
        ].tolist()
        raise ValueError(f"Duplicate question_id values detected: {duplicated_ids}")

    return normalized_df.reset_index(drop=True)


def load_existing_judged_runs(output_path: Path) -> pd.DataFrame:
    """Load previously judged runs if they already exist."""
    if not output_path.exists():
        return pd.DataFrame()

    existing_df = read_table(output_path)
    if "question_id" not in existing_df.columns:
        raise ValueError(f"Existing judged file is missing 'question_id': {output_path}")

    return existing_df.copy()


def filter_pending_runs(
    runs_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    resume: bool,
    limit_questions: int | None = None,
    question_id: str | None = None,
) -> pd.DataFrame:
    """Select the subset of rows that still needs judgment."""
    pending_df = runs_df.copy()

    if question_id is not None:
        pending_df = pending_df[pending_df["question_id"] == question_id].reset_index(drop=True)
        if pending_df.empty:
            raise ValueError(f"Question id not found in input file: {question_id}")

    if resume and not existing_df.empty and "judge_status" in existing_df.columns:
        completed_ids = set(
            existing_df.loc[
                existing_df["judge_status"].fillna("").isin(SUCCESS_JUDGE_STATUSES),
                "question_id",
            ].astype(str)
        )
        pending_df = pending_df[~pending_df["question_id"].isin(completed_ids)].reset_index(drop=True)

    if limit_questions is not None and limit_questions > 0:
        pending_df = pending_df.head(limit_questions).reset_index(drop=True)

    return pending_df


def build_judged_record(
    run_row: pd.Series,
    judgment: dict[str, Any],
    source_input_path: Path,
    llm_model: str,
) -> dict[str, Any]:
    """Merge one run row with its judgment fields."""
    record = run_row.to_dict()
    record["judge_verdict"] = judgment["verdict"]
    record["judge_justification"] = judgment["justification"]
    record["judge_status"] = judgment["status"]
    record["judge_model"] = llm_model
    record["judge_request_hash"] = judgment.get("request_hash")
    record["judge_from_cache"] = bool(judgment.get("from_cache", False))
    record["judge_created_at"] = judgment.get("created_at")
    record["judge_run_timestamp_utc"] = now_utc_iso()
    record["source_evaluation_runs_path"] = str(source_input_path.resolve())
    return record


def upsert_judged_record(existing_df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Insert or replace one judged row by question_id."""
    new_df = pd.DataFrame([record])
    if existing_df.empty:
        return new_df

    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["question_id"], keep="last")

    sort_columns = [column for column in ["question_order", "question_id"] if column in merged_df.columns]
    if sort_columns:
        merged_df = merged_df.sort_values(sort_columns).reset_index(drop=True)

    return merged_df


def save_judged_runs(output_path: Path, judged_df: pd.DataFrame) -> None:
    """Persist judged runs to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged_df.to_parquet(output_path, index=False)


def run_judge_pipeline(
    input_path: Path,
    output_path: Path,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_cache_path: Path | str = DEFAULT_LLM_CACHE_PATH,
    resume: bool = True,
    limit_questions: int | None = None,
    question_id: str | None = None,
) -> pd.DataFrame:
    """Run LLM-as-a-Judge over an evaluation runs file."""
    runs_df = load_evaluation_runs(input_path)
    existing_df = load_existing_judged_runs(output_path)
    pending_df = filter_pending_runs(
        runs_df=runs_df,
        existing_df=existing_df,
        resume=resume,
        limit_questions=limit_questions,
        question_id=question_id,
    )

    if pending_df.empty:
        logger.info("No pending rows to judge.")
        return existing_df

    logger.info("Rows loaded: %s", len(runs_df))
    logger.info("Pending rows: %s", len(pending_df))

    judged_df = existing_df.copy()

    for index in range(1, len(pending_df) + 1):
        run_row = pending_df.iloc[index - 1]
        logger.info("[%s/%s] question_id=%s", index, len(pending_df), run_row["question_id"])

        try:
            judgment = judge_answer_support(
                query=run_row["question"],
                context=run_row["retrieved_context"],
                generated_answer=run_row["generated_answer"],
                llm_model=llm_model,
                llm_cache_path=llm_cache_path,
            )
        except Exception as exc:
            logger.exception("Judge failed for question_id=%s", run_row["question_id"])
            judgment = {
                "verdict": "No",
                "justification": clean_text(exc) or "Judge execution failed.",
                "status": "error",
                "request_hash": None,
                "from_cache": False,
                "created_at": now_utc_iso(),
            }

        record = build_judged_record(
            run_row=run_row,
            judgment=judgment,
            source_input_path=input_path,
            llm_model=llm_model,
        )
        judged_df = upsert_judged_record(judged_df, record)
        save_judged_runs(output_path, judged_df)

    logger.info("Saved judged runs to: %s", output_path.resolve())
    return judged_df


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for evaluation runs.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single-item mode: question/query to judge.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Single-item mode: retrieved context to judge against.",
    )
    parser.add_argument(
        "--generated_answer",
        type=str,
        default=None,
        help="Single-item mode: generated answer to evaluate.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Batch mode: input evaluation runs path (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Batch mode output parquet path. In single-item mode, optional JSON output path.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="Judge model name.",
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default=str(DEFAULT_LLM_CACHE_PATH),
        help="Path to the JSONL LLM cache.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Batch mode: ignore existing successful judged rows.",
    )
    parser.add_argument(
        "--limit_questions",
        type=int,
        default=None,
        help="Batch mode: optional limit for smoke runs.",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="Batch mode: optional single question id to judge.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    single_mode_values = [args.query, args.context, args.generated_answer]
    single_mode_enabled = any(value is not None for value in single_mode_values)

    try:
        if single_mode_enabled:
            if not all(value is not None for value in single_mode_values):
                raise ValueError(
                    "Single-item mode requires --query, --context and --generated_answer."
                )

            result = judge_answer_support(
                query=args.query or "",
                context=args.context or "",
                generated_answer=args.generated_answer or "",
                llm_model=args.llm_model,
                llm_cache_path=args.llm_cache_path,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

            output_path = Path(args.output_path)
            if output_path.suffix.lower() == ".json":
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        else:
            run_judge_pipeline(
                input_path=Path(args.input_path),
                output_path=Path(args.output_path),
                llm_model=args.llm_model,
                llm_cache_path=args.llm_cache_path,
                resume=not args.no_resume,
                limit_questions=args.limit_questions,
                question_id=args.question_id,
            )
    except Exception as exc:
        logger.error("Judge execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
