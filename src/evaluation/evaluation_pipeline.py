from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.grounded_qa import (
    build_context_block,
    build_system_prompt,
    build_user_prompt,
    normalize_citations,
)
from src.common.io import now_utc_iso, read_table
from src.embeddings.cache import make_query_id
from src.llm.client import LLMClient
from src.retrieval.retrieve import (
    CLASSICAL_ML_RETRIEVAL_MODE,
    IMPROVED_RETRIEVAL_MODE,
    NAIVE_RETRIEVAL_MODE,
    keyword_coverage_score,
    lexical_overlap_score,
    normalize_retrieval_mode,
    retrieve_top_k_with_mode,
)

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "financebench" / "financebench_subset_local_smoke.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "evaluation" / "local_smoke" / "improved" / "evaluation_runs.parquet"
)
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_CHUNK_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
DEFAULT_QUERY_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"
DEFAULT_LLM_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_responses.jsonl"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

SUCCESS_STATUSES = {"success", "retrieval_only"}
REQUIRED_FALLBACK_CHUNK_COLUMNS = {
    "chunk_id",
    "doc_id",
    "company",
    "fiscal_year",
    "page_start",
    "page_end",
    "chunk_text",
}


def clean_optional_text(value: Any) -> str | None:
    """Normalize optional textual metadata."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def clean_optional_int(value: Any) -> int | None:
    """Normalize optional integer metadata."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if pd.isna(value):
        return None
    return int(value)


def serialize_json_field(value: Any) -> str:
    """Serialize a field as compact JSON for robust parquet round-trips."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_company_key(value: str | None) -> str | None:
    """Normalize company names for loose equality checks."""
    if value is None:
        return None
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    return normalized or None


@lru_cache(maxsize=4)
def load_fallback_chunks_cached(chunks_path_str: str) -> pd.DataFrame:
    """Load chunk metadata once for lexical fallback retrieval."""
    chunks_df = read_table(Path(chunks_path_str))
    missing_columns = REQUIRED_FALLBACK_CHUNK_COLUMNS - set(chunks_df.columns)
    if missing_columns:
        raise ValueError(f"Chunks file is missing columns required for fallback retrieval: {sorted(missing_columns)}")
    return chunks_df


def lexical_fallback_retrieve_top_k(
    chunks_path: Path,
    query_text: str,
    top_k: int,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
) -> pd.DataFrame:
    """Fallback retrieval mode based only on lexical matching."""
    chunks_df = load_fallback_chunks_cached(str(chunks_path.resolve())).copy()

    company_key = normalize_company_key(company_filter)
    if company_key is not None:
        company_keys = chunks_df["company"].fillna("").astype(str).apply(normalize_company_key)
        chunks_df = chunks_df[company_keys == company_key].copy()

    if fiscal_year_filter is not None:
        fiscal_year_series = pd.to_numeric(chunks_df["fiscal_year"], errors="coerce")
        chunks_df = chunks_df[fiscal_year_series == fiscal_year_filter].copy()

    if chunks_df.empty:
        raise ValueError("No chunks available after lexical fallback filtering.")

    chunks_df["score"] = 0.0
    chunks_df["lexical_score"] = chunks_df["chunk_text"].fillna("").apply(
        lambda text: lexical_overlap_score(query_text, text)
    )
    chunks_df["coverage_score"] = chunks_df["chunk_text"].fillna("").apply(
        lambda text: keyword_coverage_score(query_text, text)
    )
    chunks_df["section_score"] = 0.0
    chunks_df["final_score"] = 0.65 * chunks_df["lexical_score"] + 0.35 * chunks_df["coverage_score"]

    results_df = (
        chunks_df.sort_values(
            by=["final_score", "lexical_score", "coverage_score", "page_start"],
            ascending=[False, False, False, True],
        )
        .head(top_k)
        .copy()
        .reset_index(drop=True)
    )

    results_df.insert(0, "query_id", make_query_id(query_text))
    results_df.insert(1, "query_text", query_text)
    return results_df


def load_questions(questions_path: Path) -> pd.DataFrame:
    """Load and normalize the evaluation questions file."""
    raw_df = read_table(questions_path)
    if raw_df.empty:
        raise ValueError(f"Questions file is empty: {questions_path}")

    question_source_column = None
    for candidate in ["question", "query_text"]:
        if candidate in raw_df.columns:
            question_source_column = candidate
            break

    if question_source_column is None:
        raise ValueError("Questions file must contain either 'question' or 'query_text'.")

    df = raw_df.copy()
    df["question"] = df[question_source_column].fillna("").astype(str).str.strip()
    df = df[df["question"] != ""].reset_index(drop=True)

    if "question_id" not in df.columns:
        if "financebench_id" in df.columns:
            df["question_id"] = df["financebench_id"].fillna("").astype(str).str.strip()
        elif "query_id" in df.columns:
            df["question_id"] = df["query_id"].fillna("").astype(str).str.strip()
        else:
            df["question_id"] = ""

    missing_id_mask = df["question_id"].fillna("").astype(str).str.strip() == ""
    df.loc[missing_id_mask, "question_id"] = df.loc[missing_id_mask, "question"].apply(make_query_id)

    if df["question_id"].duplicated().any():
        duplicated_ids = df.loc[df["question_id"].duplicated(), "question_id"].tolist()
        raise ValueError(f"Duplicate question_id values detected: {duplicated_ids}")

    if "company_filter" not in df.columns:
        source_company_column = "company" if "company" in df.columns else None
        if source_company_column is not None:
            df["company_filter"] = df[source_company_column].apply(clean_optional_text)
        else:
            df["company_filter"] = None
    else:
        df["company_filter"] = df["company_filter"].apply(clean_optional_text)

    if "fiscal_year_filter" not in df.columns:
        source_year_column = None
        for candidate in ["doc_period", "fiscal_year"]:
            if candidate in df.columns:
                source_year_column = candidate
                break

        if source_year_column is not None:
            df["fiscal_year_filter"] = df[source_year_column].apply(clean_optional_int)
        else:
            df["fiscal_year_filter"] = None
    else:
        df["fiscal_year_filter"] = df["fiscal_year_filter"].apply(clean_optional_int)

    if "reference_answer" not in df.columns:
        for candidate in ["expected_answer", "answer"]:
            if candidate in df.columns:
                df["reference_answer"] = df[candidate].fillna("").astype(str).str.strip()
                break
        else:
            df["reference_answer"] = ""
    else:
        df["reference_answer"] = df["reference_answer"].fillna("").astype(str).str.strip()

    df.insert(0, "question_order", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def load_existing_runs(output_path: Path) -> pd.DataFrame:
    """Load existing evaluation runs if the output file already exists."""
    if not output_path.exists():
        return pd.DataFrame()

    existing_df = read_table(output_path)
    if "question_id" not in existing_df.columns:
        raise ValueError(f"Existing runs file is missing 'question_id': {output_path}")

    return existing_df.copy()


def filter_pending_questions(
    questions_df: pd.DataFrame,
    existing_runs_df: pd.DataFrame,
    resume: bool,
    limit_questions: int | None = None,
    question_id: str | None = None,
) -> pd.DataFrame:
    """Select the subset of questions that still needs to run."""
    pending_df = questions_df.copy()

    if question_id is not None:
        pending_df = pending_df[pending_df["question_id"] == question_id].reset_index(drop=True)
        if pending_df.empty:
            raise ValueError(f"Question id not found in questions file: {question_id}")

    if resume and not existing_runs_df.empty:
        completed_ids = set(
            existing_runs_df.loc[
                existing_runs_df["status"].fillna("").isin(SUCCESS_STATUSES),
                "question_id",
            ].astype(str)
        )
        pending_df = pending_df[~pending_df["question_id"].isin(completed_ids)].reset_index(drop=True)

    if limit_questions is not None and limit_questions > 0:
        pending_df = pending_df.head(limit_questions).reset_index(drop=True)

    return pending_df


def resolve_system_prompt(system_prompt_path: Path | None) -> str:
    """Load a custom prompt if provided, otherwise return the default prompt."""
    if system_prompt_path is None:
        return build_system_prompt()

    prompt_text = system_prompt_path.read_text(encoding="utf-8").strip()
    if prompt_text == "":
        logger.warning("System prompt file is empty. Falling back to the default prompt.")
        return build_system_prompt()

    return prompt_text


def extract_retrieved_chunk_ids(retrieval_results_df: pd.DataFrame) -> list[str]:
    """Extract retrieved chunk ids in ranking order."""
    if retrieval_results_df.empty or "chunk_id" not in retrieval_results_df.columns:
        return []
    return retrieval_results_df["chunk_id"].astype(str).tolist()


def build_run_record(
    question_row: pd.Series,
    questions_path: Path,
    top_k: int,
    embedding_model: str,
    llm_model: str,
    skip_llm: bool,
    retrieval_results_df: pd.DataFrame | None = None,
    generated_answer: str = "",
    citations: list[dict[str, Any]] | None = None,
    llm_metadata: dict[str, Any] | None = None,
    retrieval_mode: str = IMPROVED_RETRIEVAL_MODE,
    status: str = "success",
    error_message: str = "",
) -> dict[str, Any]:
    """Convert one question run into a checkpointable record."""
    retrieval_results_df = retrieval_results_df if retrieval_results_df is not None else pd.DataFrame()
    citations = citations or []
    llm_metadata = llm_metadata or {}

    retrieved_chunk_ids = extract_retrieved_chunk_ids(retrieval_results_df)
    retrieved_context = (
        build_context_block(retrieval_results_df) if not retrieval_results_df.empty else ""
    )
    retrieval_payload = (
        json.loads(retrieval_results_df.to_json(orient="records", force_ascii=False))
        if not retrieval_results_df.empty
        else []
    )

    company_filter = clean_optional_text(question_row.get("company_filter"))
    fiscal_year_filter = clean_optional_int(question_row.get("fiscal_year_filter"))

    record = {
        "question_id": str(question_row["question_id"]),
        "question": str(question_row["question"]),
        "retrieved_chunk_ids": serialize_json_field(retrieved_chunk_ids),
        "retrieved_context": retrieved_context,
        "generated_answer": generated_answer or "",
        "citations": serialize_json_field(citations),
        "question_order": int(question_row["question_order"]),
        "company_filter": company_filter,
        "fiscal_year_filter": fiscal_year_filter,
        "reference_answer": str(question_row.get("reference_answer", "") or ""),
        "n_retrieved_chunks": len(retrieved_chunk_ids),
        "retrieval_results": serialize_json_field(retrieval_payload),
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "skip_llm": bool(skip_llm),
        "status": status,
        "error_message": error_message,
        "llm_from_cache": bool(llm_metadata.get("from_cache", False)),
        "llm_request_hash": llm_metadata.get("request_hash"),
        "llm_created_at": llm_metadata.get("created_at"),
        "source_questions_path": str(questions_path.resolve()),
        "run_timestamp_utc": now_utc_iso(),
    }

    for optional_column in [
        "financebench_id",
        "company",
        "doc_name",
        "doc_period",
        "question_type",
        "question_reasoning",
    ]:
        if optional_column in question_row.index:
            record[optional_column] = question_row.get(optional_column)

    return record


def upsert_run_record(existing_runs_df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Insert or replace one evaluation record by question_id."""
    new_df = pd.DataFrame([record])
    if existing_runs_df.empty:
        return new_df

    merged_df = pd.concat([existing_runs_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["question_id"], keep="last")

    sort_columns = [column for column in ["question_order", "question_id"] if column in merged_df.columns]
    if sort_columns:
        merged_df = merged_df.sort_values(sort_columns).reset_index(drop=True)

    return merged_df


def save_runs(output_path: Path, runs_df: pd.DataFrame) -> None:
    """Persist evaluation runs to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_parquet(output_path, index=False)


def summarize_runs(runs_df: pd.DataFrame) -> None:
    """Log a short summary at the end of the pipeline."""
    if runs_df.empty:
        logger.info("No evaluation rows were produced.")
        return

    status_counts = runs_df["status"].fillna("<missing>").value_counts().to_dict()
    logger.info("Evaluation pipeline completed.")
    logger.info("Total rows saved: %s", len(runs_df))
    logger.info("Status counts: %s", status_counts)


def run_evaluation_pipeline(
    questions_path: Path,
    output_path: Path,
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    llm_model: str,
    top_k: int,
    llm_cache_path: Path,
    device: str | None = None,
    system_prompt_path: Path | None = None,
    skip_llm: bool = False,
    retrieval_mode: str = IMPROVED_RETRIEVAL_MODE,
    enable_metadata_filters: bool = True,
    use_question_row_filters: bool = True,
    enable_lexical_rerank: bool = True,
    enable_bm25: bool = True,
    enable_reranker: bool = True,
    allow_lexical_fallback: bool = True,
    persistent_index_mode: str = "auto",
    persistent_index_backend: str = "auto",
    persistent_index_dir: Path | None = None,
    rebuild_persistent_index: bool = False,
    resume: bool = True,
    limit_questions: int | None = None,
    question_id: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run retrieval + generation for each question and checkpoint the results."""
    questions_df = load_questions(questions_path)
    existing_runs_df = load_existing_runs(output_path)
    pending_questions_df = filter_pending_questions(
        questions_df=questions_df,
        existing_runs_df=existing_runs_df,
        resume=resume,
        limit_questions=limit_questions,
        question_id=question_id,
    )

    if pending_questions_df.empty:
        logger.info("No pending questions to run.")
        summarize_runs(existing_runs_df)
        return existing_runs_df

    retrieval_mode = normalize_retrieval_mode(retrieval_mode)

    logger.info("Questions loaded: %s", len(questions_df))
    logger.info("Pending questions: %s", len(pending_questions_df))

    dense_retrieval_ready = True
    try:
        from src.generation.rag_answer import ensure_query_embeddings_cached

        ensure_query_embeddings_cached(
            query_texts=pending_questions_df["question"].tolist(),
            query_embeddings_path=query_embeddings_path,
            embedding_model_name=embedding_model,
            device=device,
            verbose=verbose,
        )
    except Exception as exc:
        dense_retrieval_ready = False
        if not allow_lexical_fallback:
            raise
        logger.warning(
            "Dense query embedding preparation is unavailable (%s). "
            "The pipeline will use lexical fallback retrieval when needed.",
            exc,
        )

    llm_client = None
    system_prompt = None
    if not skip_llm:
        system_prompt = resolve_system_prompt(system_prompt_path)
        llm_client = LLMClient(
            model=llm_model,
            temperature=0.0,
            max_output_tokens=800,
            cache_path=llm_cache_path,
        )

    runs_df = existing_runs_df.copy()

    for current_index in range(1, len(pending_questions_df) + 1):
        question_series = pending_questions_df.iloc[current_index - 1]
        retrieval_results_df = pd.DataFrame()
        active_retrieval_mode = retrieval_mode
        logger.info(
            "[%s/%s] question_id=%s",
            current_index,
            len(pending_questions_df),
            question_series["question_id"],
        )

        try:
            company_filter = None
            fiscal_year_filter = None
            if use_question_row_filters:
                company_filter = clean_optional_text(question_series.get("company_filter"))
                fiscal_year_filter = clean_optional_int(question_series.get("fiscal_year_filter"))

            question_text = str(question_series["question"])

            if dense_retrieval_ready:
                try:
                    retrieval_results_df = retrieve_top_k_with_mode(
                        chunks_path=chunks_path,
                        chunk_embeddings_path=chunk_embeddings_path,
                        query_embeddings_path=query_embeddings_path,
                        embedding_model=embedding_model,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        query_text=question_text,
                        company_filter=company_filter,
                        fiscal_year_filter=fiscal_year_filter,
                        enable_metadata_filters=enable_metadata_filters,
                        enable_lexical_rerank=enable_lexical_rerank,
                        enable_bm25=enable_bm25,
                        enable_reranker=enable_reranker,
                        persistent_index_mode=persistent_index_mode,
                        persistent_index_backend=persistent_index_backend,
                        persistent_index_dir=persistent_index_dir,
                        rebuild_persistent_index=rebuild_persistent_index,
                        verbose=verbose,
                    )
                except Exception as exc:
                    if not allow_lexical_fallback:
                        raise
                    logger.warning(
                        "Dense retrieval failed for question_id=%s (%s). Falling back to lexical retrieval.",
                        question_series["question_id"],
                        exc,
                    )
                    active_retrieval_mode = "lexical_fallback"
                    retrieval_results_df = lexical_fallback_retrieve_top_k(
                        chunks_path=chunks_path,
                        query_text=question_text,
                        top_k=top_k,
                        company_filter=company_filter if use_question_row_filters else None,
                        fiscal_year_filter=fiscal_year_filter if use_question_row_filters else None,
                    )
            else:
                active_retrieval_mode = "lexical_fallback"
                retrieval_results_df = lexical_fallback_retrieve_top_k(
                    chunks_path=chunks_path,
                    query_text=question_text,
                    top_k=top_k,
                    company_filter=company_filter if use_question_row_filters else None,
                    fiscal_year_filter=fiscal_year_filter if use_question_row_filters else None,
                )

            if skip_llm:
                record = build_run_record(
                    question_row=question_series,
                    questions_path=questions_path,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    skip_llm=True,
                    retrieval_results_df=retrieval_results_df,
                    generated_answer="",
                    citations=[],
                    llm_metadata=None,
                    retrieval_mode=active_retrieval_mode,
                    status="retrieval_only",
                    error_message="",
                )
            else:
                assert llm_client is not None
                assert system_prompt is not None

                user_prompt = build_user_prompt(
                    question=str(question_series["question"]),
                    retrieval_results_df=retrieval_results_df,
                )
                llm_result = llm_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    task_name="evaluation_generation",
                    metadata={
                        "question_id": str(question_series["question_id"]),
                        "question": str(question_series["question"]),
                        "top_k": top_k,
                        "embedding_model": embedding_model,
                        "company_filter": company_filter,
                        "fiscal_year_filter": fiscal_year_filter,
                    },
                )
                parsed_json = llm_result.get("parsed_json", {})
                generated_answer = str(parsed_json.get("answer", "")).strip()
                citations = normalize_citations(
                    citations=parsed_json.get("citations", []),
                    retrieval_results_df=retrieval_results_df,
                )

                record = build_run_record(
                    question_row=question_series,
                    questions_path=questions_path,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    skip_llm=False,
                    retrieval_results_df=retrieval_results_df,
                    generated_answer=generated_answer,
                    citations=citations,
                    llm_metadata=llm_result,
                    retrieval_mode=active_retrieval_mode,
                    status="success",
                    error_message="",
                )

        except Exception as exc:
            logger.exception(
                "Evaluation failed for question_id=%s",
                question_series["question_id"],
            )
            record = build_run_record(
                question_row=question_series,
                questions_path=questions_path,
                top_k=top_k,
                embedding_model=embedding_model,
                llm_model=llm_model,
                skip_llm=skip_llm,
                retrieval_results_df=retrieval_results_df,
                generated_answer="",
                citations=[],
                llm_metadata=None,
                retrieval_mode=active_retrieval_mode,
                status="error",
                error_message=str(exc),
            )

        runs_df = upsert_run_record(runs_df, record)
        save_runs(output_path=output_path, runs_df=runs_df)

    summarize_runs(runs_df)
    logger.info("Saved evaluation runs to: %s", output_path.resolve())
    return runs_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a retrieval + generation evaluation pipeline over a questions file."
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Path to the questions table (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output parquet path for evaluation runs.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default=str(DEFAULT_CHUNKS_PATH),
        help="Path to chunk metadata.",
    )
    parser.add_argument(
        "--chunk_embeddings_path",
        type=str,
        default=str(DEFAULT_CHUNK_EMBEDDINGS_PATH),
        help="Path to chunk embeddings cache.",
    )
    parser.add_argument(
        "--query_embeddings_path",
        type=str,
        default=str(DEFAULT_QUERY_EMBEDDINGS_PATH),
        help="Path to query embeddings cache.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM model name.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks retrieved per question.",
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default=str(DEFAULT_LLM_CACHE_PATH),
        help="Path to the JSONL LLM cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for embeddings, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Optional path to a custom system prompt file.",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Run retrieval only and leave generated_answer empty.",
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
        help="Retrieval mode to evaluate: classical ML baseline, naive dense baseline, or improved retriever.",
    )
    parser.add_argument(
        "--disable_metadata_filters",
        action="store_true",
        help="Disable metadata filters in retrieval.",
    )
    parser.add_argument(
        "--disable_row_filters",
        action="store_true",
        help="Ignore company/year metadata carried by the questions file.",
    )
    parser.add_argument(
        "--disable_lexical_rerank",
        action="store_true",
        help="Disable lexical features in retrieval.",
    )
    parser.add_argument(
        "--disable_bm25",
        action="store_true",
        help="Disable BM25 scoring.",
    )
    parser.add_argument(
        "--disable_reranker",
        action="store_true",
        help="Disable the final reranking pass.",
    )
    parser.add_argument(
        "--disable_lexical_fallback",
        action="store_true",
        help="Fail instead of using lexical-only fallback retrieval when dense retrieval is unavailable.",
    )
    parser.add_argument(
        "--persistent_index_mode",
        type=str,
        default="auto",
        choices=["auto", "persistent", "source"],
        help="Persistent retrieval artifact mode.",
    )
    parser.add_argument(
        "--persistent_index_backend",
        type=str,
        default="auto",
        choices=["auto", "native", "faiss"],
        help="Persistent retrieval artifact backend.",
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
        help="Force rebuilding the prepared retrieval artifact before evaluation.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing successful rows in the output file.",
    )
    parser.add_argument(
        "--limit_questions",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="Optional single question id to run.",
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

    try:
        run_evaluation_pipeline(
            questions_path=Path(args.questions_path),
            output_path=Path(args.output_path),
            chunks_path=Path(args.chunks_path),
            chunk_embeddings_path=Path(args.chunk_embeddings_path),
            query_embeddings_path=Path(args.query_embeddings_path),
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            llm_cache_path=Path(args.llm_cache_path),
            device=args.device,
            system_prompt_path=Path(args.system_prompt_path) if args.system_prompt_path else None,
            skip_llm=args.skip_llm,
            retrieval_mode=args.retrieval_mode,
            enable_metadata_filters=not args.disable_metadata_filters,
            use_question_row_filters=not args.disable_row_filters,
            enable_lexical_rerank=not args.disable_lexical_rerank,
            enable_bm25=not args.disable_bm25,
            enable_reranker=not args.disable_reranker,
            allow_lexical_fallback=not args.disable_lexical_fallback,
            persistent_index_mode=args.persistent_index_mode,
            persistent_index_backend=args.persistent_index_backend,
            persistent_index_dir=Path(args.persistent_index_dir) if args.persistent_index_dir else None,
            rebuild_persistent_index=args.rebuild_persistent_index,
            resume=not args.no_resume,
            limit_questions=args.limit_questions,
            question_id=args.question_id,
            verbose=args.verbose,
        )
    except Exception as exc:
        logger.error("Evaluation pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
