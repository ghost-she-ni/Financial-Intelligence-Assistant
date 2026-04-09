from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.extraction.checkpoints import (
    build_checkpoint_record,
    get_checkpoint_path,
    get_successfully_processed_chunk_ids,
    load_or_bootstrap_checkpoint,
    save_checkpoint,
    upsert_checkpoint_records,
)
from src.llm.client import LLMClient

# Configure module logger
logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {
    "company",
    "executive",
    "product",
    "financial_metric",
}

INVALID_PRODUCT_TERMS = {
    "common stock",
    "class a",
    "class b",
    "class a common stock",
    "class b common stock",
    "software-as-a-service",
    "saas",
    "subscription",
    "subscriptions",
    "service",
    "services",
    "product",
    "products",
}

INVALID_FINANCIAL_METRIC_TERMS = {
    "adbe",
    "pfe",
    "lmt",
}

INVALID_EXECUTIVE_TERMS = {
    "chief executive officer",
    "ceo",
    "chairman",
    "executive officer",
    "officer",
}

REQUIRED_CHUNK_COLUMNS = {
    "chunk_id",
    "doc_id",
    "company",
    "fiscal_year",
    "page_start",
    "page_end",
    "chunk_text",
}

# Expanded patterns for scoring
PATTERNS_FOR_SCORING = [
    re.compile(r"\brisk\b", re.IGNORECASE),
    re.compile(r"\brisk factors\b", re.IGNORECASE),
    re.compile(r"\bcompetition\b", re.IGNORECASE),
    re.compile(r"\bcompetitive\b", re.IGNORECASE),
    re.compile(r"\bcompetitor\b", re.IGNORECASE),
    re.compile(r"\bai\b", re.IGNORECASE),
    re.compile(r"\bartificial intelligence\b", re.IGNORECASE),
    re.compile(r"\bgenerative ai\b", re.IGNORECASE),
    re.compile(r"\brevenue\b", re.IGNORECASE),
    re.compile(r"\bincome\b", re.IGNORECASE),
    re.compile(r"\bps\b", re.IGNORECASE),
    re.compile(r"\bmargin\b", re.IGNORECASE),
    re.compile(r"\bcash flow\b", re.IGNORECASE),
    re.compile(r"\bproduct\b", re.IGNORECASE),
    re.compile(r"\bproducts\b", re.IGNORECASE),
    re.compile(r"\bservice\b", re.IGNORECASE),
    re.compile(r"\bservices\b", re.IGNORECASE),
    re.compile(r"\bsegment\b", re.IGNORECASE),
    re.compile(r"\bsegments\b", re.IGNORECASE),
    re.compile(r"\bchief executive officer\b", re.IGNORECASE),
    re.compile(r"\bceo\b", re.IGNORECASE),
    re.compile(r"\bchairman\b", re.IGNORECASE),
    re.compile(r"\bexecutive\b", re.IGNORECASE),
    re.compile(r"\bofficer\b", re.IGNORECASE),
    re.compile(r"\bmarket share\b", re.IGNORECASE),
    re.compile(r"\boperating income\b", re.IGNORECASE),
    re.compile(r"\bnet income\b", re.IGNORECASE),
    re.compile(r"\bcash flow from operations\b", re.IGNORECASE),
]


def load_chunks(chunks_path: Path) -> pd.DataFrame:
    """Load chunks from Parquet or CSV."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {chunks_path}")

    if chunks_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(chunks_path)
    elif chunks_path.suffix.lower() == ".csv":
        df = pd.read_csv(chunks_path)
    else:
        raise ValueError("Chunks file must be .parquet or .csv")

    missing_cols = REQUIRED_CHUNK_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required chunk columns: {sorted(missing_cols)}")

    return df


def load_existing_entities(output_path: Path) -> pd.DataFrame:
    """Load existing entity results if they exist, otherwise return an empty DataFrame."""
    expected_columns = [
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

    if not output_path.exists():
        return pd.DataFrame(columns=expected_columns)

    if output_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(output_path)
    elif output_path.suffix.lower() == ".csv":
        df = pd.read_csv(output_path)
    else:
        raise ValueError("Output file must be .parquet or .csv")

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in existing entities file: {sorted(missing_cols)}")

    return df[expected_columns].copy()


def save_entities(df: pd.DataFrame, output_path: Path) -> None:
    """Save entities to Parquet or CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must be .parquet or .csv")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without changing semantics."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def score_interesting_chunk(text: str) -> int:
    """
    Return a score for how interesting a chunk is based on keyword matches.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    text_lower = text.lower()
    score = 0
    for pattern in PATTERNS_FOR_SCORING:
        if pattern.search(text_lower):
            score += 1
    return score


def is_interesting_chunk(text: str) -> bool:
    """Legacy binary filter: returns True if any pattern matches."""
    return score_interesting_chunk(text) > 0


def select_candidate_chunks(
    chunks_df: pd.DataFrame,
    mode: str = "all",
    max_chunks: int | None = None,
    min_pattern_score: int = 1,
) -> pd.DataFrame:
    """
    Select a subset of candidate chunks for entity extraction.

    Args:
        chunks_df: DataFrame with chunks.
        mode: "all" (no filtering), "heuristic" (binary pattern match),
              "scored" (keep chunks with score >= min_pattern_score).
        max_chunks: Optional limit on number of chunks to process.
        min_pattern_score: Minimum score to keep a chunk (used only in "scored" mode).

    Returns:
        DataFrame with candidate chunks.
    """
    working_df = chunks_df.copy()

    if mode == "all":
        candidates_df = working_df
    elif mode == "heuristic":
        candidates_df = working_df[working_df["chunk_text"].apply(is_interesting_chunk)].copy()
    elif mode == "scored":
        scores = working_df["chunk_text"].apply(score_interesting_chunk)
        mask = scores >= min_pattern_score
        candidates_df = working_df[mask].copy()
        if max_chunks is None and mode == "scored":
            # For scored mode, we can optionally sort by score (descending) before taking top
            candidates_df = candidates_df.assign(_score=scores[mask])
            candidates_df = candidates_df.sort_values("_score", ascending=False).drop(columns=["_score"])
    else:
        raise ValueError("mode must be 'all', 'heuristic', or 'scored'")

    candidates_df = candidates_df.sort_values(
        by=["company", "fiscal_year", "page_start", "page_end", "chunk_id"]
    ).reset_index(drop=True)

    if max_chunks is not None:
        candidates_df = candidates_df.head(max_chunks).copy()

    return candidates_df.reset_index(drop=True)


def filter_unprocessed_chunks(
    candidate_chunks_df: pd.DataFrame,
    existing_entities_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Keep only chunks that have not yet been processed.
    """
    processed_chunk_ids = set(existing_entities_df["chunk_id"].dropna().astype(str).unique().tolist())
    if checkpoint_df is not None:
        processed_chunk_ids |= get_successfully_processed_chunk_ids(checkpoint_df)
    remaining_df = candidate_chunks_df[~candidate_chunks_df["chunk_id"].isin(processed_chunk_ids)].copy()
    return remaining_df.reset_index(drop=True)


def build_system_prompt(system_prompt_path: Path | None = None) -> str:
    """
    Load system prompt from a file, or return default.
    """
    if system_prompt_path is not None:
        try:
            prompt = system_prompt_path.read_text(encoding="utf-8").strip()
            if prompt:
                logger.info(f"Loaded system prompt from {system_prompt_path}")
                return prompt
            else:
                logger.warning(f"System prompt file {system_prompt_path} is empty, using default.")
        except Exception as e:
            logger.warning(f"Failed to load system prompt from {system_prompt_path}: {e}, using default.")

    # Default system prompt
    return (
        "You are an information extraction assistant for financial reports.\n"
        "Extract only entities explicitly supported by the provided chunk.\n"
        "Do not infer entities that are not clearly mentioned.\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "entities": [\n'
        "    {\n"
        '      "entity_text": "...",\n'
        '      "entity_type": "company | executive | product | financial_metric",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- entity_type must be one of: company, executive, product, financial_metric\n"
        "- confidence must be a float between 0 and 1\n"
        "- Do not extract generic nouns unless they refer to a specific entity mention\n"
        "- Remove duplicates\n"
        "- If no valid entities are present, return an empty list\n"
        "\n"
        "Entity type guidance:\n"
        "- company: organization or company names explicitly mentioned\n"
        "- executive: named people in leadership or executive roles\n"
        "- product: specific named commercial products, platforms, brands, or named offerings\n"
        "- financial_metric: explicit financial measures such as revenue, net income, operating income, EPS, margin, cash flow, market capitalization, share count, growth percentages tied to financial performance\n"
        "\n"
        "Important exclusions:\n"
        "- Do NOT label ticker symbols, stock symbols, share classes, or stock listings as financial_metric\n"
        "- Do NOT label 'Common Stock', 'Class A', 'Class B', or similar stock labels as product\n"
        "- Do NOT label generic business models or category labels such as 'software-as-a-service', 'subscription', 'services', or 'products' alone as product\n"
        "- Do NOT label generic technology categories as product unless they are clearly named offerings\n"
        "- Do not extract broad generic phrases that are not true entity mentions\n"
    )


def build_user_prompt(chunk_row: pd.Series) -> str:
    """Build the user prompt for one chunk."""
    chunk_text = normalize_whitespace(str(chunk_row["chunk_text"]))

    return (
        f"Document metadata:\n"
        f"- doc_id: {chunk_row['doc_id']}\n"
        f"- company: {chunk_row['company']}\n"
        f"- fiscal_year: {chunk_row['fiscal_year']}\n"
        f"- pages: {chunk_row['page_start']}-{chunk_row['page_end']}\n"
        f"- chunk_id: {chunk_row['chunk_id']}\n\n"
        f"Chunk text:\n{chunk_text}\n\n"
        "Extract only supported entities from this chunk."
    )


def normalize_entity_type(entity_type: str) -> str | None:
    """Normalize model output to one of the allowed entity types."""
    if not isinstance(entity_type, str):
        return None

    entity_type = entity_type.strip().lower()

    mapping = {
        "company": "company",
        "companies": "company",
        "executive": "executive",
        "executives": "executive",
        "person": "executive",
        "product": "product",
        "products": "product",
        "financial_metric": "financial_metric",
        "financial metrics": "financial_metric",
        "metric": "financial_metric",
        "metrics": "financial_metric",
    }

    return mapping.get(entity_type)


def normalize_confidence(value: Any) -> float | None:
    """Normalize confidence to a float between 0 and 1."""
    try:
        confidence = float(value)
    except Exception:
        return None

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return confidence


def is_valid_entity(entity_text: str, entity_type: str) -> bool:
    """Lightweight post-processing filter for obvious false positives."""
    if not isinstance(entity_text, str) or not entity_text.strip():
        return False

    normalized = entity_text.strip().lower()

    if entity_type == "product" and normalized in INVALID_PRODUCT_TERMS:
        return False

    if entity_type == "financial_metric" and normalized in INVALID_FINANCIAL_METRIC_TERMS:
        return False

    # Keep only named people for executives, not role labels alone
    if entity_type == "executive" and normalized in INVALID_EXECUTIVE_TERMS:
        return False

    # Avoid very short noisy extractions
    if len(normalized) <= 1:
        return False

    return True


def parse_entities_response(
    llm_parsed_json: dict[str, Any],
    chunk_row: pd.Series,
    created_at: str,
) -> list[dict[str, Any]]:
    """Parse and validate the JSON response from the LLM."""
    entities = llm_parsed_json.get("entities", [])
    if not isinstance(entities, list):
        return []

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for item in entities:
        if not isinstance(item, dict):
            continue

        entity_text = str(item.get("entity_text", "")).strip()
        entity_type = normalize_entity_type(item.get("entity_type", ""))
        confidence = normalize_confidence(item.get("confidence"))

        if not entity_text:
            continue
        if entity_type not in VALID_ENTITY_TYPES:
            continue
        if not is_valid_entity(entity_text, entity_type):
            continue

        dedupe_key = (entity_text.lower(), entity_type)
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        records.append(
            {
                "chunk_id": chunk_row["chunk_id"],
                "entity_text": entity_text,
                "entity_type": entity_type,
                "confidence": confidence,
                "source_doc_id": chunk_row["doc_id"],
                "year": int(chunk_row["fiscal_year"]),
                "company": chunk_row["company"],
                "page_start": int(chunk_row["page_start"]),
                "page_end": int(chunk_row["page_end"]),
                "created_at": created_at,
            }
        )

    return records


def extract_entities_from_chunk(
    chunk_row: pd.Series,
    llm_client: LLMClient,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Run entity extraction on a single chunk."""
    user_prompt = build_user_prompt(chunk_row)

    llm_result = llm_client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        task_name="entity_extraction",
        metadata={
            "chunk_id": chunk_row["chunk_id"],
            "doc_id": chunk_row["doc_id"],
            "company": chunk_row["company"],
            "fiscal_year": int(chunk_row["fiscal_year"]),
            "page_start": int(chunk_row["page_start"]),
            "page_end": int(chunk_row["page_end"]),
        },
    )

    parsed_json = llm_result.get("parsed_json", {})
    created_at = llm_result["created_at"]

    return parse_entities_response(
        llm_parsed_json=parsed_json,
        chunk_row=chunk_row,
        created_at=created_at,
    )


def run_entity_extraction(
    chunks_path: Path,
    output_path: Path,
    llm_model: str,
    llm_cache_path: Path,
    mode: str = "all",
    max_chunks: int | None = None,
    save_every: int = 10,
    dry_run: bool = False,
    verbose: bool = False,
    system_prompt_path: Path | None = None,
    min_pattern_score: int = 1,
) -> pd.DataFrame:
    """
    End-to-end entity extraction pipeline with resumability.

    Args:
        chunks_path: Path to chunks file.
        output_path: Path to output entities file.
        llm_model: LLM model name.
        llm_cache_path: Path to LLM cache file.
        mode: "all", "heuristic" (binary pattern match), or "scored".
        max_chunks: Maximum number of candidate chunks to process.
        save_every: Save results every N processed chunks.
        dry_run: If True, only count chunks and exit without LLM calls.
        verbose: If True, log progress.
        system_prompt_path: Optional path to a file containing the system prompt.
        min_pattern_score: Minimum score to keep a chunk (used only in "scored" mode).

    Returns:
        DataFrame with extracted entities.
    """
    chunks_df = load_chunks(chunks_path)
    existing_entities_df = load_existing_entities(output_path)
    checkpoint_path = get_checkpoint_path(output_path)
    checkpoint_df = load_or_bootstrap_checkpoint(
        checkpoint_path=checkpoint_path,
        existing_output_df=existing_entities_df,
    )

    candidate_chunks_df = select_candidate_chunks(
        chunks_df=chunks_df,
        mode=mode,
        max_chunks=max_chunks,
        min_pattern_score=min_pattern_score,
    )

    remaining_chunks_df = filter_unprocessed_chunks(
        candidate_chunks_df=candidate_chunks_df,
        existing_entities_df=existing_entities_df,
        checkpoint_df=checkpoint_df,
    )

    if verbose:
        logger.info("Entity extraction setup:")
        logger.info(f"Total chunks available       : {len(chunks_df)}")
        logger.info(f"Candidate chunks selected    : {len(candidate_chunks_df)}")
        logger.info(f"Already processed chunk_ids  : {existing_entities_df['chunk_id'].nunique() if not existing_entities_df.empty else 0}")
        logger.info(f"Chunks remaining to process  : {len(remaining_chunks_df)}")

    if remaining_chunks_df.empty:
        logger.info("No remaining chunks to process.")
        return existing_entities_df

    if dry_run:
        logger.info("Dry run: exiting without LLM calls.")
        return existing_entities_df

    # Load system prompt (once)
    system_prompt = build_system_prompt(system_prompt_path)

    llm_client = LLMClient(
        model=llm_model,
        temperature=0.0,
        max_output_tokens=700,
        cache_path=llm_cache_path,
    )

    all_entities_df = existing_entities_df.copy()
    all_checkpoint_df = checkpoint_df.copy()
    new_records_buffer: list[dict[str, Any]] = []
    new_checkpoint_records: list[dict[str, Any]] = []

    for idx, chunk_row in enumerate(remaining_chunks_df.to_dict(orient="records"), start=1):
        chunk_series = pd.Series(chunk_row)

        try:
            records = extract_entities_from_chunk(
                chunk_row=chunk_series,
                llm_client=llm_client,
                system_prompt=system_prompt,
            )
            new_records_buffer.extend(records)
            new_checkpoint_records.append(
                build_checkpoint_record(
                    chunk_id=str(chunk_series["chunk_id"]),
                    status="success",
                    record_count=len(records),
                )
            )
        except Exception as exc:
            logger.error(f"Failed on chunk_id={chunk_series['chunk_id']}: {exc}")
            new_checkpoint_records.append(
                build_checkpoint_record(
                    chunk_id=str(chunk_series["chunk_id"]),
                    status="error",
                    record_count=0,
                    error_message=str(exc),
                )
            )

        if idx % save_every == 0 or idx == len(remaining_chunks_df):
            if new_records_buffer:
                new_df = pd.DataFrame(new_records_buffer)
                all_entities_df = pd.concat([all_entities_df, new_df], ignore_index=True)
                all_entities_df = all_entities_df.drop_duplicates(
                    subset=["chunk_id", "entity_text", "entity_type"],
                    keep="last",
                ).reset_index(drop=True)

                save_entities(all_entities_df, output_path)

            if new_checkpoint_records:
                all_checkpoint_df = upsert_checkpoint_records(
                    checkpoint_df=all_checkpoint_df,
                    new_records=new_checkpoint_records,
                )
                save_checkpoint(all_checkpoint_df, checkpoint_path)

            if verbose:
                logger.info(
                    f"Saved progress: processed {idx}/{len(remaining_chunks_df)} remaining chunks | "
                    f"total entity rows={len(all_entities_df)}"
                )
            new_records_buffer = []
            new_checkpoint_records = []

    return all_entities_df


def print_summary(entities_df: pd.DataFrame) -> None:
    """Print a short summary of extracted entities using logging."""
    logger.info("Entity extraction completed.")
    logger.info(f"Total extracted entity rows: {len(entities_df)}")

    if entities_df.empty:
        return

    logger.info("Counts by entity_type:")
    counts_df = entities_df.groupby("entity_type")["entity_text"].count().reset_index(name="n_entities")
    logger.info("\n" + counts_df.to_string(index=False))

    logger.info("Preview:")
    preview_cols = [
        "chunk_id",
        "entity_text",
        "entity_type",
        "confidence",
        "source_doc_id",
        "year",
    ]
    logger.info("\n" + entities_df[preview_cols].head(15).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entities from report chunks.")
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to chunks file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/entities.parquet",
        help="Path to output entities file.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name.",
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default="data/cache/llm_responses.jsonl",
        help="Path to JSONL cache used by the LLM client.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "heuristic", "scored"],
        help="Chunk selection mode.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Optional maximum number of candidate chunks to process.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save output every N processed chunks.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Optional path to a file containing the system prompt.",
    )
    parser.add_argument(
        "--min_pattern_score",
        type=int,
        default=1,
        help="Minimum pattern score to keep a chunk (used only in 'scored' mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count candidate chunks and exit without LLM calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        entities_df = run_entity_extraction(
            chunks_path=Path(args.chunks_path),
            output_path=Path(args.output_path),
            llm_model=args.llm_model,
            llm_cache_path=Path(args.llm_cache_path),
            mode=args.mode,
            max_chunks=args.max_chunks,
            save_every=args.save_every,
            dry_run=args.dry_run,
            verbose=args.verbose,
            system_prompt_path=Path(args.system_prompt_path) if args.system_prompt_path else None,
            min_pattern_score=args.min_pattern_score,
        )
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        sys.exit(1)

    if not args.dry_run:
        print_summary(entities_df)
        logger.info(f"Saved to: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
