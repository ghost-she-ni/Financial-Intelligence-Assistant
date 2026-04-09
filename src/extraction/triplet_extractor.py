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

VALID_RELATIONS = {
    "COMPETES_WITH",
    "OFFERS",
    "LEADS_BY",
    "REPORTS",
    "MENTIONS",
    "FACES_RISK_FROM",
}

INVALID_OFFERS_TERMS = {
    "offerings",
    "products",
    "services",
    "solutions",
    "line of products and services",
    "a comprehensive suite of offerings",
    "comprehensive suite of offerings",
    "suite of offerings",
}

INVALID_LEADS_BY_TERMS = {
    "chief executive officer",
    "ceo",
    "chairman",
    "officer",
    "leadership position",
    "leadership position in document-intensive industries",
}

LEADS_BY_BAD_PATTERNS = [
    re.compile(r"\bposition\b", re.IGNORECASE),
    re.compile(r"\bindustr", re.IGNORECASE),
    re.compile(r"\bmarket\b", re.IGNORECASE),
    re.compile(r"\bleadership\b", re.IGNORECASE),
    re.compile(r"\badoption\b", re.IGNORECASE),
]

REPORTS_HINT_PATTERNS = [
    re.compile(r"\brevenue\b", re.IGNORECASE),
    re.compile(r"\bincome\b", re.IGNORECASE),
    re.compile(r"\bnet income\b", re.IGNORECASE),
    re.compile(r"\boperating income\b", re.IGNORECASE),
    re.compile(r"\bps\b", re.IGNORECASE),
    re.compile(r"\bdiluted eps\b", re.IGNORECASE),
    re.compile(r"\bmargin\b", re.IGNORECASE),
    re.compile(r"\bcash flow\b", re.IGNORECASE),
    re.compile(r"\bmarket capitalization\b", re.IGNORECASE),
    re.compile(r"\bshares?\b", re.IGNORECASE),
    re.compile(r"\$\s?\d", re.IGNORECASE),
    re.compile(r"\d+(\.\d+)?\s?%", re.IGNORECASE),
]

# Expanded patterns for scoring (triplet‑relevant)
PATTERNS_FOR_SCORING = [
    re.compile(r"\bcompetition\b", re.IGNORECASE),
    re.compile(r"\bcompetitive\b", re.IGNORECASE),
    re.compile(r"\bcompetitor\b", re.IGNORECASE),
    re.compile(r"\bcompete\b", re.IGNORECASE),
    re.compile(r"\brisk\b", re.IGNORECASE),
    re.compile(r"\brisk factors\b", re.IGNORECASE),
    re.compile(r"\bthreat\b", re.IGNORECASE),
    re.compile(r"\boffers?\b", re.IGNORECASE),
    re.compile(r"\bprovide\b", re.IGNORECASE),
    re.compile(r"\bplatform\b", re.IGNORECASE),
    re.compile(r"\bproduct\b", re.IGNORECASE),
    re.compile(r"\bproducts\b", re.IGNORECASE),
    re.compile(r"\bservice\b", re.IGNORECASE),
    re.compile(r"\bservices\b", re.IGNORECASE),
    re.compile(r"\bchief executive officer\b", re.IGNORECASE),
    re.compile(r"\bceo\b", re.IGNORECASE),
    re.compile(r"\bchairman\b", re.IGNORECASE),
    re.compile(r"\bofficer\b", re.IGNORECASE),
    re.compile(r"\brevenue\b", re.IGNORECASE),
    re.compile(r"\bincome\b", re.IGNORECASE),
    re.compile(r"\bps\b", re.IGNORECASE),
    re.compile(r"\bcash flow\b", re.IGNORECASE),
    re.compile(r"\bai\b", re.IGNORECASE),
    re.compile(r"\bartificial intelligence\b", re.IGNORECASE),
    re.compile(r"\bmargin\b", re.IGNORECASE),
    re.compile(r"\boperating income\b", re.IGNORECASE),
    re.compile(r"\bnet income\b", re.IGNORECASE),
    re.compile(r"\bmarket share\b", re.IGNORECASE),
]

REQUIRED_CHUNK_COLUMNS = {
    "chunk_id",
    "doc_id",
    "company",
    "fiscal_year",
    "page_start",
    "page_end",
    "chunk_text",
}


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


def load_existing_triplets(output_path: Path) -> pd.DataFrame:
    """Load existing triplets if present, otherwise return an empty DataFrame."""
    expected_columns = [
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
        raise ValueError(f"Missing columns in existing triplets file: {sorted(missing_cols)}")

    return df[expected_columns].copy()


def save_triplets(df: pd.DataFrame, output_path: Path) -> None:
    """Save triplets to Parquet or CSV."""
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
    """Return a score for how interesting a chunk is for triplet extraction."""
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
    Select candidate chunks for triplet extraction.

    Args:
        chunks_df: DataFrame with chunks.
        mode: "all", "heuristic" (binary), or "scored".
        max_chunks: Optional limit on number of chunks to process.
        min_pattern_score: Minimum score to keep a chunk (scored mode only).

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
            # For scored mode, sort by score (descending) before taking top
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
    existing_triplets_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep only chunks not yet processed."""
    processed_chunk_ids = set(existing_triplets_df["chunk_id"].dropna().astype(str).unique().tolist())
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
        "Extract only relation triplets that are explicitly supported by the provided chunk.\n"
        "Do not infer unsupported relations.\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "triplets": [\n'
        "    {\n"
        '      "entity_a": "...",\n'
        '      "relation": "COMPETES_WITH | OFFERS | LEADS_BY | REPORTS | MENTIONS | FACES_RISK_FROM",\n'
        '      "entity_b": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- relation must be one of: COMPETES_WITH, OFFERS, LEADS_BY, REPORTS, MENTIONS, FACES_RISK_FROM\n"
        "- Keep only explicit and well-supported relations\n"
        "- Remove duplicates\n"
        "- If no valid triplets are present, return an empty list\n"
        "\n"
        "Relation guidance:\n"
        "- COMPETES_WITH: one company explicitly competes with another named company or named competitor group\n"
        "- OFFERS: a company offers a specific named product, named platform, named service, named cloud, named solution, or named brand\n"
        "- LEADS_BY: a company or organization is led by a named person only\n"
        "- REPORTS: a company reports an explicit financial metric or financial measure only\n"
        "- MENTIONS: the chunk explicitly mentions a named entity in a meaningful way\n"
        "- FACES_RISK_FROM: a company faces risk from a specific threat, dependency, uncertainty, technology, regulation, or external factor\n"
        "\n"
        "Important exclusions:\n"
        "- Do not invent competitor names if none are explicitly given\n"
        "- Do not extract vague relations with generic nouns like 'business', 'market', 'company', 'products', 'services', or 'offerings' alone\n"
        "- Do not use descriptive phrases such as 'a comprehensive suite of offerings' or 'line of products and services' as entity_b\n"
        "- For LEADS_BY, entity_b must be a named person, not a role, position, market status, or leadership description\n"
        "- For REPORTS, entity_b must be a financial metric such as revenue, income, EPS, margin, cash flow, market capitalization, share count, or a clearly financial percentage/value\n"
        "- Do not use relation labels outside the allowed list\n"
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
        "Extract only supported relation triplets from this chunk."
    )


def normalize_relation(relation: str) -> str | None:
    """Normalize model output to one of the allowed relation labels."""
    if not isinstance(relation, str):
        return None

    relation = relation.strip().upper()

    if relation in VALID_RELATIONS:
        return relation

    mapping = {
        "COMPETES": "COMPETES_WITH",
        "COMPETITOR_OF": "COMPETES_WITH",
        "PROVIDES": "OFFERS",
        "SELLS": "OFFERS",
        "LED_BY": "LEADS_BY",
        "MANAGED_BY": "LEADS_BY",
        "HAS_EXECUTIVE": "LEADS_BY",
        "MENTIONS": "MENTIONS",
        "REPORT": "REPORTS",
        "REPORTS_METRIC": "REPORTS",
        "HAS_RISK_FROM": "FACES_RISK_FROM",
        "RISK_FROM": "FACES_RISK_FROM",
    }

    return mapping.get(relation)


def looks_like_person_name(text: str) -> bool:
    """Very lightweight heuristic for named persons."""
    if not isinstance(text, str):
        return False

    tokens = text.strip().split()
    if len(tokens) < 2 or len(tokens) > 5:
        return False

    return all(token[:1].isupper() for token in tokens if token)


def has_financial_signal(text: str) -> bool:
    """Check whether a string looks like a financial metric/value."""
    if not isinstance(text, str):
        return False

    text_lower = text.lower()
    return any(pattern.search(text_lower) for pattern in REPORTS_HINT_PATTERNS)


def is_valid_triplet(entity_a: str, relation: str, entity_b: str) -> bool:
    """Lightweight structural validation to remove obvious bad triplets."""
    if not entity_a or not entity_b:
        return False

    if relation not in VALID_RELATIONS:
        return False

    a = entity_a.strip()
    b = entity_b.strip()

    if len(a) < 2 or len(b) < 2:
        return False

    generic_terms = {
        "company",
        "business",
        "market",
        "markets",
        "product",
        "products",
        "service",
        "services",
        "risk",
        "risks",
        "offerings",
        "solutions",
    }

    if a.lower() in generic_terms or b.lower() in generic_terms:
        return False

    if a.lower() == b.lower():
        return False

    # OFFERS should point to a named offering, not a generic descriptive phrase
    if relation == "OFFERS":
        if b.lower() in INVALID_OFFERS_TERMS:
            return False
        if len(b.split()) > 8:
            return False

    # LEADS_BY must point to a named person, not a role or vague phrase
    if relation == "LEADS_BY":
        if b.lower() in INVALID_LEADS_BY_TERMS:
            return False
        if any(pattern.search(b.lower()) for pattern in LEADS_BY_BAD_PATTERNS):
            return False
        if not looks_like_person_name(b):
            return False

    # REPORTS should point to a clear financial metric / value
    if relation == "REPORTS":
        if not has_financial_signal(b):
            return False

    return True


def parse_triplets_response(
    llm_parsed_json: dict[str, Any],
    chunk_row: pd.Series,
    created_at: str,
) -> list[dict[str, Any]]:
    """Parse and validate the JSON response from the LLM."""
    triplets = llm_parsed_json.get("triplets", [])
    if not isinstance(triplets, list):
        return []

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in triplets:
        if not isinstance(item, dict):
            continue

        entity_a = str(item.get("entity_a", "")).strip()
        relation = normalize_relation(item.get("relation", ""))
        entity_b = str(item.get("entity_b", "")).strip()

        if not is_valid_triplet(entity_a, relation or "", entity_b):
            continue

        dedupe_key = (entity_a.lower(), relation, entity_b.lower())
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        records.append(
            {
                "chunk_id": chunk_row["chunk_id"],
                "entity_a": entity_a,
                "relation": relation,
                "entity_b": entity_b,
                "year": int(chunk_row["fiscal_year"]),
                "company": chunk_row["company"],
                "doc_id": chunk_row["doc_id"],
                "page_start": int(chunk_row["page_start"]),
                "page_end": int(chunk_row["page_end"]),
                "created_at": created_at,
            }
        )

    return records


def extract_triplets_from_chunk(
    chunk_row: pd.Series,
    llm_client: LLMClient,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Run triplet extraction on a single chunk."""
    user_prompt = build_user_prompt(chunk_row)

    llm_result = llm_client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        task_name="triplet_extraction",
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

    return parse_triplets_response(
        llm_parsed_json=parsed_json,
        chunk_row=chunk_row,
        created_at=created_at,
    )


def run_triplet_extraction(
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
    End-to-end triplet extraction pipeline with resumability.

    Args:
        chunks_path: Path to chunks file.
        output_path: Path to output triplets file.
        llm_model: LLM model name.
        llm_cache_path: Path to LLM cache file.
        mode: "all", "heuristic", or "scored".
        max_chunks: Maximum number of candidate chunks to process.
        save_every: Save results every N processed chunks.
        dry_run: If True, only count chunks and exit without LLM calls.
        verbose: If True, log progress.
        system_prompt_path: Optional path to a file containing the system prompt.
        min_pattern_score: Minimum score to keep a chunk (scored mode only).

    Returns:
        DataFrame with extracted triplets.
    """
    chunks_df = load_chunks(chunks_path)
    existing_triplets_df = load_existing_triplets(output_path)
    checkpoint_path = get_checkpoint_path(output_path)
    checkpoint_df = load_or_bootstrap_checkpoint(
        checkpoint_path=checkpoint_path,
        existing_output_df=existing_triplets_df,
    )

    candidate_chunks_df = select_candidate_chunks(
        chunks_df=chunks_df,
        mode=mode,
        max_chunks=max_chunks,
        min_pattern_score=min_pattern_score,
    )

    remaining_chunks_df = filter_unprocessed_chunks(
        candidate_chunks_df=candidate_chunks_df,
        existing_triplets_df=existing_triplets_df,
        checkpoint_df=checkpoint_df,
    )

    if verbose:
        logger.info("Triplet extraction setup:")
        logger.info(f"Total chunks available       : {len(chunks_df)}")
        logger.info(f"Candidate chunks selected    : {len(candidate_chunks_df)}")
        logger.info(f"Already processed chunk_ids  : {existing_triplets_df['chunk_id'].nunique() if not existing_triplets_df.empty else 0}")
        logger.info(f"Chunks remaining to process  : {len(remaining_chunks_df)}")

    if remaining_chunks_df.empty:
        logger.info("No remaining chunks to process.")
        return existing_triplets_df

    if dry_run:
        logger.info("Dry run: exiting without LLM calls.")
        return existing_triplets_df

    # Load system prompt (once)
    system_prompt = build_system_prompt(system_prompt_path)

    llm_client = LLMClient(
        model=llm_model,
        temperature=0.0,
        max_output_tokens=700,
        cache_path=llm_cache_path,
    )

    all_triplets_df = existing_triplets_df.copy()
    all_checkpoint_df = checkpoint_df.copy()
    new_records_buffer: list[dict[str, Any]] = []
    new_checkpoint_records: list[dict[str, Any]] = []

    for idx, chunk_row in enumerate(remaining_chunks_df.to_dict(orient="records"), start=1):
        chunk_series = pd.Series(chunk_row)

        try:
            records = extract_triplets_from_chunk(
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
                all_triplets_df = pd.concat([all_triplets_df, new_df], ignore_index=True)
                all_triplets_df = all_triplets_df.drop_duplicates(
                    subset=["chunk_id", "entity_a", "relation", "entity_b"],
                    keep="last",
                ).reset_index(drop=True)

                save_triplets(all_triplets_df, output_path)

            if new_checkpoint_records:
                all_checkpoint_df = upsert_checkpoint_records(
                    checkpoint_df=all_checkpoint_df,
                    new_records=new_checkpoint_records,
                )
                save_checkpoint(all_checkpoint_df, checkpoint_path)

            if verbose:
                logger.info(
                    f"Saved progress: processed {idx}/{len(remaining_chunks_df)} remaining chunks | "
                    f"total triplet rows={len(all_triplets_df)}"
                )
            new_records_buffer = []
            new_checkpoint_records = []

    return all_triplets_df


def print_summary(triplets_df: pd.DataFrame) -> None:
    """Print a short summary of extracted triplets using logging."""
    logger.info("Triplet extraction completed.")
    logger.info(f"Total extracted triplet rows: {len(triplets_df)}")

    if triplets_df.empty:
        return

    logger.info("Counts by relation:")
    counts_df = triplets_df.groupby("relation")["entity_a"].count().reset_index(name="n_triplets")
    logger.info("\n" + counts_df.to_string(index=False))

    logger.info("Preview:")
    preview_cols = [
        "chunk_id",
        "entity_a",
        "relation",
        "entity_b",
        "company",
        "year",
    ]
    logger.info("\n" + triplets_df[preview_cols].head(15).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract relation triplets from report chunks.")
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to chunks file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/triplets.parquet",
        help="Path to output triplets file.",
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
        triplets_df = run_triplet_extraction(
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
        logger.error(f"Triplet extraction failed: {e}")
        sys.exit(1)

    if not args.dry_run:
        print_summary(triplets_df)
        logger.info(f"Saved to: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
