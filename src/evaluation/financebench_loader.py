from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/PatronusAI/financebench/raw/main/financebench_merged.jsonl"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "financebench"
DEFAULT_RAW_PATH = DEFAULT_DATA_DIR / "raw" / "financebench_merged.jsonl"
DEFAULT_FULL_OUTPUT_PATH = DEFAULT_DATA_DIR / "financebench_full.parquet"
DEFAULT_CORE40_OUTPUT_PATH = DEFAULT_DATA_DIR / "financebench_subset_core40.parquet"
DEFAULT_LOCAL_SMOKE_OUTPUT_PATH = DEFAULT_DATA_DIR / "financebench_subset_local_smoke.parquet"
DEFAULT_DOCS_MANIFEST_PATH = DEFAULT_DATA_DIR / "financebench_subset_core40_docs.csv"
DEFAULT_README_PATH = DEFAULT_DATA_DIR / "README.md"

LOCAL_SMOKE_SCOPE: tuple[tuple[str, int], ...] = (
    ("Adobe", 2022),
    ("Adobe", 2023),
    ("Adobe", 2024),
    ("Lockheed Martin", 2022),
    ("Lockheed Martin", 2023),
    ("Lockheed Martin", 2024),
    ("Pfizer", 2022),
    ("Pfizer", 2023),
    ("Pfizer", 2024),
)

CORE40_DOC_LIMITS: list[tuple[str, int, int]] = [
    ("AMD", 2022, 6),
    ("American Express", 2022, 6),
    ("Boeing", 2022, 5),
    ("PepsiCo", 2022, 5),
    ("Amcor", 2023, 4),
    ("3M", 2022, 3),
    ("AES Corporation", 2022, 3),
    ("Adobe", 2022, 2),
    ("Best Buy", 2023, 2),
    ("Pfizer", 2021, 3),
    ("Lockheed Martin", 2022, 1),
]

QUESTION_TYPE_PRIORITY = {
    "domain-relevant": 0,
    "metrics-generated": 1,
    "novel-generated": 2,
}

NORMALIZED_COLUMNS = [
    "financebench_id",
    "company",
    "company_slug",
    "doc_name",
    "expected_doc_id",
    "question_type",
    "question_reasoning",
    "domain_question_num",
    "question",
    "query_text",
    "answer",
    "expected_answer",
    "justification",
    "dataset_subset_label",
    "gics_sector",
    "doc_type",
    "doc_period",
    "doc_link",
    "evidence_count",
    "evidence_pages",
    "primary_evidence_page",
    "primary_evidence_doc_name",
    "primary_evidence_text",
    "question_length_chars",
    "has_justification",
]

logger = logging.getLogger(__name__)


def clean_text(value: object) -> str:
    """Convert a raw value to a stripped string."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def slugify_name(value: str) -> str:
    """Create a lowercase alphanumeric slug without separators."""
    return "".join(ch.lower() for ch in value if ch.isalnum())


def normalize_doc_name(doc_name: str) -> str:
    """Normalize a FinanceBench document name into a local doc_id-like string."""
    normalized_chars = []
    previous_was_separator = False

    for ch in doc_name.lower():
        if ch.isalnum():
            normalized_chars.append(ch)
            previous_was_separator = False
            continue

        if not previous_was_separator:
            normalized_chars.append("_")
            previous_was_separator = True

    normalized = "".join(normalized_chars).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def format_company_year_scope(
    scope: Iterable[tuple[str, int]],
    *,
    empty_label: str = "no configured company/year pairs",
) -> str:
    """Format company/year tuples as a stable human-readable scope description."""
    grouped_years: dict[str, list[int]] = {}

    for company, year in scope:
        grouped_years.setdefault(company, []).append(int(year))

    if not grouped_years:
        return empty_label

    parts: list[str] = []
    for company in sorted(grouped_years):
        years = sorted(set(grouped_years[company]))
        if len(years) >= 2 and years == list(range(years[0], years[-1] + 1)):
            parts.append(f"`{company} {years[0]}-{years[-1]}`")
        else:
            year_label = ", ".join(str(year) for year in years)
            parts.append(f"`{company} {year_label}`")

    return ", ".join(parts)


def extract_local_smoke_scope_pairs(local_smoke_df: pd.DataFrame) -> list[tuple[str, int]]:
    """Return the actual company/year pairs present in the computed local_smoke subset."""
    if local_smoke_df.empty:
        return []

    pairs = {
        (str(row.company), int(row.doc_period))
        for row in local_smoke_df[["company", "doc_period"]].dropna().itertuples(index=False)
    }
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def download_source_jsonl(source_url: str, destination_path: Path, force: bool = False) -> Path:
    """Download the FinanceBench JSONL file if it is missing or force=True."""
    if destination_path.exists() and not force:
        logger.info("Using existing FinanceBench raw file: %s", destination_path.resolve())
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        source_url,
        headers={
            "User-Agent": "financial-rag-llm-judge/0.1 (+https://huggingface.co/datasets/PatronusAI/financebench)"
        },
    )

    try:
        with urlopen(request, timeout=60) as response:
            destination_path.write_bytes(response.read())
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download FinanceBench (HTTP {exc.code}) from {source_url}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download FinanceBench from {source_url}: {exc}") from exc

    logger.info("Downloaded FinanceBench raw file to: %s", destination_path.resolve())
    return destination_path


def load_financebench_records(raw_path: Path) -> list[dict]:
    """Load the raw JSONL dataset into a list of records."""
    if not raw_path.exists():
        raise FileNotFoundError(f"FinanceBench raw file not found: {raw_path}")

    records: list[dict] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {raw_path}") from exc

    if not records:
        raise ValueError(f"No records found in {raw_path}")

    return records


def normalize_financebench_records(records: Iterable[dict]) -> pd.DataFrame:
    """Normalize FinanceBench records into a consistent tabular format."""
    normalized_rows: list[dict] = []

    for record in records:
        company = clean_text(record.get("company"))
        doc_name = clean_text(record.get("doc_name"))
        question = clean_text(record.get("question"))
        answer = clean_text(record.get("answer"))
        justification = clean_text(record.get("justification"))
        question_reasoning = clean_text(record.get("question_reasoning"))
        doc_type = clean_text(record.get("doc_type")).lower()
        doc_period = record.get("doc_period")
        if doc_period in ("", None):
            normalized_doc_period = pd.NA
        else:
            normalized_doc_period = int(doc_period)

        raw_evidence = record.get("evidence")
        evidence_items = raw_evidence if isinstance(raw_evidence, list) else []
        evidence_pages = sorted(
            {
                int(item["evidence_page_num"])
                for item in evidence_items
                if isinstance(item, dict) and item.get("evidence_page_num") not in (None, "")
            }
        )
        primary_evidence = evidence_items[0] if evidence_items else {}
        primary_evidence_text = ""
        primary_evidence_doc_name = ""
        primary_evidence_page = pd.NA

        if isinstance(primary_evidence, dict):
            primary_evidence_text = clean_text(primary_evidence.get("evidence_text"))
            primary_evidence_doc_name = clean_text(primary_evidence.get("doc_name"))
            primary_page_value = primary_evidence.get("evidence_page_num")
            if primary_page_value not in (None, ""):
                primary_evidence_page = int(primary_page_value)

        normalized_rows.append(
            {
                "financebench_id": clean_text(record.get("financebench_id")),
                "company": company,
                "company_slug": slugify_name(company),
                "doc_name": doc_name,
                "expected_doc_id": normalize_doc_name(doc_name),
                "question_type": clean_text(record.get("question_type")),
                "question_reasoning": question_reasoning,
                "domain_question_num": clean_text(record.get("domain_question_num")),
                "question": question,
                "query_text": question,
                "answer": answer,
                "expected_answer": answer,
                "justification": justification,
                "dataset_subset_label": clean_text(record.get("dataset_subset_label")),
                "gics_sector": clean_text(record.get("gics_sector")),
                "doc_type": doc_type,
                "doc_period": normalized_doc_period,
                "doc_link": clean_text(record.get("doc_link")),
                "evidence_count": len(evidence_items),
                "evidence_pages": "|".join(str(page) for page in evidence_pages),
                "primary_evidence_page": primary_evidence_page,
                "primary_evidence_doc_name": primary_evidence_doc_name or doc_name,
                "primary_evidence_text": primary_evidence_text,
                "question_length_chars": len(question),
                "has_justification": bool(justification),
            }
        )

    df = pd.DataFrame(normalized_rows)
    if df.empty:
        raise ValueError("FinanceBench normalization produced an empty DataFrame.")

    df["doc_period"] = df["doc_period"].astype("Int64")
    df["primary_evidence_page"] = df["primary_evidence_page"].astype("Int64")
    df["evidence_count"] = df["evidence_count"].astype(int)
    df["question_length_chars"] = df["question_length_chars"].astype(int)
    df["has_justification"] = df["has_justification"].astype(bool)

    df = df.sort_values(
        by=["company", "doc_period", "doc_name", "financebench_id"], ascending=True
    ).reset_index(drop=True)

    missing_columns = [column for column in NORMALIZED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing normalized columns: {missing_columns}")

    return df[NORMALIZED_COLUMNS]


def build_selection_sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a sorted copy that prefers answerable and better-structured questions."""
    sorted_df = df.copy()
    sorted_df["_question_type_priority"] = sorted_df["question_type"].map(
        QUESTION_TYPE_PRIORITY
    ).fillna(99)
    sorted_df["_reasoning_missing"] = (
        sorted_df["question_reasoning"].fillna("").astype(str).str.strip() == ""
    ).astype(int)
    sorted_df["_doc_link_missing"] = (
        sorted_df["doc_link"].fillna("").astype(str).str.strip() == ""
    ).astype(int)

    sorted_df = sorted_df.sort_values(
        by=[
            "_question_type_priority",
            "_reasoning_missing",
            "_doc_link_missing",
            "evidence_count",
            "question_length_chars",
            "financebench_id",
        ],
        ascending=[True, True, True, False, True, True],
    ).reset_index(drop=True)

    return sorted_df.drop(
        columns=["_question_type_priority", "_reasoning_missing", "_doc_link_missing"]
    )


def select_subset_by_doc_limits(
    df: pd.DataFrame,
    doc_limits: Iterable[tuple[str, int, int]],
    subset_name: str,
) -> pd.DataFrame:
    """Select a deterministic subset using per-document quotas."""
    selected_parts: list[pd.DataFrame] = []

    for company, doc_period, limit in doc_limits:
        mask = (
            df["company"].eq(company)
            & df["doc_period"].eq(doc_period)
            & df["doc_type"].str.lower().eq("10k")
        )
        matching_df = build_selection_sort_frame(df.loc[mask].copy())

        if len(matching_df) < limit:
            raise ValueError(
                f"Requested {limit} questions for {company} {doc_period}, "
                f"but only found {len(matching_df)}."
            )

        picked_df = matching_df.head(limit).copy()
        picked_df["subset_name"] = subset_name
        picked_df["subset_group"] = f"{normalize_doc_name(company)}_{doc_period}_10k"
        picked_df["selection_note"] = "curated_doc_quota"
        picked_df["subset_rank"] = range(1, len(picked_df) + 1)
        selected_parts.append(picked_df)

    subset_df = pd.concat(selected_parts, ignore_index=True)
    subset_df = subset_df.drop_duplicates(subset=["financebench_id"]).reset_index(drop=True)
    subset_df["global_subset_rank"] = range(1, len(subset_df) + 1)

    return subset_df


def select_core40_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the main 40-question evaluation subset."""
    candidates_df = df[df["doc_type"].str.lower().eq("10k")].copy()
    subset_df = select_subset_by_doc_limits(
        candidates_df,
        doc_limits=CORE40_DOC_LIMITS,
        subset_name="core40",
    )

    if len(subset_df) != 40:
        raise ValueError(f"core40 subset must contain 40 rows, found {len(subset_df)}")

    return subset_df.reset_index(drop=True)


def select_local_smoke_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Select the exact overlap with the current local corpus."""
    candidates_df = df[df["doc_type"].str.lower().eq("10k")].copy()
    local_smoke_scope_df = pd.DataFrame(LOCAL_SMOKE_SCOPE, columns=["company", "doc_period"])
    mask_df = candidates_df.merge(
        local_smoke_scope_df,
        on=["company", "doc_period"],
        how="inner",
    )
    subset_df = build_selection_sort_frame(mask_df).reset_index(drop=True)
    subset_df["subset_name"] = "local_smoke"
    subset_df["subset_group"] = "current_local_corpus_overlap"
    subset_df["selection_note"] = "exact_local_corpus_scope_match"
    subset_df["subset_rank"] = range(1, len(subset_df) + 1)
    subset_df["global_subset_rank"] = range(1, len(subset_df) + 1)
    return subset_df


def build_docs_manifest(subset_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the source documents needed to run a subset."""
    rows = []
    grouped = subset_df.groupby(
        ["company", "doc_period", "doc_type", "doc_name", "expected_doc_id", "doc_link"],
        dropna=False,
        sort=True,
    )

    for group_values, group_df in grouped:
        company, doc_period, doc_type, doc_name, expected_doc_id, doc_link = group_values
        question_types = ", ".join(sorted(group_df["question_type"].dropna().unique().tolist()))
        financebench_ids = ", ".join(group_df["financebench_id"].tolist())

        rows.append(
            {
                "company": company,
                "doc_period": int(doc_period),
                "doc_type": doc_type,
                "doc_name": doc_name,
                "expected_doc_id": expected_doc_id,
                "doc_link": doc_link,
                "n_questions": len(group_df),
                "question_types": question_types,
                "financebench_ids": financebench_ids,
            }
        )

    docs_df = pd.DataFrame(rows)
    if docs_df.empty:
        return docs_df

    docs_df = docs_df.sort_values(
        by=["n_questions", "company", "doc_period"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return docs_df


def save_dataframe_variants(df: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Save a DataFrame to Parquet and sibling CSV files."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


def render_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a simple markdown table without extra dependencies."""

    def sanitize(cell: object) -> str:
        text = clean_text(cell)
        return text.replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(sanitize(cell) for cell in row) + " |")

    return "\n".join(lines)


def build_distribution_rows(df: pd.DataFrame, column_name: str) -> list[list[object]]:
    """Build count/share rows for a categorical column."""
    counts = df[column_name].fillna("").replace("", "<empty>").value_counts()
    total = len(df)
    rows = []
    for value, count in counts.items():
        share = 100.0 * count / total if total else 0.0
        rows.append([value, count, f"{share:.1f}%"])
    return rows


def build_readme_text(
    full_df: pd.DataFrame,
    core40_df: pd.DataFrame,
    local_smoke_df: pd.DataFrame,
    docs_manifest_df: pd.DataFrame,
    source_url: str,
) -> str:
    """Build the markdown documentation stored next to the subset artifacts."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    local_scope_label = format_company_year_scope(LOCAL_SMOKE_SCOPE)
    actual_overlap_label = format_company_year_scope(
        extract_local_smoke_scope_pairs(local_smoke_df),
        empty_label="no overlap in the current open-source FinanceBench sample",
    )

    file_rows = [
        ["`financebench_full.parquet`", "Normalized full 150-row open-source FinanceBench sample."],
        ["`financebench_subset_core40.parquet`", "Main 40-question evaluation subset for the project."],
        ["`financebench_subset_core40_docs.csv`", "Unique 10-K filings required to execute `core40`."],
        [
            "`financebench_subset_local_smoke.parquet`",
            "Exact overlap between the FinanceBench sample and the local corpus scope defined in this loader.",
        ],
    ]

    tenk_df = full_df[full_df["doc_type"].str.lower().eq("10k")]
    summary_rows = [
        ["full sample", len(full_df), full_df["doc_name"].nunique(), full_df["company"].nunique()],
        ["10-K only", len(tenk_df), tenk_df["doc_name"].nunique(), tenk_df["company"].nunique()],
        ["core40", len(core40_df), core40_df["doc_name"].nunique(), core40_df["company"].nunique()],
        [
            "local_smoke",
            len(local_smoke_df),
            local_smoke_df["doc_name"].nunique(),
            local_smoke_df["company"].nunique(),
        ],
    ]

    docs_rows = []
    for row in docs_manifest_df.itertuples(index=False):
        docs_rows.append([row.company, row.doc_period, row.doc_name, row.n_questions, row.question_types])

    lines = [
        "# FinanceBench Evaluation Artifacts",
        "",
        f"Generated by `src/evaluation/financebench_loader.py` on {generated_at}.",
        "",
        "## Source",
        "",
        f"- Dataset: [PatronusAI/financebench]({source_url})",
        "- Format used here: raw JSONL sample published on Hugging Face",
        "- Public sample size: 150 questions",
        "- License on the dataset card: `CC-BY-NC-4.0`",
        "",
        "## Why two subsets?",
        "",
        "- `core40` is the recommended subset: 40 questions, 10-K only, limited number of source filings, and a mix of analyst-style prompts.",
        "- `local_smoke` is intentionally tiny: it contains only the exact company/year overlap with the documents already present in this repository.",
        f"- The exact current overlap is {len(local_smoke_df)} questions, which is too small for a serious evaluation but useful for smoke tests.",
        "",
        "## Files",
        "",
        render_markdown_table(["File", "Purpose"], file_rows),
        "",
        "## Coverage Summary",
        "",
        render_markdown_table(["Scope", "Questions", "Source docs", "Companies"], summary_rows),
        "",
        "## core40 Selection Rules",
        "",
        "- Only `10-K` filings are kept in the main subset.",
        "- The subset is document-quota based: it prefers a small number of filings with several questions each, so later ingestion stays manageable.",
        "- Within each filing, rows are sorted to prefer `domain-relevant`, then `metrics-generated`, then `novel-generated` questions, with a preference for rows that expose explicit reasoning metadata.",
        "- The current `core40` needs 11 unique source filings.",
        "",
        "## core40 Source Documents",
        "",
        render_markdown_table(
            ["Company", "Year", "Doc", "Questions", "Question types"],
            docs_rows,
        ),
        "",
        "## core40 Question-Type Distribution",
        "",
        render_markdown_table(
            ["Question type", "Count", "Share"],
            build_distribution_rows(core40_df, "question_type"),
        ),
        "",
        "## core40 Reasoning Distribution",
        "",
        render_markdown_table(
            ["Reasoning", "Count", "Share"],
            build_distribution_rows(core40_df, "question_reasoning"),
        ),
        "",
        "## local_smoke Notes",
        "",
        f"- `local_smoke` is filtered on this configured local corpus scope: {local_scope_label}.",
        f"- In the current open-source FinanceBench sample, that scope yields this actual overlap: {actual_overlap_label}.",
        "- Keep `local_smoke` for pipeline debugging; use `core40` for the real evaluation loop once the missing filings are ingested.",
        "",
    ]

    return "\n".join(lines)


def write_readme(
    readme_path: Path,
    full_df: pd.DataFrame,
    core40_df: pd.DataFrame,
    local_smoke_df: pd.DataFrame,
    docs_manifest_df: pd.DataFrame,
    source_url: str,
) -> None:
    """Write the FinanceBench subset documentation markdown."""
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        build_readme_text(
            full_df=full_df,
            core40_df=core40_df,
            local_smoke_df=local_smoke_df,
            docs_manifest_df=docs_manifest_df,
            source_url=source_url,
        ),
        encoding="utf-8",
    )


def print_summary(full_df: pd.DataFrame, core40_df: pd.DataFrame, local_smoke_df: pd.DataFrame) -> None:
    """Log a short execution summary."""
    logger.info("FinanceBench normalization complete.")
    logger.info("Full sample rows: %s", len(full_df))
    logger.info(
        "10-K rows in full sample: %s",
        int(full_df["doc_type"].str.lower().eq("10k").sum()),
    )
    logger.info("core40 rows: %s", len(core40_df))
    logger.info("local_smoke rows: %s", len(local_smoke_df))
    logger.info(
        "local_smoke companies/years: %s",
        sorted(
            {
                (row.company, int(row.doc_period))
                for row in local_smoke_df[["company", "doc_period"]].itertuples(index=False)
            }
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, normalize and subset the open FinanceBench sample."
    )
    parser.add_argument(
        "--source_url",
        type=str,
        default=DEFAULT_SOURCE_URL,
        help="Source URL for the FinanceBench raw JSONL file.",
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=str(DEFAULT_RAW_PATH),
        help="Local path of the raw JSONL file.",
    )
    parser.add_argument(
        "--full_output_path",
        type=str,
        default=str(DEFAULT_FULL_OUTPUT_PATH),
        help="Output path for the normalized full dataset (.parquet).",
    )
    parser.add_argument(
        "--core40_output_path",
        type=str,
        default=str(DEFAULT_CORE40_OUTPUT_PATH),
        help="Output path for the main 40-question subset (.parquet).",
    )
    parser.add_argument(
        "--local_smoke_output_path",
        type=str,
        default=str(DEFAULT_LOCAL_SMOKE_OUTPUT_PATH),
        help="Output path for the local smoke subset (.parquet).",
    )
    parser.add_argument(
        "--docs_manifest_path",
        type=str,
        default=str(DEFAULT_DOCS_MANIFEST_PATH),
        help="Output CSV path for the core40 document manifest.",
    )
    parser.add_argument(
        "--readme_path",
        type=str,
        default=str(DEFAULT_README_PATH),
        help="Output markdown path for subset documentation.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download the raw JSONL even if it already exists locally.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Fail instead of downloading when the raw JSONL is missing.",
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

    raw_path = Path(args.raw_path)
    full_output_path = Path(args.full_output_path)
    core40_output_path = Path(args.core40_output_path)
    local_smoke_output_path = Path(args.local_smoke_output_path)
    docs_manifest_path = Path(args.docs_manifest_path)
    readme_path = Path(args.readme_path)

    if not raw_path.exists():
        if args.skip_download:
            raise FileNotFoundError(
                f"FinanceBench raw file does not exist and --skip_download was set: {raw_path}"
            )
        download_source_jsonl(
            source_url=args.source_url,
            destination_path=raw_path,
            force=args.force_download,
        )
    elif args.force_download:
        download_source_jsonl(
            source_url=args.source_url,
            destination_path=raw_path,
            force=True,
        )

    records = load_financebench_records(raw_path)
    full_df = normalize_financebench_records(records)
    core40_df = select_core40_subset(full_df)
    local_smoke_df = select_local_smoke_subset(full_df)
    docs_manifest_df = build_docs_manifest(core40_df)

    save_dataframe_variants(full_df, full_output_path)
    save_dataframe_variants(core40_df, core40_output_path)
    save_dataframe_variants(local_smoke_df, local_smoke_output_path)

    docs_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    docs_manifest_df.to_csv(docs_manifest_path, index=False)

    write_readme(
        readme_path=readme_path,
        full_df=full_df,
        core40_df=core40_df,
        local_smoke_df=local_smoke_df,
        docs_manifest_df=docs_manifest_df,
        source_url=args.source_url,
    )

    print_summary(full_df=full_df, core40_df=core40_df, local_smoke_df=local_smoke_df)
    logger.info("Artifacts written under: %s", full_output_path.parent.resolve())


if __name__ == "__main__":
    main()
