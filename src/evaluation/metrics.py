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

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "local_smoke"
    / "improved"
    / "evaluation_runs_judged.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "local_smoke" / "improved" / "analysis"

REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "generated_answer",
    "citations",
    "retrieved_context",
    "n_retrieved_chunks",
    "judge_verdict",
    "judge_justification",
}


def clean_text(value: Any) -> str:
    """Normalize arbitrary text input."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_json_list(value: Any) -> list[Any]:
    """Parse a JSON-serialized list field, falling back gracefully."""
    if value is None:
        return []
    if isinstance(value, list):
        return value

    text = clean_text(value)
    if text == "":
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    return parsed if isinstance(parsed, list) else []


def load_judged_runs(input_path: Path) -> pd.DataFrame:
    """Load judged evaluation runs and enrich them with analysis-ready columns."""
    df = read_table(input_path)
    if df.empty:
        raise ValueError(f"Judged runs file is empty: {input_path}")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Judged runs file is missing columns: {sorted(missing_columns)}")

    working_df = df.copy()
    text_columns = [
        "question",
        "generated_answer",
        "retrieved_context",
        "judge_verdict",
        "judge_justification",
        "status",
        "judge_status",
        "retrieval_mode",
    ]
    for column in text_columns:
        if column in working_df.columns:
            working_df[column] = working_df[column].fillna("").astype(str).str.strip()

    if "reference_answer" in working_df.columns:
        working_df["reference_answer"] = (
            working_df["reference_answer"].fillna("").astype(str).str.strip()
        )
    else:
        working_df["reference_answer"] = ""

    if "question_type" not in working_df.columns:
        working_df["question_type"] = ""
    else:
        working_df["question_type"] = working_df["question_type"].fillna("").astype(str).str.strip()

    if "company" not in working_df.columns:
        working_df["company"] = ""
    else:
        working_df["company"] = working_df["company"].fillna("").astype(str).str.strip()

    working_df["n_retrieved_chunks"] = pd.to_numeric(
        working_df["n_retrieved_chunks"], errors="coerce"
    ).fillna(0).astype(int)
    working_df["citations_list"] = working_df["citations"].apply(parse_json_list)
    working_df["n_citations"] = working_df["citations_list"].apply(len)
    working_df["has_citations"] = working_df["n_citations"] > 0
    working_df["is_yes"] = working_df["judge_verdict"].str.lower().eq("yes")
    working_df["is_no"] = working_df["judge_verdict"].str.lower().eq("no")
    working_df["generated_answer_empty"] = working_df["generated_answer"] == ""
    working_df["retrieved_context_empty"] = working_df["retrieved_context"] == ""
    working_df["context_length_chars"] = working_df["retrieved_context"].str.len()
    working_df["answer_length_chars"] = working_df["generated_answer"].str.len()
    working_df["no_citation"] = ~working_df["has_citations"]

    return working_df.reset_index(drop=True)


def classify_probable_error_cause(row: pd.Series) -> str:
    """Heuristic assignment of one probable error cause for failed examples."""
    if not bool(row.get("is_no", False)):
        return ""

    if bool(row.get("generated_answer_empty", False)) or clean_text(row.get("status")) == "error":
        return "generation imprecise"

    if bool(row.get("retrieved_context_empty", False)) or int(row.get("n_retrieved_chunks", 0)) == 0:
        return "mauvais retrieval"

    if clean_text(row.get("retrieval_mode")) == "lexical_fallback":
        return "mauvais retrieval"

    context_length = int(row.get("context_length_chars", 0))
    n_chunks = int(row.get("n_retrieved_chunks", 0))
    has_citations = bool(row.get("has_citations", False))

    if n_chunks <= 2 or context_length < 1200:
        return "contexte incomplet"

    if n_chunks >= 5 and context_length >= 6000 and not has_citations:
        return "chunking inadequat"

    return "generation imprecise"


def build_results_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Create the final per-question analysis table."""
    results_df = runs_df.copy()
    results_df["accuracy_hit"] = results_df["is_yes"].astype(int)
    results_df["probable_error_cause"] = results_df.apply(classify_probable_error_cause, axis=1)

    ordered_columns = [
        "question_id",
        "company",
        "question_type",
        "question",
        "judge_verdict",
        "judge_justification",
        "accuracy_hit",
        "no_citation",
        "n_citations",
        "n_retrieved_chunks",
        "retrieval_mode",
        "probable_error_cause",
        "generated_answer",
        "reference_answer",
        "status",
        "judge_status",
    ]
    existing_columns = [column for column in ordered_columns if column in results_df.columns]
    trailing_columns = [column for column in results_df.columns if column not in existing_columns]
    return results_df[existing_columns + trailing_columns].reset_index(drop=True)


def compute_summary_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level aggregate metrics."""
    total_questions = int(len(results_df))
    number_of_yes = int(results_df["is_yes"].sum())
    number_of_no = int(results_df["is_no"].sum())
    number_without_citation = int(results_df["no_citation"].sum())

    accuracy = (number_of_yes / total_questions) if total_questions else 0.0
    no_citation_rate = (number_without_citation / total_questions) if total_questions else 0.0

    error_causes = (
        results_df.loc[results_df["is_no"], "probable_error_cause"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .to_dict()
    )
    verdict_counts = results_df["judge_verdict"].replace("", "<missing>").value_counts().to_dict()

    return {
        "generated_at_utc": now_utc_iso(),
        "total_questions": total_questions,
        "number_of_yes": number_of_yes,
        "number_of_no": number_of_no,
        "accuracy": accuracy,
        "number_without_citation": number_without_citation,
        "no_citation_rate": no_citation_rate,
        "verdict_counts": verdict_counts,
        "probable_error_causes": error_causes,
    }


def select_example_rows(
    results_df: pd.DataFrame,
    verdict: str,
    limit: int = 3,
) -> pd.DataFrame:
    """Select a few representative success or failure examples."""
    verdict_mask = results_df["judge_verdict"].str.lower().eq(verdict.lower())
    example_df = results_df.loc[verdict_mask].copy()

    if example_df.empty:
        return example_df

    if verdict.lower() == "yes":
        example_df = example_df.sort_values(
            by=["n_citations", "answer_length_chars", "n_retrieved_chunks"],
            ascending=[False, False, False],
        )
    else:
        example_df = example_df.sort_values(
            by=["probable_error_cause", "answer_length_chars", "n_retrieved_chunks"],
            ascending=[True, False, False],
        )

    columns = [
        "question_id",
        "company",
        "question",
        "generated_answer",
        "judge_verdict",
        "judge_justification",
        "probable_error_cause",
        "n_citations",
        "n_retrieved_chunks",
    ]
    existing_columns = [column for column in columns if column in example_df.columns]
    return example_df[existing_columns].head(limit).reset_index(drop=True)


def save_dataframe_bundle(df: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Save one dataframe as parquet and csv."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple markdown table."""
    def sanitize(value: Any) -> str:
        return clean_text(value).replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(sanitize(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_distribution_rows(series: pd.Series) -> list[list[Any]]:
    """Build count/share rows from a categorical series."""
    counts = series.replace("", "<empty>").fillna("<empty>").value_counts()
    total = int(counts.sum())
    rows: list[list[Any]] = []
    for value, count in counts.items():
        share = (100.0 * count / total) if total else 0.0
        rows.append([value, int(count), f"{share:.1f}%"])
    return rows


def dataframe_to_example_rows(df: pd.DataFrame) -> list[list[Any]]:
    """Convert an example dataframe to markdown table rows."""
    if df.empty:
        return [["None", "", "", "", ""]]

    rows: list[list[Any]] = []
    for row in df.itertuples(index=False):
        rows.append(
            [
                getattr(row, "question_id", ""),
                getattr(row, "company", ""),
                getattr(row, "judge_verdict", ""),
                getattr(row, "probable_error_cause", ""),
                getattr(row, "judge_justification", ""),
            ]
        )
    return rows


def write_summary_report(
    output_path: Path,
    summary_metrics: dict[str, Any],
    results_df: pd.DataFrame,
    good_examples_df: pd.DataFrame,
    failure_examples_df: pd.DataFrame,
    source_input_path: Path,
) -> None:
    """Write a compact markdown report for the evaluation analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headline_rows = [
        ["Total questions", summary_metrics["total_questions"]],
        ["Accuracy", f"{100.0 * summary_metrics['accuracy']:.1f}%"],
        ["Yes verdicts", summary_metrics["number_of_yes"]],
        ["No verdicts", summary_metrics["number_of_no"]],
        ["No-citation rate", f"{100.0 * summary_metrics['no_citation_rate']:.1f}%"],
        ["Rows without citations", summary_metrics["number_without_citation"]],
    ]

    verdict_rows = build_distribution_rows(results_df["judge_verdict"])
    citation_rows = build_distribution_rows(results_df["no_citation"].map({True: "No citation", False: "Has citation"}))
    error_cause_rows = build_distribution_rows(
        results_df.loc[results_df["is_no"], "probable_error_cause"]
    )

    lines = [
        "# Evaluation Metrics",
        "",
        f"Generated from `{source_input_path.resolve()}` on {summary_metrics['generated_at_utc']}.",
        "",
        "## Headline Metrics",
        "",
        render_markdown_table(["Metric", "Value"], headline_rows),
        "",
        "## Verdict Distribution",
        "",
        render_markdown_table(["Verdict", "Count", "Share"], verdict_rows),
        "",
        "## Citation Coverage",
        "",
        render_markdown_table(["Category", "Count", "Share"], citation_rows),
        "",
        "## Probable Error Causes",
        "",
        render_markdown_table(["Cause", "Count", "Share"], error_cause_rows),
        "",
        "## Good Examples",
        "",
        render_markdown_table(
            ["Question ID", "Company", "Verdict", "Probable cause", "Judge justification"],
            dataframe_to_example_rows(good_examples_df),
        ),
        "",
        "## Failure Examples",
        "",
        render_markdown_table(
            ["Question ID", "Company", "Verdict", "Probable cause", "Judge justification"],
            dataframe_to_example_rows(failure_examples_df),
        ),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_optional_charts(results_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save a few simple charts when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.info("matplotlib is not available. Skipping chart generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    verdict_counts = results_df["judge_verdict"].replace("", "<missing>").value_counts()
    verdict_path = output_dir / "verdict_distribution.png"
    ax = verdict_counts.plot(kind="bar", figsize=(7, 4), color=["#2e8b57", "#b22222", "#708090"])
    ax.set_title("Judge Verdict Distribution")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Number of questions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(verdict_path, dpi=200)
    plt.close()
    chart_paths.append(verdict_path)

    citation_counts = results_df["no_citation"].map({True: "No citation", False: "Has citation"}).value_counts()
    citation_path = output_dir / "citation_coverage.png"
    ax = citation_counts.plot(kind="bar", figsize=(7, 4), color=["#cc7722", "#1f77b4"])
    ax.set_title("Citation Coverage")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of questions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(citation_path, dpi=200)
    plt.close()
    chart_paths.append(citation_path)

    error_cause_series = (
        results_df.loc[results_df["is_no"], "probable_error_cause"]
        .replace("", pd.NA)
        .dropna()
    )
    if not error_cause_series.empty:
        error_path = output_dir / "probable_error_causes.png"
        ax = error_cause_series.value_counts().plot(
            kind="bar",
            figsize=(8, 4),
            color="#8c564b",
        )
        ax.set_title("Probable Error Causes")
        ax.set_xlabel("Cause")
        ax.set_ylabel("Number of failed questions")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(error_path, dpi=200)
        plt.close()
        chart_paths.append(error_path)

    return chart_paths


def run_metrics_pipeline(
    input_path: Path,
    output_dir: Path,
    example_limit: int = 3,
) -> dict[str, Any]:
    """Compute aggregate metrics, save artifacts and return the summary payload."""
    runs_df = load_judged_runs(input_path)
    results_df = build_results_table(runs_df)
    summary_metrics = compute_summary_metrics(results_df)
    good_examples_df = select_example_rows(results_df, verdict="Yes", limit=example_limit)
    failure_examples_df = select_example_rows(results_df, verdict="No", limit=example_limit)

    output_dir.mkdir(parents=True, exist_ok=True)

    final_results_path = output_dir / "final_results_table.parquet"
    good_examples_path = output_dir / "good_examples.parquet"
    failure_examples_path = output_dir / "failure_examples.parquet"
    summary_json_path = output_dir / "metrics_summary.json"
    summary_report_path = output_dir / "metrics_report.md"

    save_dataframe_bundle(results_df, final_results_path)
    save_dataframe_bundle(good_examples_df, good_examples_path)
    save_dataframe_bundle(failure_examples_df, failure_examples_path)

    write_summary_report(
        output_path=summary_report_path,
        summary_metrics=summary_metrics,
        results_df=results_df,
        good_examples_df=good_examples_df,
        failure_examples_df=failure_examples_df,
        source_input_path=input_path,
    )

    chart_paths = save_optional_charts(results_df, output_dir)
    summary_metrics["chart_paths"] = [str(path.resolve()) for path in chart_paths]

    summary_json_path.write_text(
        json.dumps(summary_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Metrics summary saved to: %s", summary_json_path.resolve())
    logger.info("Final results table saved to: %s", final_results_path.resolve())
    logger.info("Summary report saved to: %s", summary_report_path.resolve())

    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute global metrics from judged evaluation runs.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to judged evaluation runs (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--example_limit",
        type=int,
        default=3,
        help="Number of good/failure examples to keep.",
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
        run_metrics_pipeline(
            input_path=Path(args.input_path),
            output_dir=Path(args.output_dir),
            example_limit=args.example_limit,
        )
    except Exception as exc:
        logger.error("Metrics pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
