from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SOURCE_COMPANY_ALIASES = {
    "adobe": {"adobe", "adobe inc.", "adobe inc"},
    "lockheedmartin": {
        "lockheed martin",
        "lockheed martin corporation",
        "lockheedmartin",
    },
    "pfizer": {"pfizer", "pfizer inc.", "pfizer inc"},
}

COMPETITION_RISK_PATTERNS = [
    r"\bcompetition\b",
    r"\bcompetitive\b",
    r"\bcompetitor",
    r"\bcompete\b",
    r"\brisk\b",
    r"\brisk factors\b",
]

LEGAL_SUFFIX_PATTERNS = [
    r",?\s+inc\.?$",
    r",?\s+incorporated$",
    r",?\s+corp\.?$",
    r",?\s+corporation$",
    r",?\s+ltd\.?$",
    r",?\s+limited$",
    r",?\s+plc$",
    r",?\s+llc$",
    r",?\s+group$",
]

BLACKLIST_EXACT = {
    "adobe foundation",
    "european commission",
    "u.s. internal revenue service",
    "internal revenue service",
    "irs",
    "the metropolitan museum of art",
    "pfizer",
    "adobe",
    "lockheedmartin",
    "lockheed martin",
}

BLACKLIST_PATTERNS = [
    r"\bfoundation\b",
    r"\bcommission\b",
    r"\bdepartment\b",
    r"\bministry\b",
    r"\bgovernment\b",
    r"\bagency\b",
    r"\bauthorit",
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bmuseum\b",
    r"\binternal revenue service\b",
    r"\birs\b",
    r"\bcommissioner\b",
    r"\bassociation\b",
    r"\bcouncil\b",
    r"\badministration\b",
]

GENERIC_COMPETITOR_PATTERNS = [
    r"\bproducts?\b",
    r"\bservices?\b",
    r"\bsolutions?\b",
    r"\bsystems?\b",
    r"\bplatforms?\b",
    r"\bsoftware\b",
    r"\bpublishing\b",
    r"\bmarket\b",
    r"\bmarkets\b",
    r"\bcompany\b",
    r"\bcompanies\b",
]


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a dataframe from parquet or csv.
    """
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def normalize_text(text: str) -> str:
    """
    Normalize text for lightweight matching.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def canonicalize_source_company(source_company: str) -> str:
    """
    Map source company to canonical project identifier.
    """
    source_company = normalize_text(source_company)

    for canonical_name, aliases in SOURCE_COMPANY_ALIASES.items():
        if source_company == canonical_name or source_company in aliases:
            return canonical_name

    return source_company


def canonicalize_competitor_name(entity_text: str) -> str:
    """
    Normalize competitor names while preserving a readable form.
    """
    if not isinstance(entity_text, str):
        return ""

    text = entity_text.strip()
    text = re.sub(r"\s+", " ", text)

    normalized = normalize_text(text)

    for canonical_name, aliases in SOURCE_COMPANY_ALIASES.items():
        if normalized == canonical_name or normalized in aliases:
            return canonical_name

    cleaned = normalized
    for pattern in LEGAL_SUFFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.title() if cleaned else text


def is_self_mention(entity_text: str, source_company: str) -> bool:
    """
    Return True if the company mention refers to the source company itself.
    """
    normalized_entity = normalize_text(entity_text)
    canonical_source = canonicalize_source_company(source_company)

    source_aliases = SOURCE_COMPANY_ALIASES.get(canonical_source, {canonical_source})
    return normalized_entity == canonical_source or normalized_entity in source_aliases


def looks_like_valid_competitor_name(entity_text: str) -> bool:
    """
    Lightweight filter for company-like competitor names.
    """
    if not isinstance(entity_text, str):
        return False

    text = entity_text.strip()
    if len(text) < 2:
        return False

    generic_terms = {
        "company",
        "business",
        "market",
        "markets",
        "industry",
        "industries",
        "competitor",
        "competitors",
    }

    if normalize_text(text) in generic_terms:
        return False

    return True


def is_blacklisted_competitor_name(entity_text: str) -> bool:
    """
    Remove obvious non-competitor organizations and noisy names.
    """
    normalized = normalize_text(entity_text)

    if normalized in BLACKLIST_EXACT:
        return True

    if any(re.search(pattern, normalized) for pattern in BLACKLIST_PATTERNS):
        return True

    return False


def is_too_generic_competitor_name(entity_text: str) -> bool:
    """
    Remove vague descriptions that are not real competitor names.
    """
    normalized = normalize_text(entity_text)

    if len(normalized.split()) > 8:
        return True

    generic_hits = sum(
        1 for pattern in GENERIC_COMPETITOR_PATTERNS if re.search(pattern, normalized)
    )
    if generic_hits >= 2:
        return True

    return False


def has_competition_or_risk_signal(text: str) -> bool:
    """
    Detect if a chunk belongs to a competition/risk-related context.
    """
    text_lower = normalize_text(text)
    return any(re.search(pattern, text_lower) for pattern in COMPETITION_RISK_PATTERNS)


def build_mentions_from_entities(
    entities_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build competitor mention candidates from company entities joined with chunk context.
    """
    required_entity_cols = {
        "chunk_id",
        "entity_text",
        "entity_type",
        "source_doc_id",
        "year",
        "company",
    }
    required_chunk_cols = {
        "chunk_id",
        "chunk_text",
        "page_start",
        "page_end",
    }

    missing_entity_cols = required_entity_cols - set(entities_df.columns)
    if missing_entity_cols:
        raise ValueError(f"Missing columns in entities_df: {sorted(missing_entity_cols)}")

    missing_chunk_cols = required_chunk_cols - set(chunks_df.columns)
    if missing_chunk_cols:
        raise ValueError(f"Missing columns in chunks_df: {sorted(missing_chunk_cols)}")

    company_entities_df = entities_df[entities_df["entity_type"] == "company"].copy()

    # Prevent suffixing (_x, _y) by dropping overlapping columns from entities first
    overlap_cols = [c for c in ["page_start", "page_end"] if c in company_entities_df.columns]
    if overlap_cols:
        company_entities_df = company_entities_df.drop(columns=overlap_cols)

    merged_df = company_entities_df.merge(
        chunks_df[["chunk_id", "chunk_text", "page_start", "page_end"]],
        on="chunk_id",
        how="left",
    )

    merged_df["source_company"] = merged_df["company"].apply(canonicalize_source_company)
    merged_df["competitor_name"] = merged_df["entity_text"].apply(canonicalize_competitor_name)
    merged_df["is_self_mention"] = merged_df.apply(
        lambda row: is_self_mention(row["entity_text"], row["source_company"]),
        axis=1,
    )
    merged_df["is_valid_competitor_name"] = merged_df["entity_text"].apply(
        looks_like_valid_competitor_name
    )
    merged_df["is_blacklisted_competitor"] = merged_df["competitor_name"].apply(
        is_blacklisted_competitor_name
    )
    merged_df["is_too_generic_competitor"] = merged_df["competitor_name"].apply(
        is_too_generic_competitor_name
    )
    merged_df["has_competition_risk_signal"] = merged_df["chunk_text"].fillna("").apply(
        has_competition_or_risk_signal
    )

    mentions_df = merged_df[
        (~merged_df["is_self_mention"])
        & (merged_df["is_valid_competitor_name"])
        & (~merged_df["is_blacklisted_competitor"])
        & (~merged_df["is_too_generic_competitor"])
    ].copy()

    mentions_df["explicit_competes_with"] = False
    mentions_df["mention_source"] = "entity"

    return mentions_df[
        [
            "chunk_id",
            "source_doc_id",
            "source_company",
            "year",
            "competitor_name",
            "page_start",
            "page_end",
            "has_competition_risk_signal",
            "explicit_competes_with",
            "mention_source",
        ]
    ].reset_index(drop=True)


def build_mentions_from_triplets(triplets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build explicit competitor mention candidates from COMPETES_WITH triplets.
    """
    required_triplet_cols = {
        "chunk_id",
        "entity_a",
        "relation",
        "entity_b",
        "year",
        "company",
        "doc_id",
        "page_start",
        "page_end",
    }

    missing_cols = required_triplet_cols - set(triplets_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in triplets_df: {sorted(missing_cols)}")

    triplets_df = triplets_df.copy()
    triplets_df["source_company"] = triplets_df["company"].apply(canonicalize_source_company)

    compete_df = triplets_df[triplets_df["relation"] == "COMPETES_WITH"].copy()
    if compete_df.empty:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "source_doc_id",
                "source_company",
                "year",
                "competitor_name",
                "page_start",
                "page_end",
                "has_competition_risk_signal",
                "explicit_competes_with",
                "mention_source",
            ]
        )

    compete_df["competitor_name"] = compete_df["entity_b"].apply(canonicalize_competitor_name)
    compete_df["is_self_mention"] = compete_df.apply(
        lambda row: is_self_mention(row["competitor_name"], row["source_company"]),
        axis=1,
    )
    compete_df["is_valid_competitor_name"] = compete_df["competitor_name"].apply(
        looks_like_valid_competitor_name
    )
    compete_df["is_blacklisted_competitor"] = compete_df["competitor_name"].apply(
        is_blacklisted_competitor_name
    )
    compete_df["is_too_generic_competitor"] = compete_df["competitor_name"].apply(
        is_too_generic_competitor_name
    )

    compete_df = compete_df[
        (~compete_df["is_self_mention"])
        & (compete_df["is_valid_competitor_name"])
        & (~compete_df["is_blacklisted_competitor"])
        & (~compete_df["is_too_generic_competitor"])
    ].copy()

    compete_df["has_competition_risk_signal"] = True
    compete_df["explicit_competes_with"] = True
    compete_df["mention_source"] = "triplet"

    return compete_df[
        [
            "chunk_id",
            "doc_id",
            "source_company",
            "year",
            "competitor_name",
            "page_start",
            "page_end",
            "has_competition_risk_signal",
            "explicit_competes_with",
            "mention_source",
        ]
    ].rename(columns={"doc_id": "source_doc_id"}).reset_index(drop=True)


def build_competitor_mentions(
    chunks_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    triplets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the detailed competitor mentions table.
    """
    entity_mentions_df = build_mentions_from_entities(entities_df, chunks_df)
    triplet_mentions_df = build_mentions_from_triplets(triplets_df)

    mentions_df = pd.concat(
        [entity_mentions_df, triplet_mentions_df],
        ignore_index=True,
    )

    if mentions_df.empty:
        return mentions_df

    mentions_df = mentions_df.drop_duplicates(
        subset=["chunk_id", "source_company", "year", "competitor_name", "mention_source"],
        keep="first",
    ).reset_index(drop=True)

    return mentions_df


def build_competitor_summary(mentions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate competitor mentions by source company, year, and competitor.
    """
    if mentions_df.empty:
        return pd.DataFrame(
            columns=[
                "source_company",
                "year",
                "competitor_name",
                "mention_count",
                "competition_risk_mentions",
                "explicit_competes_with_count",
                "first_page_seen",
                "last_page_seen",
            ]
        )

    summary_df = (
        mentions_df.groupby(["source_company", "year", "competitor_name"], as_index=False)
        .agg(
            mention_count=("chunk_id", "count"),
            competition_risk_mentions=("has_competition_risk_signal", "sum"),
            explicit_competes_with_count=("explicit_competes_with", "sum"),
            first_page_seen=("page_start", "min"),
            last_page_seen=("page_end", "max"),
        )
        .sort_values(
            by=["source_company", "year", "mention_count", "explicit_competes_with_count"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )

    return summary_df


def build_clean_competitor_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only competitors supported by stronger evidence.
    Rules:
    - explicit COMPETES_WITH always kept
    - otherwise require competition/risk context + repeated mention
    """
    if summary_df.empty:
        return summary_df.copy()

    clean_df = summary_df[
        (summary_df["explicit_competes_with_count"] > 0)
        | (
            (summary_df["competition_risk_mentions"] >= 1)
            & (summary_df["mention_count"] >= 2)
        )
    ].copy()

    clean_df = clean_df.sort_values(
        by=["source_company", "year", "mention_count", "explicit_competes_with_count"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    return clean_df


def build_new_competitors_by_year(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify competitors that appear for the first time by source company and year.
    """
    if summary_df.empty:
        return pd.DataFrame(columns=["source_company", "year", "new_competitor_name"])

    rows = []

    for source_company, company_df in summary_df.groupby("source_company"):
        seen_before: set[str] = set()

        for year in sorted(company_df["year"].unique().tolist()):
            year_df = company_df[company_df["year"] == year]
            current_competitors = set(year_df["competitor_name"].tolist())
            new_competitors = sorted(current_competitors - seen_before)

            for competitor_name in new_competitors:
                rows.append(
                    {
                        "source_company": source_company,
                        "year": year,
                        "new_competitor_name": competitor_name,
                    }
                )

            seen_before |= current_competitors

    return pd.DataFrame(rows)


def write_observations(
    summary_df: pd.DataFrame,
    new_competitors_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write simple, interpretable observations to a text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("Competitive Analysis Observations")
    lines.append("=" * 40)
    lines.append("")

    if summary_df.empty:
        lines.append("No competitor mentions were detected in the current extraction outputs.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("Top competitor mentions by source company and year:")
    lines.append("")

    for (source_company, year), group_df in summary_df.groupby(["source_company", "year"]):
        top_row = group_df.iloc[0]
        lines.append(
            f"- {source_company} | {year}: top cited competitor = {top_row['competitor_name']} "
            f"(mentions={int(top_row['mention_count'])}, explicit_competes_with={int(top_row['explicit_competes_with_count'])})"
        )

    lines.append("")
    lines.append("New competitors appearing by year:")
    lines.append("")

    if new_competitors_df.empty:
        lines.append("- No new competitors identified across the available years.")
    else:
        for (source_company, year), group_df in new_competitors_df.groupby(["source_company", "year"]):
            competitors = ", ".join(sorted(group_df["new_competitor_name"].tolist()))
            lines.append(f"- {source_company} | {year}: {competitors}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_optional_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a simple chart of the number of unique competitors by source company and year.
    If matplotlib is not available, skip gracefully.
    """
    if summary_df.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[INFO] matplotlib is not available. Skipping chart generation.")
        return

    chart_df = (
        summary_df.groupby(["source_company", "year"], as_index=False)["competitor_name"]
        .nunique()
        .rename(columns={"competitor_name": "n_unique_competitors"})
    )

    pivot_df = (
        chart_df.pivot(index="year", columns="source_company", values="n_unique_competitors")
        .fillna(0)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Unique competitors mentioned by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of unique competitors")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple competitive analysis from entities and triplets."
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to chunks file.",
    )
    parser.add_argument(
        "--entities_path",
        type=str,
        default="data/processed/entities.parquet",
        help="Path to entities file.",
    )
    parser.add_argument(
        "--triplets_path",
        type=str,
        default="data/processed/triplets.parquet",
        help="Path to triplets file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/analysis",
        help="Directory for analysis outputs.",
    )

    args = parser.parse_args()

    chunks_df = load_dataframe(Path(args.chunks_path))
    entities_df = load_dataframe(Path(args.entities_path))
    triplets_df = load_dataframe(Path(args.triplets_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mentions_df = build_competitor_mentions(
        chunks_df=chunks_df,
        entities_df=entities_df,
        triplets_df=triplets_df,
    )

    summary_df = build_competitor_summary(mentions_df)
    new_competitors_df = build_new_competitors_by_year(summary_df)

    clean_summary_df = build_clean_competitor_summary(summary_df)
    clean_new_competitors_df = build_new_competitors_by_year(clean_summary_df)

    mentions_path = output_dir / "competitor_mentions.parquet"
    summary_path = output_dir / "competitor_summary.parquet"
    new_competitors_path = output_dir / "new_competitors_by_year.csv"
    observations_path = output_dir / "competitor_observations.txt"
    chart_path = output_dir / "unique_competitors_by_year.png"

    clean_summary_path = output_dir / "competitor_summary_clean.parquet"
    clean_new_competitors_path = output_dir / "new_competitors_by_year_clean.csv"
    clean_observations_path = output_dir / "competitor_observations_clean.txt"
    clean_chart_path = output_dir / "unique_competitors_by_year_clean.png"

    mentions_df.to_parquet(mentions_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    new_competitors_df.to_csv(new_competitors_path, index=False)

    write_observations(summary_df, new_competitors_df, observations_path)
    save_optional_chart(summary_df, chart_path)

    clean_summary_df.to_parquet(clean_summary_path, index=False)
    clean_new_competitors_df.to_csv(clean_new_competitors_path, index=False)

    write_observations(clean_summary_df, clean_new_competitors_df, clean_observations_path)
    save_optional_chart(clean_summary_df, clean_chart_path)

    print("\nCompetitive analysis completed.")
    print(f"Detailed mentions rows: {len(mentions_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Clean summary rows: {len(clean_summary_df)}")

    if not summary_df.empty:
        print("\nTop competitors by source company/year (raw):")
        top_rows = (
            summary_df.groupby(["source_company", "year"], as_index=False)
            .first()[["source_company", "year", "competitor_name", "mention_count", "explicit_competes_with_count"]]
        )
        print(top_rows.to_string(index=False))

    if not clean_summary_df.empty:
        print("\nTop competitors by source company/year (clean):")
        clean_top_rows = (
            clean_summary_df.groupby(["source_company", "year"], as_index=False)
            .first()[["source_company", "year", "competitor_name", "mention_count", "explicit_competes_with_count"]]
        )
        print(clean_top_rows.to_string(index=False))

    print(f"\nSaved detailed mentions to: {mentions_path.resolve()}")
    print(f"Saved summary to: {summary_path.resolve()}")
    print(f"Saved new competitors table to: {new_competitors_path.resolve()}")
    print(f"Saved observations to: {observations_path.resolve()}")

    if chart_path.exists():
        print(f"Saved chart to: {chart_path.resolve()}")

    print(f"Saved clean summary to: {clean_summary_path.resolve()}")
    print(f"Saved clean new competitors table to: {clean_new_competitors_path.resolve()}")
    print(f"Saved clean observations to: {clean_observations_path.resolve()}")

    if clean_chart_path.exists():
        print(f"Saved clean chart to: {clean_chart_path.resolve()}")


if __name__ == "__main__":
    main()