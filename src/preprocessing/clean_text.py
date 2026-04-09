from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "doc_id",
    "company",
    "fiscal_year",
    "document_type",
    "file_name",
    "page_num",
    "raw_text",
}

# Exact lines that are always noise (navigation, separators)
EXACT_NOISE_LINES = {
    "",
    "Table of Contents",
    "Page",
    "PART I",
    "PART II",
    "PART III",
    "PART IV",
    "Item 1.",
    "Item 1A.",
    "Item 1B.",
    "Item 2.",
    "Item 3.",
    "Item 4.",
    "Item 5.",
    "Item 6.",
    "Item 7.",
    "Item 7A.",
    "Item 8.",
    "Item 9.",
    "Item 9A.",
    "Item 9B.",
    "Item 10.",
    "Item 11.",
    "Item 12.",
    "Item 13.",
    "Item 14.",
    "Item 15.",
}

# Regex patterns for obvious page noise
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                         # page number alone
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),  # "Page 1"
    re.compile(r"^\s*[Ff]\-\d+\s*$"),                  # footnote page "F-1"
    re.compile(r"^[\s\-_=]+$"),                         # line of dashes/underscores/equals
]

# Aggressive noise patterns (only used when --aggressive is set)
AGGRESSIVE_NOISE_PATTERNS = [
    re.compile(r"^\s*\d{1,2}\s*$"),      # very short numbers (could be headers, but often noise)
    re.compile(r"^\s*\.+\s*$"),          # ellipsis only
    re.compile(r"^\s*[\d\s,\.]+\s*$"),   # numbers and punctuation only (tables are removed)
]

# Character normalization mapping (extended)
CHAR_REPLACEMENTS = {
    "\u00a0": " ",   # non-breaking space
    "\u2007": " ",   # figure space
    "\u202f": " ",   # narrow no-break space
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM
    "\x00": "",      # null character
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2022": "-",   # bullet
    "\u25aa": "-",
    "\u2023": "-",
    "\u2026": "...",  # ellipsis
    "\u22ef": "...",
}


def load_extracted_pages(input_path: Path) -> pd.DataFrame:
    """Load extracted PDF pages from a Parquet or CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must be .parquet or .csv")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    return df


def normalize_characters(text: str) -> str:
    """Apply light character normalization."""
    if not isinstance(text, str):
        return ""
    for old, new in CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_line(line: str) -> str:
    """Clean a single line while preserving financial content."""
    line = line.strip()
    line = re.sub(r"[ \t]+", " ", line)   # collapse multiple spaces
    return line


def is_noise_line(line: str, aggressive: bool = False) -> bool:
    """Check whether a line is obvious noise."""
    if line in EXACT_NOISE_LINES:
        return True

    for pattern in NOISE_PATTERNS:
        if pattern.match(line):
            return True

    if aggressive:
        for pattern in AGGRESSIVE_NOISE_PATTERNS:
            if pattern.match(line):
                return True

    return False


def collapse_repeated_blank_lines(lines: list[str]) -> list[str]:
    """Remove repeated blank lines while keeping paragraph boundaries."""
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        is_blank = (line == "")
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return cleaned_lines


def clean_page_text(text: str, aggressive: bool = False) -> str:
    """
    Apply page-level cleaning.

    Args:
        text: Raw text from PDF.
        aggressive: If True, apply more aggressive noise removal.

    Returns:
        Cleaned text.
    """
    if text is None:
        return ""

    text = normalize_characters(text)

    raw_lines = text.split("\n")
    cleaned_lines = []

    for raw_line in raw_lines:
        line = normalize_line(raw_line)

        if is_noise_line(line, aggressive=aggressive):
            continue

        cleaned_lines.append(line)

    # Remove repeated blank lines
    cleaned_lines = collapse_repeated_blank_lines(cleaned_lines)

    cleaned_text = "\n".join(cleaned_lines).strip()

    # Additional aggressive cleanup: remove lines that are mostly punctuation/digits
    if aggressive:
        cleaned_lines_2 = []
        for line in cleaned_text.split("\n"):
            # Remove lines where alphanumeric content is less than 30% of the line
            # (useful for removing tables, but may remove actual content)
            if not line:
                continue
            alpha_num = sum(c.isalnum() for c in line)
            if alpha_num / max(1, len(line)) < 0.3:
                continue
            cleaned_lines_2.append(line)
        cleaned_text = "\n".join(cleaned_lines_2).strip()

    return cleaned_text


def process_pages(
    df: pd.DataFrame,
    progress: bool = True,
    limit: int | None = None,
    aggressive: bool = False,
) -> pd.DataFrame:
    """
    Create a processed version of extracted pages.

    Args:
        df: Input DataFrame with raw extracted pages.
        progress: Show progress bar if True and tqdm installed.
        limit: Process only the first N rows (for testing).
        aggressive: Apply aggressive cleaning (removes more noise).

    Returns:
        Processed DataFrame with cleaned text and statistics.
    """
    if limit is not None:
        df = df.head(limit)
        logging.info(f"Limiting to first {limit} rows.")

    processed_df = df.copy()

    # Attempt to import tqdm for progress bar
    iterator = processed_df.iterrows()
    if progress:
        try:
            from tqdm import tqdm
            total = len(processed_df)
            iterator = tqdm(iterator, total=total, desc="Cleaning pages")
        except ImportError:
            logging.warning("tqdm not installed, progress bar disabled.")

    clean_texts = []
    raw_char_counts = []
    clean_char_counts = []
    raw_word_counts = []
    clean_word_counts = []
    is_nearly_empty = []

    for idx, row in iterator:
        raw_text = row["raw_text"] if pd.notna(row["raw_text"]) else ""
        clean_text = clean_page_text(raw_text, aggressive=aggressive)

        clean_texts.append(clean_text)
        raw_char_counts.append(len(raw_text))
        clean_char_counts.append(len(clean_text))
        raw_word_counts.append(len(raw_text.split()))
        clean_word_counts.append(len(clean_text.split()))
        is_nearly_empty.append(len(clean_text.split()) < 20)

    processed_df["clean_text"] = clean_texts
    processed_df["raw_char_count"] = raw_char_counts
    processed_df["clean_char_count"] = clean_char_counts
    processed_df["raw_word_count"] = raw_word_counts
    processed_df["clean_word_count"] = clean_word_counts
    processed_df["is_nearly_empty"] = is_nearly_empty

    processed_df = processed_df.sort_values(
        by=["company", "fiscal_year", "doc_id", "page_num"]
    ).reset_index(drop=True)

    return processed_df


def save_processed_pages(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed pages to Parquet or CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end with .parquet or .csv")


def print_summary(df: pd.DataFrame) -> None:
    """Print a short summary using logging."""
    logging.info("Text cleaning completed.")
    logging.info(f"Total processed pages: {len(df)}")
    logging.info(f"Documents covered: {df['doc_id'].nunique()}")
    logging.info(f"Nearly empty pages: {int(df['is_nearly_empty'].sum())}")

    preview_cols = [
        "doc_id",
        "page_num",
        "raw_word_count",
        "clean_word_count",
        "is_nearly_empty",
    ]

    logging.info("Preview:")
    logging.info("\n" + df[preview_cols].head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean extracted PDF text page by page."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/extracted/extracted_pages.parquet",
        help="Input extracted pages file (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/processed_pages.parquet",
        help="Output processed pages file (.parquet or .csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for testing).",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Apply aggressive cleaning (removes more noise, may lose some content).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and preview, but do not save output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (even if tqdm is installed).",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Load data
    try:
        extracted_df = load_extracted_pages(input_path)
    except Exception as e:
        logging.error(f"Failed to load input: {e}")
        sys.exit(1)

    if args.dry_run:
        logging.info("Dry-run mode: no output will be saved.")
        logging.info(f"Loaded {len(extracted_df)} pages.")
        logging.info(f"Columns: {list(extracted_df.columns)}")
        # Show a small preview of raw text
        logging.info("Sample of raw_text (first 3 pages):")
        for idx, row in extracted_df.head(3).iterrows():
            preview = row["raw_text"][:200] + "..." if len(row["raw_text"]) > 200 else row["raw_text"]
            logging.info(f"Doc {row['doc_id']}, page {row['page_num']}: {preview}")
        sys.exit(0)

    # Process pages
    try:
        processed_df = process_pages(
            extracted_df,
            progress=not args.no_progress,
            limit=args.limit,
            aggressive=args.aggressive,
        )
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

    # Save
    try:
        save_processed_pages(processed_df, output_path)
        logging.info(f"Saved to: {output_path.resolve()}")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
        sys.exit(1)

    print_summary(processed_df)


if __name__ == "__main__":
    main()
