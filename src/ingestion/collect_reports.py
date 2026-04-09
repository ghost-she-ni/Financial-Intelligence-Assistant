from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# Expected format:
# adobe_2022_10k.pdf
# lockheedmartin_2023_10k.pdf
# pfizer_2024_10k.pdf
FILENAME_PATTERN = re.compile(
    r"^(?P<company>[a-z0-9]+)_(?P<fiscal_year>\d{4})_(?P<document_type>10k)\.pdf$"
)


def parse_filename(file_name: str) -> dict:
    """
    Parse a PDF file name using the project naming convention.
    """
    match = FILENAME_PATTERN.match(file_name)
    if not match:
        raise ValueError(
            f"Invalid file name: '{file_name}'. "
            "Expected format: {company}_{fiscal_year}_10k.pdf"
        )

    parsed = match.groupdict()
    parsed["fiscal_year"] = int(parsed["fiscal_year"])
    return parsed


def build_doc_id(company: str, fiscal_year: int, document_type: str) -> str:
    """
    Build a stable document identifier.
    """
    return f"{company}_{fiscal_year}_{document_type}"


def collect_pdf_metadata(
    raw_dir: Path, strict: bool = True, recursive: bool = True
) -> pd.DataFrame:
    """
    Scan the raw PDF directory and collect metadata for all reports.

    Args:
        raw_dir: Directory containing PDF files.
        strict: If True, raise an error on invalid filenames. If False,
                log a warning and skip the file.
        recursive: If True, search subdirectories recursively.

    Returns:
        DataFrame with metadata for each valid PDF.

    Raises:
        FileNotFoundError: If raw_dir does not exist.
        ValueError: If no PDF files are found, or if duplicate doc_ids exist,
                    or if strict=True and invalid filenames are encountered.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {raw_dir}")

    # Find PDF files
    if recursive:
        pdf_files = sorted(raw_dir.rglob("*.pdf"))
    else:
        pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {raw_dir}")

    records = []
    invalid_files = []

    for pdf_path in pdf_files:
        file_name = pdf_path.name

        try:
            parsed = parse_filename(file_name)
        except ValueError as e:
            if strict:
                invalid_files.append(str(pdf_path))
            else:
                logging.warning(f"Skipping invalid file: {file_name} - {e}")
            continue

        company = parsed["company"]
        fiscal_year = parsed["fiscal_year"]
        document_type = parsed["document_type"]

        records.append(
            {
                "doc_id": build_doc_id(company, fiscal_year, document_type),
                "company": company,
                "fiscal_year": fiscal_year,
                "document_type": document_type,
                "file_name": file_name,
                "file_path": str(pdf_path.resolve()),
                "source_folder": str(pdf_path.parent.resolve()),
                "file_size_bytes": pdf_path.stat().st_size,
            }
        )

    if invalid_files and strict:
        invalid_list = "\n".join(f"- {path}" for path in invalid_files)
        raise ValueError(
            "Some PDF files do not follow the required naming convention:\n"
            f"{invalid_list}"
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid PDF files were parsed.")

    df = df.sort_values(by=["company", "fiscal_year"]).reset_index(drop=True)

    duplicated_doc_ids = df[df["doc_id"].duplicated()]["doc_id"].tolist()
    if duplicated_doc_ids:
        raise ValueError(f"Duplicate doc_id values found: {duplicated_doc_ids}")

    return df


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save metadata to Parquet or CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end with .parquet or .csv")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a short summary for quick verification.
    """
    logging.info("Metadata collection completed.")
    logging.info(f"Documents found: {len(df)}")
    logging.info(f"Companies: {sorted(df['company'].unique().tolist())}")
    logging.info(f"Fiscal years: {sorted(df['fiscal_year'].unique().tolist())}")

    logging.info("Documents per company:")
    counts = df.groupby("company")["doc_id"].count().reset_index(name="n_documents")
    logging.info("\n" + counts.to_string(index=False))

    logging.info("Metadata preview:")
    preview_cols = [
        "doc_id",
        "company",
        "fiscal_year",
        "document_type",
        "file_name",
    ]
    logging.info("\n" + df[preview_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect metadata from raw PDF reports."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_pdfs",
        help="Directory containing raw PDF files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/metadata/documents_metadata.parquet",
        help="Output metadata file (.parquet or .csv).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on invalid filenames. (default: True)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip invalid filenames instead of failing.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories recursively. (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect metadata but do not save to file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    try:
        df = collect_pdf_metadata(
            raw_dir=input_dir,
            strict=args.strict,
            recursive=args.recursive,
        )
    except Exception as e:
        logging.error(f"Metadata collection failed: {e}")
        sys.exit(1)

    if args.dry_run:
        logging.info("Dry-run mode: metadata not saved.")
    else:
        try:
            save_metadata(df, output_path)
            logging.info(f"Saved to: {output_path.resolve()}")
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            sys.exit(1)

    print_summary(df)


if __name__ == "__main__":
    main()