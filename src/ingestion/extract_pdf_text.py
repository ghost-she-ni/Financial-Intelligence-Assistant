from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

REQUIRED_METADATA_COLUMNS = {
    "doc_id",
    "company",
    "fiscal_year",
    "document_type",
    "file_name",
    "file_path",
}


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load document metadata from a Parquet or CSV file.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    if metadata_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(metadata_path)
    elif metadata_path.suffix.lower() == ".csv":
        df = pd.read_csv(metadata_path)
    else:
        raise ValueError("Metadata file must be .parquet or .csv")

    missing_cols = REQUIRED_METADATA_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_cols)}")

    return df


def extract_pages_from_pdf(
    pdf_path: Path,
    doc_id: str,
    company: str,
    fiscal_year: int,
    document_type: str,
    file_name: str,
    max_pages: int | None = None,
) -> list[dict]:
    """
    Extract raw text page by page from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: Document identifier.
        company: Company name.
        fiscal_year: Fiscal year.
        document_type: Type of document (e.g., "10k").
        file_name: Original file name.
        max_pages: Maximum number of pages to extract (None = all pages).

    Returns:
        List of dictionaries, each representing a page.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    records = []

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_index in range(pages_to_process):
            page = pdf[page_index]
            raw_text = page.get_text("text")
            raw_text = raw_text if raw_text is not None else ""

            records.append(
                {
                    "doc_id": doc_id,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "document_type": document_type,
                    "file_name": file_name,
                    "page_num": page_index + 1,  # 1-indexed
                    "raw_text": raw_text,
                    "char_count": len(raw_text),
                    "word_count": len(raw_text.split()),
                }
            )

    return records


def extract_all_pages(
    metadata_df: pd.DataFrame,
    skip_errors: bool = False,
    max_pages: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Extract raw text for all documents listed in the metadata file.

    Args:
        metadata_df: DataFrame containing document metadata.
        skip_errors: If True, skip PDFs that cannot be processed and continue.
        max_pages: Maximum pages to extract per PDF (None = all).
        progress: If True, show a progress bar (requires tqdm).

    Returns:
        DataFrame with one row per extracted page.
    """
    all_records = []
    failed_docs = []

    # Attempt to import tqdm for progress bar
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(metadata_df.itertuples(index=False), total=len(metadata_df), desc="Extracting PDFs")
        except ImportError:
            logging.warning("tqdm not installed, progress bar disabled.")
            iterator = metadata_df.itertuples(index=False)
    else:
        iterator = metadata_df.itertuples(index=False)

    for row in iterator:
        pdf_path = Path(row.file_path)
        try:
            page_records = extract_pages_from_pdf(
                pdf_path=pdf_path,
                doc_id=row.doc_id,
                company=row.company,
                fiscal_year=row.fiscal_year,
                document_type=row.document_type,
                file_name=row.file_name,
                max_pages=max_pages,
            )
            all_records.extend(page_records)
            if not progress:
                logging.info(f"Extracted {len(page_records)} pages from {row.doc_id}")
        except Exception as e:
            if skip_errors:
                logging.error(f"Failed to extract {row.doc_id}: {e}")
                failed_docs.append((row.doc_id, str(e)))
            else:
                raise

    if failed_docs and skip_errors:
        logging.warning(f"Skipped {len(failed_docs)} documents due to errors.")
        for doc_id, err in failed_docs:
            logging.warning(f"  {doc_id}: {err}")

    extracted_df = pd.DataFrame(all_records)

    if extracted_df.empty:
        raise ValueError("No pages were extracted.")

    extracted_df = extracted_df.sort_values(
        by=["company", "fiscal_year", "doc_id", "page_num"]
    ).reset_index(drop=True)

    return extracted_df


def save_extracted_pages(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save extracted pages to Parquet or CSV.
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
    Print a short summary using logging.
    """
    logging.info("PDF text extraction completed.")
    logging.info(f"Total extracted pages: {len(df)}")
    logging.info(f"Documents covered: {df['doc_id'].nunique()}")
    logging.info(f"Companies: {sorted(df['company'].unique().tolist())}")
    logging.info(f"Fiscal years: {sorted(df['fiscal_year'].unique().tolist())}")

    logging.info("Pages per document:")
    page_counts = df.groupby("doc_id")["page_num"].count().reset_index(name="n_pages")
    logging.info("\n" + page_counts.to_string(index=False))

    logging.info("Preview:")
    preview_cols = [
        "doc_id",
        "page_num",
        "char_count",
        "word_count",
    ]
    logging.info("\n" + df[preview_cols].head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract raw text page by page from PDF reports."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/metadata/documents_metadata.parquet",
        help="Path to the document metadata file (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/extracted/extracted_pages.parquet",
        help="Output file for extracted pages (.parquet or .csv).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip PDFs that cannot be extracted and continue.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to extract per PDF (for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check metadata and PDF existence, do not extract.",
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

    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)

    # Load metadata
    try:
        metadata_df = load_metadata(metadata_path)
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        sys.exit(1)

    if args.dry_run:
        logging.info("Dry-run mode: checking PDF files.")
        for row in metadata_df.itertuples(index=False):
            pdf_path = Path(row.file_path)
            if pdf_path.exists():
                logging.info(f"Found: {row.doc_id} -> {pdf_path}")
            else:
                logging.error(f"Missing: {row.doc_id} -> {pdf_path}")
        sys.exit(0)

    # Extract pages
    try:
        extracted_df = extract_all_pages(
            metadata_df=metadata_df,
            skip_errors=args.skip_errors,
            max_pages=args.max_pages,
            progress=not args.no_progress,
        )
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        sys.exit(1)

    # Save
    try:
        save_extracted_pages(extracted_df, output_path)
        logging.info(f"Saved to: {output_path.resolve()}")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
        sys.exit(1)

    print_summary(extracted_df)


if __name__ == "__main__":
    main()
