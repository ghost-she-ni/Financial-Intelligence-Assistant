from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.preprocessing.sections import annotate_document_sections

# Optional sentence tokenizer
try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
        from nltk.tokenize import sent_tokenize

        SENTENCE_TOKENIZER = sent_tokenize
    except LookupError:
        SENTENCE_TOKENIZER = None
        logging.warning(
            "nltk punkt tokenizer not available locally; using simple sentence splitting fallback."
        )
except ImportError:
    SENTENCE_TOKENIZER = None
    logging.warning("nltk not installed; using simple sentence splitting fallback.")


REQUIRED_COLUMNS = {
    "doc_id",
    "company",
    "fiscal_year",
    "document_type",
    "file_name",
    "page_num",
    "clean_text",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
NO_SPACE_BEFORE_TOKENS = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}"}
NO_SPACE_AFTER_TOKENS = {"(", "[", "{", "$"}


def load_processed_pages(input_path: Path) -> pd.DataFrame:
    """Load processed pages from a Parquet or CSV file."""
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


def build_chunk_id(
    doc_id: str,
    page_start: int,
    page_end: int,
    chunk_index: int,
) -> str:
    """Build a stable chunk identifier."""
    return f"{doc_id}_p{page_start:04d}_p{page_end:04d}_c{chunk_index:04d}"


def tokenize_for_chunking(text: str) -> list[str]:
    """Split text into token units for fixed-size token chunking."""
    if not isinstance(text, str) or not text.strip():
        return []
    return TOKEN_PATTERN.findall(text)


def count_text_tokens(text: str) -> int:
    """Count chunking tokens in a text span."""
    return len(tokenize_for_chunking(text))


def untokenize_tokens(tokens: list[str]) -> str:
    """Rebuild readable text from token units with lightweight spacing rules."""
    if not tokens:
        return ""

    parts = [tokens[0]]
    previous_token = tokens[0]
    for token in tokens[1:]:
        if token in NO_SPACE_BEFORE_TOKENS or previous_token in NO_SPACE_AFTER_TOKENS:
            parts.append(token)
        else:
            parts.append(f" {token}")
        previous_token = token

    return "".join(parts).strip()


def build_word_records(doc_df: pd.DataFrame) -> list[dict]:
    """Convert document pages into a flat list of word records."""
    word_records: list[dict] = []
    for row in doc_df.sort_values("page_num").itertuples(index=False):
        text = row.clean_text if isinstance(row.clean_text, str) else ""
        words = text.split()
        for word in words:
            word_records.append(
                {
                    "word": word,
                    "page_num": int(row.page_num),
                    "section_id": getattr(row, "section_id", "front_matter__01"),
                    "section_code": getattr(row, "section_code", "front_matter"),
                    "section_title": getattr(row, "section_title", "Front Matter"),
                    "section_group": getattr(row, "section_group", "front_matter"),
                }
            )
    return word_records


def build_token_records(doc_df: pd.DataFrame) -> list[dict]:
    """Convert document pages into a flat list of token records."""
    token_records: list[dict] = []
    for row in doc_df.sort_values("page_num").itertuples(index=False):
        text = row.clean_text if isinstance(row.clean_text, str) else ""
        tokens = tokenize_for_chunking(text)
        for token in tokens:
            token_records.append(
                {
                    "token": token,
                    "page_num": int(row.page_num),
                    "section_id": getattr(row, "section_id", "front_matter__01"),
                    "section_code": getattr(row, "section_code", "front_matter"),
                    "section_title": getattr(row, "section_title", "Front Matter"),
                    "section_group": getattr(row, "section_group", "front_matter"),
                }
            )
    return token_records


def compute_chunk_starts(
    total_units: int,
    chunk_size: int,
    overlap: int,
    min_chunk_units: int = 100,
) -> list[int]:
    """Compute sliding-window start offsets for fixed-size chunk units."""
    if total_units <= 0:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly smaller than chunk_size")
    if total_units <= chunk_size:
        return [0]

    step = chunk_size - overlap
    starts = [0]

    while True:
        next_start = starts[-1] + step
        if next_start >= total_units:
            break
        remaining_units = total_units - next_start
        if remaining_units < min_chunk_units:
            tail_start = max(0, total_units - chunk_size)
            if tail_start > starts[-1]:
                starts.append(tail_start)
            break
        starts.append(next_start)
    return starts


def build_sentence_records(doc_df: pd.DataFrame) -> list[dict]:
    """Convert document pages into a flat list of sentence records."""
    sentence_records: list[dict] = []
    for row in doc_df.sort_values("page_num").itertuples(index=False):
        text = row.clean_text if isinstance(row.clean_text, str) else ""
        if not text.strip():
            continue

        if SENTENCE_TOKENIZER:
            sentences = SENTENCE_TOKENIZER(text)
        else:
            sentences = [sentence.strip() for sentence in text.split(". ") if sentence.strip()]
            if text and not sentences:
                sentences = [text]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            token_count = count_text_tokens(sentence)
            if token_count == 0:
                continue

            sentence_records.append(
                {
                    "sentence": sentence,
                    "page_num": int(row.page_num),
                    "word_count": len(sentence.split()),
                    "token_count": token_count,
                    "section_id": getattr(row, "section_id", "front_matter__01"),
                    "section_code": getattr(row, "section_code", "front_matter"),
                    "section_title": getattr(row, "section_title", "Front Matter"),
                    "section_group": getattr(row, "section_group", "front_matter"),
                }
            )
    return sentence_records


def get_document_metadata(doc_df: pd.DataFrame) -> dict[str, object]:
    """Extract stable document metadata used in chunk records."""
    first_row = doc_df.iloc[0]
    return {
        "doc_id": first_row["doc_id"],
        "company": first_row["company"],
        "fiscal_year": int(first_row["fiscal_year"]),
        "document_type": first_row["document_type"],
        "file_name": first_row["file_name"],
    }


def build_chunks_for_document_sentence(
    doc_df: pd.DataFrame,
    chunk_size: int,
    overlap_sentences: int,
    min_chunk_words: int = 100,
    tolerance: float = 1.2,
) -> list[dict]:
    """
    Build chunks using sentence boundaries.

    This legacy semantic mode now interprets `chunk_size` in tokens so its scale
    stays comparable with the fixed-size token baseline.
    """
    if doc_df.empty:
        return []

    doc_df = annotate_document_sections(doc_df).sort_values("page_num").reset_index(drop=True)
    metadata = get_document_metadata(doc_df)

    sentence_records = build_sentence_records(doc_df)
    if not sentence_records:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    start_index = 0
    total_sentences = len(sentence_records)

    while start_index < total_sentences:
        section_id = sentence_records[start_index]["section_id"]
        current_token_count = 0
        end_index = start_index

        while end_index < total_sentences:
            if sentence_records[end_index]["section_id"] != section_id:
                break

            next_token_count = current_token_count + sentence_records[end_index]["token_count"]
            if current_token_count > 0 and next_token_count > chunk_size * tolerance:
                break
            current_token_count = next_token_count
            end_index += 1

        if end_index == start_index:
            end_index = start_index + 1
            current_token_count = sentence_records[start_index]["token_count"]

        chunk_sentences = sentence_records[start_index:end_index]
        chunk_text = " ".join(record["sentence"] for record in chunk_sentences).strip()
        page_start = min(record["page_num"] for record in chunk_sentences)
        page_end = max(record["page_num"] for record in chunk_sentences)

        chunks.append(
            {
                "chunk_id": build_chunk_id(metadata["doc_id"], page_start, page_end, chunk_index),
                "chunk_index": chunk_index,
                "doc_id": metadata["doc_id"],
                "company": metadata["company"],
                "fiscal_year": metadata["fiscal_year"],
                "document_type": metadata["document_type"],
                "file_name": metadata["file_name"],
                "page_start": page_start,
                "page_end": page_end,
                "chunk_text": chunk_text,
                "word_count": len(chunk_text.split()),
                "token_count": current_token_count,
                "char_count": len(chunk_text),
                "section_id": section_id,
                "section_code": chunk_sentences[0]["section_code"],
                "section_title": chunk_sentences[0]["section_title"],
                "section_group": chunk_sentences[0]["section_group"],
            }
        )
        chunk_index += 1

        if end_index >= total_sentences:
            break

        if sentence_records[end_index]["section_id"] != section_id:
            start_index = end_index
        else:
            start_index = max(start_index + 1, end_index - overlap_sentences)

    if (
        len(chunks) >= 2
        and chunks[-1]["token_count"] < min_chunk_words
        and chunks[-1]["section_id"] == chunks[-2]["section_id"]
    ):
        last_chunk = chunks.pop()
        previous_chunk = chunks[-1]
        previous_chunk["chunk_text"] = f"{previous_chunk['chunk_text']} {last_chunk['chunk_text']}".strip()
        previous_chunk["word_count"] = len(previous_chunk["chunk_text"].split())
        previous_chunk["token_count"] = previous_chunk["token_count"] + last_chunk["token_count"]
        previous_chunk["char_count"] = len(previous_chunk["chunk_text"])
        previous_chunk["page_end"] = last_chunk["page_end"]

    return chunks


def build_chunks_for_document_word(
    doc_df: pd.DataFrame,
    chunk_size: int,
    overlap: int,
    min_chunk_words: int = 100,
) -> list[dict]:
    """Build chunks using legacy whitespace-word sliding windows."""
    if doc_df.empty:
        return []

    doc_df = annotate_document_sections(doc_df).sort_values("page_num").reset_index(drop=True)
    metadata = get_document_metadata(doc_df)

    word_records = build_word_records(doc_df)
    total_words = len(word_records)
    if total_words == 0:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    start_offset = 0

    while start_offset < total_words:
        section_id = word_records[start_offset]["section_id"]
        end_offset = start_offset
        while end_offset < total_words and word_records[end_offset]["section_id"] == section_id:
            end_offset += 1

        section_word_records = word_records[start_offset:end_offset]
        starts = compute_chunk_starts(
            total_units=len(section_word_records),
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_units=min_chunk_words,
        )

        for local_start in starts:
            local_end = min(local_start + chunk_size, len(section_word_records))
            chunk_word_records = section_word_records[local_start:local_end]
            chunk_words = [record["word"] for record in chunk_word_records]
            page_nums = [record["page_num"] for record in chunk_word_records]
            page_start = min(page_nums)
            page_end = max(page_nums)
            chunk_text = " ".join(chunk_words).strip()

            chunks.append(
                {
                    "chunk_id": build_chunk_id(metadata["doc_id"], page_start, page_end, chunk_index),
                    "chunk_index": chunk_index,
                    "doc_id": metadata["doc_id"],
                    "company": metadata["company"],
                    "fiscal_year": metadata["fiscal_year"],
                    "document_type": metadata["document_type"],
                    "file_name": metadata["file_name"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_text": chunk_text,
                    "word_count": len(chunk_words),
                    "token_count": count_text_tokens(chunk_text),
                    "char_count": len(chunk_text),
                    "section_id": section_id,
                    "section_code": chunk_word_records[0]["section_code"],
                    "section_title": chunk_word_records[0]["section_title"],
                    "section_group": chunk_word_records[0]["section_group"],
                }
            )
            chunk_index += 1

        start_offset = end_offset

    return chunks


def build_chunks_for_document_token(
    doc_df: pd.DataFrame,
    chunk_size: int,
    overlap: int,
    min_chunk_words: int = 100,
) -> list[dict]:
    """Build fixed-size chunks using token units."""
    if doc_df.empty:
        return []

    doc_df = annotate_document_sections(doc_df).sort_values("page_num").reset_index(drop=True)
    metadata = get_document_metadata(doc_df)

    token_records = build_token_records(doc_df)
    total_tokens = len(token_records)
    if total_tokens == 0:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    start_offset = 0

    while start_offset < total_tokens:
        section_id = token_records[start_offset]["section_id"]
        end_offset = start_offset
        while end_offset < total_tokens and token_records[end_offset]["section_id"] == section_id:
            end_offset += 1

        section_token_records = token_records[start_offset:end_offset]
        starts = compute_chunk_starts(
            total_units=len(section_token_records),
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_units=min_chunk_words,
        )

        for local_start in starts:
            local_end = min(local_start + chunk_size, len(section_token_records))
            chunk_token_records = section_token_records[local_start:local_end]
            chunk_tokens = [record["token"] for record in chunk_token_records]
            page_nums = [record["page_num"] for record in chunk_token_records]
            page_start = min(page_nums)
            page_end = max(page_nums)
            chunk_text = untokenize_tokens(chunk_tokens)

            chunks.append(
                {
                    "chunk_id": build_chunk_id(metadata["doc_id"], page_start, page_end, chunk_index),
                    "chunk_index": chunk_index,
                    "doc_id": metadata["doc_id"],
                    "company": metadata["company"],
                    "fiscal_year": metadata["fiscal_year"],
                    "document_type": metadata["document_type"],
                    "file_name": metadata["file_name"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "token_count": len(chunk_token_records),
                    "char_count": len(chunk_text),
                    "section_id": section_id,
                    "section_code": chunk_token_records[0]["section_code"],
                    "section_title": chunk_token_records[0]["section_title"],
                    "section_group": chunk_token_records[0]["section_group"],
                }
            )
            chunk_index += 1

        start_offset = end_offset

    return chunks


def chunk_all_documents(
    processed_pages_df: pd.DataFrame,
    chunk_size: int,
    overlap: int,
    min_chunk_words: int = 100,
    method: str = "token",
    overlap_sentences: int = 1,
    progress: bool = True,
    limit_docs: int | None = None,
) -> pd.DataFrame:
    """Build chunks for all documents."""
    all_doc_ids = sorted(processed_pages_df["doc_id"].unique())
    if limit_docs is not None:
        doc_ids_to_process = all_doc_ids[:limit_docs]
        processed_pages_df = processed_pages_df[
            processed_pages_df["doc_id"].isin(doc_ids_to_process)
        ].copy()
        logging.info("Limiting to first %s documents.", limit_docs)

    grouped = processed_pages_df.groupby("doc_id", sort=True)
    iterator = grouped
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(grouped, total=grouped.ngroups, desc="Chunking documents")
        except ImportError:
            logging.warning("tqdm not installed, progress bar disabled.")

    all_chunks: list[dict] = []
    for _, doc_df in iterator:
        if method == "token":
            doc_chunks = build_chunks_for_document_token(
                doc_df=doc_df,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_words=min_chunk_words,
            )
        elif method == "sentence":
            doc_chunks = build_chunks_for_document_sentence(
                doc_df=doc_df,
                chunk_size=chunk_size,
                overlap_sentences=overlap_sentences,
                min_chunk_words=min_chunk_words,
            )
        elif method == "word":
            doc_chunks = build_chunks_for_document_word(
                doc_df=doc_df,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_words=min_chunk_words,
            )
        else:
            raise ValueError("method must be 'token', 'sentence', or 'word'")
        all_chunks.extend(doc_chunks)

    chunks_df = pd.DataFrame(all_chunks)
    if chunks_df.empty:
        raise ValueError("No chunks were created.")

    duplicated_ids = chunks_df[chunks_df["chunk_id"].duplicated()]["chunk_id"].tolist()
    if duplicated_ids:
        raise ValueError(f"Duplicate chunk_id values found: {duplicated_ids}")

    return chunks_df.sort_values(
        by=["company", "fiscal_year", "doc_id", "page_start", "chunk_index"]
    ).reset_index(drop=True)


def save_chunks(df: pd.DataFrame, output_path: Path) -> None:
    """Save chunks to Parquet or CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end with .parquet or .csv")


def print_summary(df: pd.DataFrame) -> None:
    """Print a short chunking summary using logging."""
    logging.info("Chunking completed.")
    logging.info("Total chunks: %s", len(df))
    logging.info("Documents covered: %s", df["doc_id"].nunique())
    logging.info("Average tokens per chunk: %.2f", df["token_count"].mean())
    logging.info("Min tokens per chunk: %s", df["token_count"].min())
    logging.info("Max tokens per chunk: %s", df["token_count"].max())
    logging.info("Average words per chunk: %.2f", df["word_count"].mean())
    logging.info("Min words per chunk: %s", df["word_count"].min())
    logging.info("Max words per chunk: %s", df["word_count"].max())
    logging.info("Chunks per document:")
    counts = df.groupby("doc_id")["chunk_id"].count().reset_index(name="n_chunks")
    logging.info("\n%s", counts.to_string(index=False))
    logging.info("Preview:")
    preview_cols = ["chunk_id", "doc_id", "page_start", "page_end", "token_count", "word_count"]
    logging.info("\n%s", df[preview_cols].head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chunks from processed pages.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/processed_pages.parquet",
        help="Input processed pages file (.parquet or .csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Output chunks file (.parquet or .csv).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Chunk size. For token/sentence methods this is counted in tokens; for word it is counted in words.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=75,
        help="For token/word methods: overlap in tokens/words. For sentence method: ignored; use --overlap_sentences.",
    )
    parser.add_argument(
        "--min_chunk_words",
        type=int,
        default=100,
        help="Minimum trailing chunk size before tail adjustment.",
    )
    parser.add_argument(
        "--method",
        choices=["token", "sentence", "word"],
        default="token",
        help="Chunking method: 'token' (fixed-size baseline, default), 'sentence' (semantic), or 'word' (legacy sliding window).",
    )
    parser.add_argument(
        "--overlap_sentences",
        type=int,
        default=1,
        help="For sentence method: number of sentences to overlap between chunks.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only the first N documents (by sorted doc_id).",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Process only the first N pages per document (for testing).",
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

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.chunk_size <= 0:
        logging.error("chunk_size must be positive.")
        sys.exit(1)
    if args.overlap < 0:
        logging.error("overlap must be >= 0.")
        sys.exit(1)
    if args.min_chunk_words < 0:
        logging.error("min_chunk_words must be >= 0.")
        sys.exit(1)
    if args.method in {"token", "word"} and args.overlap >= args.chunk_size:
        logging.error("For token/word methods, overlap must be strictly smaller than chunk_size.")
        sys.exit(1)
    if args.method == "sentence" and args.overlap_sentences < 0:
        logging.error("overlap_sentences must be >= 0.")
        sys.exit(1)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    try:
        processed_pages_df = load_processed_pages(input_path)
    except Exception as exc:
        logging.error("Failed to load input: %s", exc)
        sys.exit(1)

    if args.limit_pages is not None:
        logging.info("Limiting to first %s pages per document.", args.limit_pages)
        processed_pages_df = (
            processed_pages_df.sort_values(["doc_id", "page_num"])
            .groupby("doc_id")
            .head(args.limit_pages)
            .reset_index(drop=True)
        )

    if args.dry_run:
        logging.info("Dry-run mode: no output will be saved.")
        logging.info(
            "Loaded %s pages from %s documents.",
            len(processed_pages_df),
            processed_pages_df["doc_id"].nunique(),
        )
        for row in processed_pages_df.head(3).itertuples(index=False):
            preview = row.clean_text[:200] + "..." if len(row.clean_text) > 200 else row.clean_text
            logging.info("Doc %s, page %s: %s", row.doc_id, row.page_num, preview)
        sys.exit(0)

    try:
        chunks_df = chunk_all_documents(
            processed_pages_df=processed_pages_df,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_words=args.min_chunk_words,
            method=args.method,
            overlap_sentences=args.overlap_sentences,
            progress=not args.no_progress,
            limit_docs=args.limit_docs,
        )
    except Exception as exc:
        logging.error("Chunking failed: %s", exc)
        sys.exit(1)

    try:
        save_chunks(chunks_df, output_path)
        logging.info("Saved to: %s", output_path.resolve())
    except Exception as exc:
        logging.error("Failed to save output: %s", exc)
        sys.exit(1)

    print_summary(chunks_df)


if __name__ == "__main__":
    main()
