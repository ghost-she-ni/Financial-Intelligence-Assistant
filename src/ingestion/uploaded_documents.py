from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.embeddings.cache import build_chunk_embedding_record, compute_text_hash
from src.preprocessing.chunking import chunk_all_documents
from src.preprocessing.clean_text import process_pages

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}
DEFAULT_UPLOAD_CHUNK_SIZE = 500
DEFAULT_UPLOAD_CHUNK_OVERLAP = 75
DEFAULT_UPLOAD_MIN_CHUNK_WORDS = 40
UPLOAD_PIPELINE_VERSION = "v1"


@dataclass(frozen=True)
class UploadedFilePayload:
    file_name: str
    content: bytes


@dataclass
class UploadedDocumentBundle:
    corpus_id: str
    chunks_df: pd.DataFrame
    chunk_embeddings_df: pd.DataFrame
    documents: list[dict[str, Any]]


@dataclass(frozen=True)
class UploadedRuntimeCorpus:
    corpus_id: str
    chunks_path: Path
    chunk_embeddings_path: Path
    documents: list[dict[str, Any]]
    uploaded_chunk_count: int


def sanitize_name_component(value: str) -> str:
    """Convert a free-form file stem into a stable ASCII identifier."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "uploaded_document"


def decode_text_document(content: bytes) -> str:
    """Decode text-like uploads with tolerant fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def extract_year_hint(file_name: str) -> int:
    """Extract a likely year from the file name, otherwise return 0."""
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", file_name)
    return int(match.group(1)) if match else 0


def build_uploaded_corpus_id(
    uploaded_files: Sequence[UploadedFilePayload],
    embedding_model: str,
    base_chunks_path: Path | None = None,
    base_chunk_embeddings_path: Path | None = None,
) -> str:
    """Build a stable cache key for a session upload bundle."""
    base_signatures: dict[str, Any] = {}
    for label, path in {
        "base_chunks": base_chunks_path,
        "base_chunk_embeddings": base_chunk_embeddings_path,
    }.items():
        if path is None:
            continue
        resolved_path = path.resolve()
        stat = resolved_path.stat()
        base_signatures[label] = {
            "path": str(resolved_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }

    payload = {
        "pipeline_version": UPLOAD_PIPELINE_VERSION,
        "embedding_model": embedding_model,
        "base_signatures": base_signatures,
        "files": [
            {
                "file_name": file.file_name,
                "size": len(file.content),
                "content_hash": hashlib.sha256(file.content).hexdigest(),
            }
            for file in uploaded_files
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from parquet or CSV."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Persist a small JSON manifest next to the runtime corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_uploaded_metadata(
    uploaded_file: UploadedFilePayload,
    position: int,
) -> dict[str, Any]:
    """Create stable metadata for one uploaded document."""
    file_path = Path(uploaded_file.file_name)
    file_stem = file_path.stem or f"document_{position}"
    file_extension = file_path.suffix.lower()

    if file_extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for '{uploaded_file.file_name}'. "
            "Supported types: pdf, txt, md."
        )

    slug = sanitize_name_component(file_stem)
    content_hash = hashlib.sha256(uploaded_file.content).hexdigest()[:10]

    return {
        "doc_id": f"uploaded_{slug}_{content_hash}",
        "company": slug,
        "fiscal_year": extract_year_hint(uploaded_file.file_name),
        "document_type": file_extension.lstrip("."),
        "file_name": uploaded_file.file_name,
    }


def extract_pdf_pages(
    uploaded_file: UploadedFilePayload,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract text page by page from an uploaded PDF."""
    records: list[dict[str, Any]] = []

    with fitz.open(stream=uploaded_file.content, filetype="pdf") as pdf:
        for page_index, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text") or ""
            records.append(
                {
                    **metadata,
                    "page_num": page_index,
                    "raw_text": raw_text,
                    "char_count": len(raw_text),
                    "word_count": len(raw_text.split()),
                }
            )

    return records


def extract_text_pages(
    uploaded_file: UploadedFilePayload,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Wrap a text or markdown document into a single-page record."""
    raw_text = decode_text_document(uploaded_file.content)
    return [
        {
            **metadata,
            "page_num": 1,
            "raw_text": raw_text,
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()),
        }
    ]


def extract_uploaded_pages(
    uploaded_files: Sequence[UploadedFilePayload],
) -> pd.DataFrame:
    """Extract raw pages from all uploaded documents."""
    all_records: list[dict[str, Any]] = []

    for position, uploaded_file in enumerate(uploaded_files, start=1):
        metadata = build_uploaded_metadata(uploaded_file=uploaded_file, position=position)
        extension = Path(uploaded_file.file_name).suffix.lower()

        if extension == ".pdf":
            all_records.extend(extract_pdf_pages(uploaded_file=uploaded_file, metadata=metadata))
        else:
            all_records.extend(extract_text_pages(uploaded_file=uploaded_file, metadata=metadata))

    if not all_records:
        raise ValueError("No uploaded document content could be extracted.")

    return pd.DataFrame(all_records)


def build_uploaded_document_summaries(
    extracted_pages_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Summarize uploaded documents for the Streamlit UI and result payloads."""
    page_counts = extracted_pages_df.groupby("doc_id")["page_num"].count().to_dict()
    char_counts = extracted_pages_df.groupby("doc_id")["char_count"].sum().to_dict()
    chunk_counts = chunks_df.groupby("doc_id")["chunk_id"].count().to_dict()

    document_rows: list[dict[str, Any]] = []
    metadata_df = (
        extracted_pages_df[
            ["doc_id", "file_name", "company", "fiscal_year", "document_type"]
        ]
        .drop_duplicates(subset=["doc_id"])
        .sort_values("file_name")
    )

    for row in metadata_df.itertuples(index=False):
        document_rows.append(
            {
                "doc_id": row.doc_id,
                "file_name": row.file_name,
                "company": row.company,
                "fiscal_year": int(row.fiscal_year),
                "document_type": row.document_type,
                "page_count": int(page_counts.get(row.doc_id, 0)),
                "chunk_count": int(chunk_counts.get(row.doc_id, 0)),
                "char_count": int(char_counts.get(row.doc_id, 0)),
            }
        )

    return document_rows


def compute_uploaded_embeddings(
    chunks_df: pd.DataFrame,
    embedding_model: str,
    encode_texts: Callable[[list[str]], list[list[float]]] | None = None,
) -> pd.DataFrame:
    """Create chunk embeddings for uploaded documents."""
    texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()

    if encode_texts is None:
        model = SentenceTransformer(embedding_model)
        embedding_matrix = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = [embedding.tolist() for embedding in embedding_matrix]
    else:
        embeddings = encode_texts(texts)

    if len(embeddings) != len(chunks_df):
        raise ValueError("Uploaded chunk embeddings do not align with the generated chunks.")

    records: list[dict[str, Any]] = []
    for row, embedding in zip(chunks_df.itertuples(index=False), embeddings):
        records.append(
            build_chunk_embedding_record(
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                company=row.company,
                fiscal_year=int(row.fiscal_year),
                embedding_model=embedding_model,
                text_hash=compute_text_hash(row.chunk_text),
                embedding=embedding,
            )
        )

    return pd.DataFrame(records)


def build_uploaded_document_bundle(
    uploaded_files: Sequence[UploadedFilePayload],
    embedding_model: str,
    corpus_id: str | None = None,
    encode_texts: Callable[[list[str]], list[list[float]]] | None = None,
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
    overlap: int = DEFAULT_UPLOAD_CHUNK_OVERLAP,
    min_chunk_words: int = DEFAULT_UPLOAD_MIN_CHUNK_WORDS,
) -> UploadedDocumentBundle:
    """Turn uploaded files into cleaned chunks plus embeddings."""
    if not uploaded_files:
        raise ValueError("At least one uploaded file is required.")

    extracted_pages_df = extract_uploaded_pages(uploaded_files)
    processed_pages_df = process_pages(
        extracted_pages_df,
        progress=False,
        aggressive=False,
    )
    chunks_df = chunk_all_documents(
        processed_pages_df=processed_pages_df,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_words=min_chunk_words,
        method="token",
        progress=False,
    )
    chunks_df = chunks_df.copy()
    chunks_df["document_source"] = "uploaded"

    chunk_embeddings_df = compute_uploaded_embeddings(
        chunks_df=chunks_df,
        embedding_model=embedding_model,
        encode_texts=encode_texts,
    )

    return UploadedDocumentBundle(
        corpus_id=corpus_id or build_uploaded_corpus_id(uploaded_files, embedding_model),
        chunks_df=chunks_df,
        chunk_embeddings_df=chunk_embeddings_df,
        documents=build_uploaded_document_summaries(
            extracted_pages_df=extracted_pages_df,
            chunks_df=chunks_df,
        ),
    )


def prepare_uploaded_runtime_corpus(
    base_chunks_path: Path,
    base_chunk_embeddings_path: Path,
    uploaded_files: Sequence[UploadedFilePayload],
    embedding_model: str,
    output_dir: Path,
    encode_texts: Callable[[list[str]], list[list[float]]] | None = None,
) -> UploadedRuntimeCorpus:
    """Build or reuse a session-scoped merged corpus for uploaded documents."""
    if not uploaded_files:
        raise ValueError("At least one uploaded file is required.")

    corpus_id = build_uploaded_corpus_id(
        uploaded_files=uploaded_files,
        embedding_model=embedding_model,
        base_chunks_path=base_chunks_path,
        base_chunk_embeddings_path=base_chunk_embeddings_path,
    )
    corpus_dir = output_dir / corpus_id
    chunks_path = corpus_dir / "chunks.parquet"
    chunk_embeddings_path = corpus_dir / "chunk_embeddings.parquet"
    manifest_path = corpus_dir / "manifest.json"

    if chunks_path.exists() and chunk_embeddings_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return UploadedRuntimeCorpus(
            corpus_id=corpus_id,
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            documents=manifest.get("documents", []),
            uploaded_chunk_count=int(manifest.get("uploaded_chunk_count", 0)),
        )

    bundle = build_uploaded_document_bundle(
        uploaded_files=uploaded_files,
        embedding_model=embedding_model,
        corpus_id=corpus_id,
        encode_texts=encode_texts,
    )

    base_chunks_df = load_dataframe(base_chunks_path)
    base_chunk_embeddings_df = load_dataframe(base_chunk_embeddings_path)
    base_chunk_embeddings_df = base_chunk_embeddings_df[
        base_chunk_embeddings_df["embedding_model"] == embedding_model
    ].copy()

    combined_chunks_df = pd.concat(
        [base_chunks_df, bundle.chunks_df],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(subset=["chunk_id"], keep="last")
    combined_chunk_embeddings_df = pd.concat(
        [base_chunk_embeddings_df, bundle.chunk_embeddings_df],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(subset=["chunk_id", "embedding_model"], keep="last")

    corpus_dir.mkdir(parents=True, exist_ok=True)
    combined_chunks_df.to_parquet(chunks_path, index=False)
    combined_chunk_embeddings_df.to_parquet(chunk_embeddings_path, index=False)

    write_manifest(
        manifest_path,
        {
            "corpus_id": corpus_id,
            "embedding_model": embedding_model,
            "documents": bundle.documents,
            "uploaded_chunk_count": int(len(bundle.chunks_df)),
        },
    )

    return UploadedRuntimeCorpus(
        corpus_id=corpus_id,
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        documents=bundle.documents,
        uploaded_chunk_count=int(len(bundle.chunks_df)),
    )
