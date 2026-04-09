from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.uploaded_documents import (
    UploadedFilePayload,
    build_uploaded_document_bundle,
    prepare_uploaded_runtime_corpus,
)


def fake_encode_texts(texts: list[str]) -> list[list[float]]:
    return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


def test_build_uploaded_document_bundle_creates_chunks_and_embeddings() -> None:
    uploaded_files = [
        UploadedFilePayload(
            file_name="meeting_notes_2025.txt",
            content=(
                b"Revenue expanded in 2025 because AI products improved retention and pricing. "
                b"Management also highlighted margin expansion and enterprise demand."
            ),
        )
    ]

    bundle = build_uploaded_document_bundle(
        uploaded_files=uploaded_files,
        embedding_model="test-model",
        encode_texts=fake_encode_texts,
    )

    assert bundle.corpus_id
    assert bundle.documents[0]["file_name"] == "meeting_notes_2025.txt"
    assert bundle.documents[0]["fiscal_year"] == 2025
    assert not bundle.chunks_df.empty
    assert bundle.chunks_df["document_source"].unique().tolist() == ["uploaded"]
    assert bundle.chunk_embeddings_df["embedding_model"].unique().tolist() == ["test-model"]


def test_prepare_uploaded_runtime_corpus_merges_base_and_uploaded_chunks(tmp_path: Path) -> None:
    base_chunks_path = tmp_path / "chunks.parquet"
    base_chunk_embeddings_path = tmp_path / "chunk_embeddings.parquet"
    runtime_dir = tmp_path / "runtime_corpus"

    pd.DataFrame(
        [
            {
                "chunk_id": "base_chunk_1",
                "chunk_index": 0,
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "document_type": "10k",
                "file_name": "adobe_2024_10k.pdf",
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe reported stronger subscription revenue in 2024.",
                "word_count": 8,
                "token_count": 8,
                "char_count": 55,
                "section_id": "business__01",
                "section_code": "business",
                "section_title": "Business",
                "section_group": "business",
            }
        ]
    ).to_parquet(base_chunks_path, index=False)

    pd.DataFrame(
        [
            {
                "chunk_id": "base_chunk_1",
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "embedding_model": "test-model",
                "text_hash": "hash-base",
                "embedding": [0.1, 0.9],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_parquet(base_chunk_embeddings_path, index=False)

    uploaded_files = [
        UploadedFilePayload(
            file_name="analysis_2025.md",
            content=(
                b"# Analysis\n\nThe uploaded memo says AI demand accelerated and operating margins improved."
            ),
        )
    ]

    runtime_corpus = prepare_uploaded_runtime_corpus(
        base_chunks_path=base_chunks_path,
        base_chunk_embeddings_path=base_chunk_embeddings_path,
        uploaded_files=uploaded_files,
        embedding_model="test-model",
        output_dir=runtime_dir,
        encode_texts=fake_encode_texts,
    )

    merged_chunks_df = pd.read_parquet(runtime_corpus.chunks_path)
    merged_embeddings_df = pd.read_parquet(runtime_corpus.chunk_embeddings_path)

    assert runtime_corpus.chunks_path.exists()
    assert runtime_corpus.chunk_embeddings_path.exists()
    assert "base_chunk_1" in merged_chunks_df["chunk_id"].tolist()
    assert (merged_chunks_df["document_source"] == "uploaded").sum() >= 1
    assert set(merged_embeddings_df["embedding_model"].unique().tolist()) == {"test-model"}
    assert runtime_corpus.documents[0]["file_name"] == "analysis_2025.md"

    reused_runtime_corpus = prepare_uploaded_runtime_corpus(
        base_chunks_path=base_chunks_path,
        base_chunk_embeddings_path=base_chunk_embeddings_path,
        uploaded_files=uploaded_files,
        embedding_model="test-model",
        output_dir=runtime_dir,
        encode_texts=fake_encode_texts,
    )

    assert reused_runtime_corpus.chunks_path == runtime_corpus.chunks_path
    assert reused_runtime_corpus.chunk_embeddings_path == runtime_corpus.chunk_embeddings_path
