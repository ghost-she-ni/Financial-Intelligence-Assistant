from __future__ import annotations

import pandas as pd

from src.retrieval.retrieve import (
    build_source_signature_payload,
    normalize_retrieval_mode,
    retrieve_top_k_with_mode,
)


def test_normalize_retrieval_mode_supports_legacy_aliases() -> None:
    assert normalize_retrieval_mode("classical_ml") == "classical_ml"
    assert normalize_retrieval_mode("ml") == "classical_ml"
    assert normalize_retrieval_mode("naive") == "naive"
    assert normalize_retrieval_mode("baseline") == "naive"
    assert normalize_retrieval_mode("improved") == "improved"
    assert normalize_retrieval_mode("dense_hybrid") == "improved"


def test_source_signature_payload_uses_portable_hashes(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    chunks_path = tmp_path / "data" / "processed" / "chunks.parquet"
    embeddings_path = tmp_path / "data" / "embeddings" / "chunk_embeddings.parquet"
    chunks_path.parent.mkdir(parents=True)
    embeddings_path.parent.mkdir(parents=True)
    chunks_path.write_text("chunks", encoding="utf-8")
    embeddings_path.write_text("embeddings", encoding="utf-8")

    payload = build_source_signature_payload(
        chunks_path=chunks_path,
        chunk_embeddings_path=embeddings_path,
    )

    assert payload["chunks"]["path"] == "data/processed/chunks.parquet"
    assert payload["chunk_embeddings"]["path"] == "data/embeddings/chunk_embeddings.parquet"
    assert "sha256" in payload["chunks"]
    assert "mtime_ns" not in payload["chunks"]


def test_retrieve_top_k_with_mode_naive_returns_dense_baseline_schema(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    chunk_embeddings_path = tmp_path / "chunk_embeddings.parquet"
    query_embeddings_path = tmp_path / "query_embeddings.parquet"

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "adobe_2022_10k",
                "company": "adobe",
                "fiscal_year": 2022,
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe revenue improved in 2022.",
            },
            {
                "chunk_id": "chunk_2",
                "doc_id": "pfizer_2022_10k",
                "company": "pfizer",
                "fiscal_year": 2022,
                "page_start": 8,
                "page_end": 8,
                "chunk_text": "Pfizer launched a new product.",
            },
        ]
    ).to_parquet(chunks_path, index=False)

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "embedding_model": "test-model",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "chunk_2",
                "embedding_model": "test-model",
                "embedding": [0.0, 1.0],
            },
        ]
    ).to_parquet(chunk_embeddings_path, index=False)

    pd.DataFrame(
        [
            {
                "query_id": "q_1",
                "query_text": "Did Adobe improve revenue?",
                "embedding_model": "test-model",
                "embedding": [1.0, 0.0],
            }
        ]
    ).to_parquet(query_embeddings_path, index=False)

    results_df = retrieve_top_k_with_mode(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        query_embeddings_path=query_embeddings_path,
        embedding_model="test-model",
        top_k=1,
        retrieval_mode="naive",
        query_text="Did Adobe improve revenue?",
        enable_metadata_filters=False,
        enable_noise_filter=False,
        persistent_index_mode="source",
    )

    assert results_df["chunk_id"].tolist() == ["chunk_1"]
    assert results_df.loc[0, "retrieval_mode"] == "naive"
    assert float(results_df.loc[0, "score"]) == float(results_df.loc[0, "final_score"])
    assert float(results_df.loc[0, "lexical_score"]) == 0.0
    assert float(results_df.loc[0, "hybrid_score"]) == 0.0


def test_retrieve_top_k_with_mode_classical_ml_returns_tfidf_baseline_schema(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    chunk_embeddings_path = tmp_path / "chunk_embeddings.parquet"
    query_embeddings_path = tmp_path / "query_embeddings.parquet"

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "adobe_2022_10k",
                "company": "adobe",
                "fiscal_year": 2022,
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe revenue improved in 2022 with subscription growth.",
            },
            {
                "chunk_id": "chunk_2",
                "doc_id": "pfizer_2022_10k",
                "company": "pfizer",
                "fiscal_year": 2022,
                "page_start": 8,
                "page_end": 8,
                "chunk_text": "Pfizer launched a new product and expanded trials.",
            },
        ]
    ).to_parquet(chunks_path, index=False)

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "embedding_model": "test-model",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "chunk_2",
                "embedding_model": "test-model",
                "embedding": [0.0, 1.0],
            },
        ]
    ).to_parquet(chunk_embeddings_path, index=False)

    pd.DataFrame(
        [
            {
                "query_id": "q_1",
                "query_text": "Did Adobe improve revenue?",
                "embedding_model": "test-model",
                "embedding": [1.0, 0.0],
            }
        ]
    ).to_parquet(query_embeddings_path, index=False)

    results_df = retrieve_top_k_with_mode(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        query_embeddings_path=query_embeddings_path,
        embedding_model="test-model",
        top_k=1,
        retrieval_mode="classical_ml",
        query_text="Did Adobe improve revenue?",
        enable_metadata_filters=False,
        enable_noise_filter=False,
        persistent_index_mode="source",
    )

    assert results_df["chunk_id"].tolist() == ["chunk_1"]
    assert results_df.loc[0, "retrieval_mode"] == "classical_ml"
    assert float(results_df.loc[0, "score"]) == float(results_df.loc[0, "final_score"])
    assert float(results_df.loc[0, "score"]) > 0.0
    assert float(results_df.loc[0, "hybrid_score"]) == 0.0
