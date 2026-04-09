from __future__ import annotations

import numpy as np
import pandas as pd

from src.embeddings.cache import (
    build_chunk_embedding_record,
    build_query_embedding_record,
    compute_text_hash,
    make_query_id,
)
from src.embeddings.embed_chunks import process_missing_chunks
from src.embeddings.embed_queries import process_missing_queries
from src.generation.rag_answer import ensure_query_embeddings_cached


class DummyEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        return np.asarray(
            [[float(len(text)), 1.0] for text in texts],
            dtype=np.float32,
        )


def test_process_missing_chunks_recomputes_blank_text_hash_and_replaces_stale_entry(tmp_path) -> None:
    cache_path = tmp_path / "chunk_embeddings.parquet"
    chunk_text = "Revenue increased year over year."
    existing_cache_df = pd.DataFrame(
        [
            build_chunk_embedding_record(
                chunk_id="chunk_1",
                doc_id="adobe_2024_10k",
                company="adobe",
                fiscal_year=2024,
                embedding_model="test-model",
                text_hash="",
                embedding=[0.0, 0.0],
            )
        ]
    )

    missing_chunks_df = pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "chunk_text": chunk_text,
                "text_hash": "",
            }
        ]
    )

    updated_cache_df = process_missing_chunks(
        missing_chunks_df=missing_chunks_df,
        cache_df=existing_cache_df,
        cache_path=cache_path,
        model=DummyEmbeddingModel(),
        embedding_model_name="test-model",
        batch_size=8,
        save_every_batches=1,
    )

    assert len(updated_cache_df) == 1
    assert updated_cache_df.loc[0, "text_hash"] == compute_text_hash(chunk_text)

    saved_cache_df = pd.read_parquet(cache_path)
    assert len(saved_cache_df) == 1
    assert saved_cache_df.loc[0, "text_hash"] == compute_text_hash(chunk_text)


def test_process_missing_queries_recomputes_blank_text_hash_and_replaces_stale_entry(tmp_path) -> None:
    cache_path = tmp_path / "query_embeddings.parquet"
    query_text = "What changed in 2024?"
    query_id = make_query_id(query_text)
    existing_cache_df = pd.DataFrame(
        [
            build_query_embedding_record(
                query_id=query_id,
                query_text=query_text,
                embedding_model="test-model",
                text_hash="",
                embedding=[0.0, 0.0],
            )
        ]
    )

    missing_queries_df = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "query_text": query_text,
                "text_hash": "",
            }
        ]
    )

    updated_cache_df = process_missing_queries(
        missing_queries_df=missing_queries_df,
        cache_df=existing_cache_df,
        cache_path=cache_path,
        model=DummyEmbeddingModel(),
        embedding_model_name="test-model",
        batch_size=8,
        save_every_batches=1,
    )

    assert len(updated_cache_df) == 1
    assert updated_cache_df.loc[0, "text_hash"] == compute_text_hash(query_text)

    saved_cache_df = pd.read_parquet(cache_path)
    assert len(saved_cache_df) == 1
    assert saved_cache_df.loc[0, "text_hash"] == compute_text_hash(query_text)


def test_ensure_query_embeddings_cached_replaces_stale_hash_for_same_query(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "query_embeddings.parquet"
    query_text = "Did Adobe mention AI opportunities?"
    query_id = make_query_id(query_text)
    stale_cache_df = pd.DataFrame(
        [
            build_query_embedding_record(
                query_id=query_id,
                query_text=query_text,
                embedding_model="test-model",
                text_hash="",
                embedding=[0.0, 0.0],
            )
        ]
    )
    stale_cache_df.to_parquet(cache_path, index=False)

    monkeypatch.setattr(
        "src.generation.rag_answer.load_sentence_transformer",
        lambda model_name, device=None: DummyEmbeddingModel(),
    )

    ensure_query_embeddings_cached(
        query_texts=[query_text],
        query_embeddings_path=cache_path,
        embedding_model_name="test-model",
        batch_size=8,
    )

    saved_cache_df = pd.read_parquet(cache_path)
    assert len(saved_cache_df) == 1
    assert saved_cache_df.loc[0, "query_id"] == query_id
    assert saved_cache_df.loc[0, "text_hash"] == compute_text_hash(query_text)
