from __future__ import annotations

import pandas as pd

from src.extraction.checkpoints import get_checkpoint_path
from src.extraction.entity_extractor import run_entity_extraction
from src.extraction.triplet_extractor import run_triplet_extraction


class DummyLLMClient:
    def __init__(self, *args, **kwargs) -> None:
        pass


def write_chunks_fixture(chunks_path) -> None:
    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe discussed AI opportunities.",
            },
            {
                "chunk_id": "chunk_2",
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "page_start": 11,
                "page_end": 11,
                "chunk_text": "Adobe discussed competition and risks.",
            },
        ]
    ).to_parquet(chunks_path, index=False)


def test_entity_extraction_checkpoints_empty_chunks_for_resume(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "entities.parquet"
    write_chunks_fixture(chunks_path)

    processed_chunk_ids: list[str] = []

    def fake_extract(chunk_row, llm_client, system_prompt):
        processed_chunk_ids.append(str(chunk_row["chunk_id"]))
        return []

    monkeypatch.setattr("src.extraction.entity_extractor.LLMClient", DummyLLMClient)
    monkeypatch.setattr(
        "src.extraction.entity_extractor.extract_entities_from_chunk",
        fake_extract,
    )

    first_df = run_entity_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        save_every=1,
    )

    assert first_df.empty
    assert processed_chunk_ids == ["chunk_1", "chunk_2"]

    checkpoint_df = pd.read_parquet(get_checkpoint_path(output_path))
    assert checkpoint_df["chunk_id"].tolist() == ["chunk_1", "chunk_2"]
    assert checkpoint_df["status"].tolist() == ["success", "success"]
    assert checkpoint_df["record_count"].tolist() == [0, 0]

    processed_chunk_ids.clear()

    second_df = run_entity_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        save_every=1,
    )

    assert second_df.empty
    assert processed_chunk_ids == []


def test_entity_extraction_retries_error_chunks_until_success(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "entities.parquet"
    write_chunks_fixture(chunks_path)

    call_count = {"count": 0}

    def fake_extract(chunk_row, llm_client, system_prompt):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise RuntimeError("temporary failure")
        return []

    monkeypatch.setattr("src.extraction.entity_extractor.LLMClient", DummyLLMClient)
    monkeypatch.setattr(
        "src.extraction.entity_extractor.extract_entities_from_chunk",
        fake_extract,
    )

    run_entity_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        max_chunks=1,
        save_every=1,
    )

    checkpoint_df = pd.read_parquet(get_checkpoint_path(output_path))
    assert checkpoint_df.loc[0, "chunk_id"] == "chunk_1"
    assert checkpoint_df.loc[0, "status"] == "error"

    run_entity_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        max_chunks=1,
        save_every=1,
    )

    checkpoint_df = pd.read_parquet(get_checkpoint_path(output_path))
    assert checkpoint_df.loc[0, "chunk_id"] == "chunk_1"
    assert checkpoint_df.loc[0, "status"] == "success"
    assert checkpoint_df.loc[0, "record_count"] == 0
    assert call_count["count"] == 2


def test_triplet_extraction_checkpoints_empty_chunks_for_resume(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "triplets.parquet"
    write_chunks_fixture(chunks_path)

    processed_chunk_ids: list[str] = []

    def fake_extract(chunk_row, llm_client, system_prompt):
        processed_chunk_ids.append(str(chunk_row["chunk_id"]))
        return []

    monkeypatch.setattr("src.extraction.triplet_extractor.LLMClient", DummyLLMClient)
    monkeypatch.setattr(
        "src.extraction.triplet_extractor.extract_triplets_from_chunk",
        fake_extract,
    )

    first_df = run_triplet_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        save_every=1,
    )

    assert first_df.empty
    assert processed_chunk_ids == ["chunk_1", "chunk_2"]

    checkpoint_df = pd.read_parquet(get_checkpoint_path(output_path))
    assert checkpoint_df["chunk_id"].tolist() == ["chunk_1", "chunk_2"]
    assert checkpoint_df["status"].tolist() == ["success", "success"]
    assert checkpoint_df["record_count"].tolist() == [0, 0]

    processed_chunk_ids.clear()

    second_df = run_triplet_extraction(
        chunks_path=chunks_path,
        output_path=output_path,
        llm_model="dummy-model",
        llm_cache_path=tmp_path / "llm_cache.jsonl",
        save_every=1,
    )

    assert second_df.empty
    assert processed_chunk_ids == []
