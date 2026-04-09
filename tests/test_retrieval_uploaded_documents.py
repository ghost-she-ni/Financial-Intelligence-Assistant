from __future__ import annotations

import pandas as pd

from src.retrieval.retrieve import build_retrieval_mask


def test_uploaded_documents_bypass_company_and_year_filters() -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "chunk_id": "adobe_chunk",
                "doc_id": "adobe_2024_10k",
                "company": "adobe",
                "fiscal_year": 2024,
                "page_start": 10,
                "page_end": 10,
                "chunk_text": "Adobe discussed subscription growth.",
                "document_source": None,
            },
            {
                "chunk_id": "pfizer_chunk",
                "doc_id": "pfizer_2024_10k",
                "company": "pfizer",
                "fiscal_year": 2024,
                "page_start": 12,
                "page_end": 12,
                "chunk_text": "Pfizer discussed product launches.",
                "document_source": None,
            },
            {
                "chunk_id": "uploaded_chunk",
                "doc_id": "uploaded_notes_abc123",
                "company": "meeting_notes",
                "fiscal_year": 0,
                "page_start": 1,
                "page_end": 1,
                "chunk_text": "The uploaded notes mention Adobe's AI pricing strategy.",
                "document_source": "uploaded",
            },
        ]
    )

    mask = build_retrieval_mask(
        metadata_df=metadata_df,
        query_text="What did Adobe say about AI in 2024?",
        enable_metadata_filters=False,
        enable_noise_filter=False,
        company_filter="adobe",
        fiscal_year_filter=2024,
    )

    assert mask.tolist() == [True, False, True]
