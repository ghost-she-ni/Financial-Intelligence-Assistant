from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.extract_pdf_text import extract_all_pages
from src.preprocessing.chunking import chunk_all_documents
from src.preprocessing.clean_text import process_pages


def test_extract_all_pages_orders_same_company_year_by_doc_id(monkeypatch) -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "doc_id": "acme_2023_10k_b",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "b.pdf",
                "file_path": "b.pdf",
            },
            {
                "doc_id": "acme_2023_10k_a",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "a.pdf",
                "file_path": "a.pdf",
            },
        ]
    )

    def fake_extract_pages_from_pdf(
        pdf_path: Path,
        doc_id: str,
        company: str,
        fiscal_year: int,
        document_type: str,
        file_name: str,
        max_pages: int | None = None,
    ) -> list[dict]:
        return [
            {
                "doc_id": doc_id,
                "company": company,
                "fiscal_year": fiscal_year,
                "document_type": document_type,
                "file_name": file_name,
                "page_num": 1,
                "raw_text": f"text for {doc_id}",
                "char_count": 1,
                "word_count": 1,
            }
        ]

    monkeypatch.setattr(
        "src.ingestion.extract_pdf_text.extract_pages_from_pdf",
        fake_extract_pages_from_pdf,
    )

    extracted_df = extract_all_pages(metadata_df, progress=False)

    assert extracted_df["doc_id"].tolist() == ["acme_2023_10k_a", "acme_2023_10k_b"]


def test_process_pages_orders_same_company_year_by_doc_id() -> None:
    extracted_df = pd.DataFrame(
        [
            {
                "doc_id": "acme_2023_10k_b",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "b.pdf",
                "page_num": 1,
                "raw_text": "Second document page.",
            },
            {
                "doc_id": "acme_2023_10k_a",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "a.pdf",
                "page_num": 1,
                "raw_text": "First document page.",
            },
        ]
    )

    processed_df = process_pages(extracted_df, progress=False)

    assert processed_df["doc_id"].tolist() == ["acme_2023_10k_a", "acme_2023_10k_b"]


def test_chunk_all_documents_orders_same_company_year_by_doc_id(monkeypatch) -> None:
    processed_pages_df = pd.DataFrame(
        [
            {
                "doc_id": "acme_2023_10k_b",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "b.pdf",
                "page_num": 1,
                "clean_text": "beta document with enough words for one chunk",
            },
            {
                "doc_id": "acme_2023_10k_a",
                "company": "Acme",
                "fiscal_year": 2023,
                "document_type": "10k",
                "file_name": "a.pdf",
                "page_num": 1,
                "clean_text": "alpha document with enough words for one chunk",
            },
        ]
    )

    def fake_annotate_document_sections(doc_df: pd.DataFrame) -> pd.DataFrame:
        annotated_df = doc_df.copy()
        annotated_df["section_id"] = "business__01"
        annotated_df["section_code"] = "business"
        annotated_df["section_title"] = "Business"
        annotated_df["section_group"] = "business"
        return annotated_df

    monkeypatch.setattr(
        "src.preprocessing.chunking.annotate_document_sections",
        fake_annotate_document_sections,
    )

    chunks_df = chunk_all_documents(
        processed_pages_df=processed_pages_df,
        chunk_size=100,
        overlap=0,
        min_chunk_words=1,
        method="word",
        progress=False,
    )

    assert chunks_df["doc_id"].tolist() == ["acme_2023_10k_a", "acme_2023_10k_b"]
