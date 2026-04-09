from __future__ import annotations

import pandas as pd

from src.evaluation.financebench_loader import (
    LOCAL_SMOKE_SCOPE,
    build_readme_text,
    extract_local_smoke_scope_pairs,
    format_company_year_scope,
    normalize_financebench_records,
    select_local_smoke_subset,
    select_subset_by_doc_limits,
)


def test_normalize_financebench_records_extracts_expected_fields() -> None:
    records = [
        {
            "financebench_id": "financebench_id_1",
            "company": "Adobe",
            "doc_name": "ADOBE_2022_10K",
            "question_type": "domain-relevant",
            "question_reasoning": "Information extraction",
            "domain_question_num": "dg01",
            "question": "What changed in FY2022?",
            "answer": "Revenue increased.",
            "justification": "The increase came from subscriptions.",
            "dataset_subset_label": "OPEN_SOURCE",
            "gics_sector": "Information Technology",
            "doc_type": "10k",
            "doc_period": 2022,
            "doc_link": "https://example.com/adobe-2022-10k.pdf",
            "evidence": [
                {
                    "doc_name": "ADOBE_2022_10K",
                    "evidence_page_num": 87,
                    "evidence_text": "Revenue increased year over year.",
                },
                {
                    "doc_name": "ADOBE_2022_10K",
                    "evidence_page_num": 88,
                    "evidence_text": "Subscription revenue drove the increase.",
                },
            ],
        }
    ]

    df = normalize_financebench_records(records)

    assert len(df) == 1
    assert df.loc[0, "query_text"] == "What changed in FY2022?"
    assert df.loc[0, "expected_doc_id"] == "adobe_2022_10k"
    assert df.loc[0, "company_slug"] == "adobe"
    assert df.loc[0, "evidence_pages"] == "87|88"
    assert int(df.loc[0, "primary_evidence_page"]) == 87
    assert df.loc[0, "primary_evidence_doc_name"] == "ADOBE_2022_10K"
    assert bool(df.loc[0, "has_justification"]) is True


def test_select_subset_by_doc_limits_prefers_structured_questions() -> None:
    df = pd.DataFrame(
        [
            {
                "financebench_id": "id_domain",
                "company": "AMD",
                "doc_period": 2022,
                "doc_type": "10k",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 2,
                "question_length_chars": 50,
            },
            {
                "financebench_id": "id_metrics",
                "company": "AMD",
                "doc_period": 2022,
                "doc_type": "10k",
                "question_type": "metrics-generated",
                "question_reasoning": "Numerical reasoning",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 2,
                "question_length_chars": 60,
            },
            {
                "financebench_id": "id_novel",
                "company": "AMD",
                "doc_period": 2022,
                "doc_type": "10k",
                "question_type": "novel-generated",
                "question_reasoning": "",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 1,
                "question_length_chars": 40,
            },
        ]
    )

    subset_df = select_subset_by_doc_limits(
        df=df,
        doc_limits=[("AMD", 2022, 2)],
        subset_name="test_subset",
    )

    assert subset_df["financebench_id"].tolist() == ["id_domain", "id_metrics"]
    assert subset_df["subset_name"].tolist() == ["test_subset", "test_subset"]


def test_select_local_smoke_subset_keeps_only_current_overlap_and_only_10ks() -> None:
    df = pd.DataFrame(
        [
            {
                "financebench_id": "id_1",
                "company": "Adobe",
                "doc_period": 2022,
                "doc_type": "10k",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 1,
                "question_length_chars": 10,
            },
            {
                "financebench_id": "id_2",
                "company": "Pfizer",
                "doc_period": 2024,
                "doc_type": "10k",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 1,
                "question_length_chars": 10,
            },
            {
                "financebench_id": "id_2b",
                "company": "Pfizer",
                "doc_period": 2024,
                "doc_type": "10q",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 1,
                "question_length_chars": 10,
            },
            {
                "financebench_id": "id_3",
                "company": "Boeing",
                "doc_period": 2022,
                "doc_type": "10k",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
                "doc_link": "https://example.com/doc.pdf",
                "evidence_count": 1,
                "question_length_chars": 10,
            },
        ]
    )

    subset_df = select_local_smoke_subset(df)

    assert subset_df["financebench_id"].tolist() == ["id_1", "id_2"]
    assert set(subset_df["subset_name"]) == {"local_smoke"}


def test_local_smoke_scope_is_centralized_and_human_readable() -> None:
    assert format_company_year_scope(LOCAL_SMOKE_SCOPE) == (
        "`Adobe 2022-2024`, `Lockheed Martin 2022-2024`, `Pfizer 2022-2024`"
    )


def test_format_company_year_scope_supports_custom_empty_label() -> None:
    assert (
        format_company_year_scope([], empty_label="no overlap in the sample")
        == "no overlap in the sample"
    )


def test_extract_local_smoke_scope_pairs_uses_actual_subset_values() -> None:
    local_smoke_df = pd.DataFrame(
        [
            {"company": "Adobe", "doc_period": 2022},
            {"company": "Adobe", "doc_period": 2022},
            {"company": "Lockheed Martin", "doc_period": 2022},
        ]
    )

    assert extract_local_smoke_scope_pairs(local_smoke_df) == [
        ("Adobe", 2022),
        ("Lockheed Martin", 2022),
    ]


def test_build_readme_text_describes_actual_local_smoke_overlap_dynamically() -> None:
    full_df = pd.DataFrame(
        [
            {"doc_type": "10k", "doc_name": "ADOBE_2022_10K", "company": "Adobe"},
            {"doc_type": "10k", "doc_name": "PFIZER_2024_10K", "company": "Pfizer"},
            {"doc_type": "8k", "doc_name": "ADOBE_2022_8K", "company": "Adobe"},
        ]
    )
    core40_df = pd.DataFrame(
        [
            {
                "doc_name": "ADOBE_2022_10K",
                "company": "Adobe",
                "question_type": "domain-relevant",
                "question_reasoning": "Information extraction",
            }
        ]
    )
    local_smoke_df = pd.DataFrame(
        [
            {"doc_name": "ADOBE_2022_10K", "company": "Adobe", "doc_period": 2022},
            {"doc_name": "PFIZER_2024_10K", "company": "Pfizer", "doc_period": 2024},
        ]
    )
    docs_manifest_df = pd.DataFrame(
        [
            {
                "company": "Adobe",
                "doc_period": 2022,
                "doc_name": "ADOBE_2022_10K",
                "n_questions": 1,
                "question_types": "domain-relevant",
            }
        ]
    )

    readme_text = build_readme_text(
        full_df=full_df,
        core40_df=core40_df,
        local_smoke_df=local_smoke_df,
        docs_manifest_df=docs_manifest_df,
        source_url="https://example.com/financebench.jsonl",
    )

    assert (
        "- `local_smoke` is filtered on this configured local corpus scope: "
        "`Adobe 2022-2024`, `Lockheed Martin 2022-2024`, `Pfizer 2022-2024`."
    ) in readme_text
    assert (
        "- In the current open-source FinanceBench sample, that scope yields this actual overlap: "
        "`Adobe 2022`, `Pfizer 2024`."
    ) in readme_text
    assert "`Lockheed Martin 2022`." not in readme_text
