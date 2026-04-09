from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.workflow import run_financial_analyst_agent
from src.extraction.knowledge_base import get_knowledge_artifacts
from src.generation.rag_answer import generate_rag_answer

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
CHUNK_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
QUERY_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"
LLM_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_responses.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "streamlit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

SAMPLE_QUESTIONS = [
    "What are Adobe's main risk factors in 2024?",
    "How did Lockheed Martin describe competition in 2023?",
    "What financial metrics are highlighted in Pfizer's 2024 annual report?",
    "Did Adobe mention AI-related opportunities or risks in 2024?",
    "What are the main business segments discussed by Pfizer in 2023?",
]

COMPANY_OPTIONS = [
    ("All", None),
    ("Adobe", "adobe"),
    ("Lockheed Martin", "lockheedmartin"),
    ("Pfizer", "pfizer"),
]


def format_citation(citation: dict) -> str:
    return f"{citation['doc_id']} | page {citation['page']}"


def company_label_to_value(label: str) -> str | None:
    for option_label, option_value in COMPANY_OPTIONS:
        if option_label == label:
            return option_value
    return None


def year_label_to_value(label: str) -> int | None:
    return None if label == "All" else int(label)


def apply_filters(
    df: pd.DataFrame,
    company_filter: str | None,
    year_filter: int | None,
    company_column: str,
    year_column: str,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if company_filter is not None and company_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[company_column] == company_filter]

    if year_filter is not None and year_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[year_column] == year_filter]

    return filtered_df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_knowledge_views():
    return get_knowledge_artifacts(chunks_path=CHUNKS_PATH)


def render_retrieval_results(retrieval_results: list[dict]) -> None:
    if not retrieval_results:
        st.info("No retrieval results available.")
        return

    st.subheader("Retrieved Chunks")

    for idx, row in enumerate(retrieval_results, start=1):
        title = (
            f"Rank {idx} - {row.get('doc_id', 'unknown_doc')} "
            f"(pages {row.get('page_start', '?')}-{row.get('page_end', '?')})"
        )

        with st.expander(title):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Company", row.get("company", "N/A"))
            col2.metric("Fiscal year", row.get("fiscal_year", "N/A"))
            col3.metric("Final score", f"{row.get('final_score', row.get('score', 0.0)):.4f}")
            col4.metric("Knowledge", f"{row.get('knowledge_score', 0.0):.4f}")

            section_group = row.get("section_group")
            section_title = row.get("section_title")
            if section_group or section_title:
                st.caption(
                    f"Section: {section_group or 'unknown'} | {section_title or 'unknown'}"
                )

            st.caption(
                f"Cosine: {row.get('score', 0.0):.4f} | "
                f"BM25: {row.get('bm25_score', 0.0):.4f} | "
                f"Coverage: {row.get('coverage_score', 0.0):.4f} | "
                f"Section: {row.get('section_score', 0.0):.4f} | "
                f"Knowledge: {row.get('knowledge_score', 0.0):.4f}"
            )

            entities_preview = row.get("knowledge_entities_preview", "")
            triplets_preview = row.get("knowledge_triplets_preview", "")
            if entities_preview:
                st.caption(f"Entities: {entities_preview}")
            if triplets_preview:
                st.caption(f"Triplets: {triplets_preview}")

            st.text_area(
                label=f"Chunk text {idx}",
                value=row.get("chunk_text", ""),
                height=220,
                key=f"chunk_text_{idx}",
            )


def render_safety_flags(flags: list[str]) -> None:
    if not flags:
        return

    st.subheader("Safety Flags")
    st.caption("Guardrail-related signals attached to the final answer.")
    st.write(", ".join(f"`{flag}`" for flag in flags))


def render_tool_trace(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return

    st.subheader("Agent Tool Trace")
    for tool_call in tool_calls:
        title = (
            f"Step {tool_call.get('step', '?')} - {tool_call.get('tool_name', 'unknown_tool')} "
            f"({tool_call.get('status', 'unknown')})"
        )
        with st.expander(title):
            st.caption(tool_call.get("summary", ""))
            st.code(
                json.dumps(tool_call.get("arguments", {}), indent=2, ensure_ascii=False),
                language="json",
            )
            st.code(
                json.dumps(tool_call.get("result", {}), indent=2, ensure_ascii=False),
                language="json",
            )


def render_knowledge_tab(
    knowledge_artifacts,
    company_filter: str | None,
    year_filter: int | None,
) -> None:
    entities_df = apply_filters(
        knowledge_artifacts.entities_df,
        company_filter=company_filter,
        year_filter=year_filter,
        company_column="company",
        year_column="year",
    )
    triplets_df = apply_filters(
        knowledge_artifacts.triplets_df,
        company_filter=company_filter,
        year_filter=year_filter,
        company_column="company",
        year_column="year",
    )

    if entities_df.empty and triplets_df.empty:
        st.info("No extracted entities or triplets are available for the current filters.")
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Entity rows", len(entities_df))
    metric2.metric("Triplet rows", len(triplets_df))
    metric3.metric(
        "Unique chunks with knowledge",
        pd.Index(entities_df["chunk_id"]).union(pd.Index(triplets_df["chunk_id"])).nunique(),
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Entities")
        entity_types = sorted(entities_df["entity_type"].dropna().astype(str).unique().tolist())
        selected_entity_types = st.multiselect(
            "Entity types",
            options=entity_types,
            default=entity_types,
            key="entity_types_filter",
        )
        visible_entities_df = entities_df.copy()
        if selected_entity_types:
            visible_entities_df = visible_entities_df[
                visible_entities_df["entity_type"].isin(selected_entity_types)
            ]

        top_entities_df = (
            visible_entities_df.groupby(["entity_text", "entity_type"], as_index=False)
            .agg(
                mention_count=("chunk_id", "count"),
                first_year=("year", "min"),
                last_year=("year", "max"),
            )
            .sort_values(["mention_count", "entity_text"], ascending=[False, True])
        )
        st.dataframe(top_entities_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Triplets")
        relations = sorted(triplets_df["relation"].dropna().astype(str).unique().tolist())
        selected_relations = st.multiselect(
            "Relations",
            options=relations,
            default=relations,
            key="relation_filter",
        )
        visible_triplets_df = triplets_df.copy()
        if selected_relations:
            visible_triplets_df = visible_triplets_df[
                visible_triplets_df["relation"].isin(selected_relations)
            ]

        top_triplets_df = (
            visible_triplets_df.groupby(["entity_a", "relation", "entity_b"], as_index=False)
            .agg(
                mention_count=("chunk_id", "count"),
                first_year=("year", "min"),
                last_year=("year", "max"),
            )
            .sort_values(["mention_count", "relation"], ascending=[False, True])
        )
        st.dataframe(top_triplets_df, use_container_width=True, hide_index=True)

    st.subheader("Source Evidence")
    evidence_mode = st.radio(
        "Evidence source",
        options=["Entities", "Triplets"],
        horizontal=True,
    )

    if evidence_mode == "Entities":
        evidence_df = entities_df[
            [
                "entity_text",
                "entity_type",
                "company",
                "year",
                "source_doc_id",
                "page_start",
                "page_end",
                "chunk_id",
            ]
        ].copy()
    else:
        evidence_df = triplets_df[
            [
                "entity_a",
                "relation",
                "entity_b",
                "company",
                "year",
                "doc_id",
                "page_start",
                "page_end",
                "chunk_id",
            ]
        ].copy()

    st.dataframe(evidence_df, use_container_width=True, hide_index=True)


def render_competitors_tab(
    knowledge_artifacts,
    company_filter: str | None,
    year_filter: int | None,
) -> None:
    use_clean_summary = st.checkbox(
        "Use clean competitor summary",
        value=True,
        help="The clean view keeps only stronger competitor evidence.",
    )

    summary_df = (
        knowledge_artifacts.competitor_summary_clean_df
        if use_clean_summary
        else knowledge_artifacts.competitor_summary_df
    )
    new_competitors_df = (
        knowledge_artifacts.new_competitors_clean_df
        if use_clean_summary
        else knowledge_artifacts.new_competitors_df
    )

    summary_df = apply_filters(
        summary_df,
        company_filter=company_filter,
        year_filter=year_filter,
        company_column="source_company",
        year_column="year",
    )
    mentions_df = apply_filters(
        knowledge_artifacts.competitor_mentions_df,
        company_filter=company_filter,
        year_filter=year_filter,
        company_column="source_company",
        year_column="year",
    )
    new_competitors_df = apply_filters(
        new_competitors_df,
        company_filter=company_filter,
        year_filter=year_filter,
        company_column="source_company",
        year_column="year",
    )

    if summary_df.empty:
        st.info("No competitor evidence is available for the current filters.")
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Competitor rows", len(summary_df))
    metric2.metric("Unique competitors", summary_df["competitor_name"].nunique())
    metric3.metric("Evidence chunks", mentions_df["chunk_id"].nunique())

    st.subheader("Competitor Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("New Competitors by Year")
    if new_competitors_df.empty:
        st.caption("No new competitors identified in the current slice.")
    else:
        st.dataframe(new_competitors_df, use_container_width=True, hide_index=True)

    competitor_options = summary_df["competitor_name"].dropna().astype(str).unique().tolist()
    selected_competitor = st.selectbox(
        "Inspect competitor evidence",
        options=competitor_options,
    )

    competitor_mentions_df = mentions_df[
        mentions_df["competitor_name"] == selected_competitor
    ].copy()
    competitor_mentions_df = competitor_mentions_df[
        [
            "source_company",
            "year",
            "competitor_name",
            "mention_source",
            "explicit_competes_with",
            "has_competition_risk_signal",
            "source_doc_id",
            "page_start",
            "page_end",
            "chunk_id",
            "chunk_text_preview",
        ]
    ]

    st.subheader("Evidence")
    st.dataframe(competitor_mentions_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Financial RAG Demo",
        page_icon="F",
        layout="wide",
    )

    st.title("Financial Intelligence Demo")
    st.write(
        "Ask questions over the 10-K corpus, inspect extracted knowledge, and review "
        "competitor evidence with source pages."
    )

    knowledge_artifacts = load_knowledge_views()

    with st.sidebar:
        st.header("Settings")
        assistant_mode = st.radio(
            "Assistant mode",
            options=["Direct RAG", "Agent Analyst"],
            index=0,
            help="Direct RAG answers from retrieved chunks immediately. Agent Analyst can call local tools before answering.",
        )
        top_k = st.slider("Top-k retrieved chunks", min_value=3, max_value=8, value=5, step=1)
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=["improved", "naive", "classical_ml"],
            index=0,
            help="Compare the improved retriever against the naive dense baseline and the classical TF-IDF baseline.",
        )
        embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
        llm_model = st.text_input("LLM model", value=DEFAULT_LLM_MODEL)
        show_retrieval = st.checkbox("Show retrieved chunks", value=True)

        st.markdown("---")
        st.subheader("Filters")
        company_label = st.selectbox(
            "Company",
            options=[label for label, _ in COMPANY_OPTIONS],
            index=0,
        )
        year_label = st.selectbox(
            "Fiscal year",
            options=["All", "2022", "2023", "2024"],
            index=0,
        )

        st.markdown("---")
        st.caption("Knowledge extraction is used both in retrieval and in the explorer tabs.")

    company_filter = company_label_to_value(company_label)
    year_filter = year_label_to_value(year_label)

    ask_tab, knowledge_tab, competitors_tab = st.tabs(
        ["Ask", "Knowledge Explorer", "Competitors"]
    )

    with ask_tab:
        selected_question = st.selectbox(
            "Sample questions",
            options=[""] + SAMPLE_QUESTIONS,
            index=0,
        )

        question = st.text_area(
            "Your question",
            value=selected_question,
            height=100,
            placeholder="Type a question about Adobe, Lockheed Martin, or Pfizer...",
        )

        if company_filter or year_filter:
            filter_parts = []
            if company_filter:
                filter_parts.append(f"company={company_filter}")
            if year_filter:
                filter_parts.append(f"fiscal_year={year_filter}")
            st.caption("Active retrieval filters: " + " | ".join(filter_parts))

        run_button = st.button("Run Assistant", type="primary")

        if run_button:
            if not question.strip():
                st.warning("Please enter a question.")
                st.stop()

            try:
                with st.spinner("Running retrieval and generation..."):
                    if assistant_mode == "Agent Analyst":
                        result = run_financial_analyst_agent(
                            question=question.strip(),
                            llm_model=llm_model,
                            llm_cache_path=LLM_CACHE_PATH,
                            chunks_path=CHUNKS_PATH,
                            chunk_embeddings_path=CHUNK_EMBEDDINGS_PATH,
                            query_embeddings_path=QUERY_EMBEDDINGS_PATH,
                            embedding_model=embedding_model,
                            top_k=top_k,
                            retrieval_mode=retrieval_mode,
                            company_filter=company_filter,
                            fiscal_year_filter=year_filter,
                        )
                    else:
                        result = generate_rag_answer(
                            question=question.strip(),
                            chunks_path=CHUNKS_PATH,
                            chunk_embeddings_path=CHUNK_EMBEDDINGS_PATH,
                            query_embeddings_path=QUERY_EMBEDDINGS_PATH,
                            embedding_model=embedding_model,
                            llm_model=llm_model,
                            top_k=top_k,
                            retrieval_mode=retrieval_mode,
                            llm_cache_path=LLM_CACHE_PATH,
                            company_filter=company_filter,
                            fiscal_year_filter=year_filter,
                        )

                st.subheader("Answer")
                st.write(result["answer"] or "No answer returned.")

                st.subheader("Citations")
                if result["citations"]:
                    for citation in result["citations"]:
                        st.markdown(f"- `{format_citation(citation)}`")
                else:
                    st.info("No validated citations returned.")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Top-k", result["top_k"])
                col2.metric("Mode", result.get("mode", assistant_mode))
                col3.metric("LLM cache hit", "Yes" if result["llm_from_cache"] else "No")
                col4.metric("Company/year", f"{company_filter or 'None'} / {year_filter or 'None'}")
                st.caption(f"Retrieval mode: {result.get('retrieval_mode', retrieval_mode)}")

                render_safety_flags(result.get("safety_flags", []))
                render_tool_trace(result.get("tool_calls", []))

                if show_retrieval:
                    render_retrieval_results(result["retrieval_results"])

                output_path = OUTPUT_DIR / f"latest_{result.get('mode', 'assistant')}_result.json"
                output_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                st.download_button(
                    label="Download latest result as JSON",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"{result.get('mode', 'assistant')}_result.json",
                    mime="application/json",
                )

            except Exception as exc:
                st.error("The assistant pipeline failed.")
                st.exception(exc)

    with knowledge_tab:
        render_knowledge_tab(
            knowledge_artifacts=knowledge_artifacts,
            company_filter=company_filter,
            year_filter=year_filter,
        )

    with competitors_tab:
        render_competitors_tab(
            knowledge_artifacts=knowledge_artifacts,
            company_filter=company_filter,
            year_filter=year_filter,
        )


if __name__ == "__main__":
    main()
