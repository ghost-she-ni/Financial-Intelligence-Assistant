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
from src.ingestion.uploaded_documents import (
    UploadedFilePayload,
    prepare_uploaded_runtime_corpus,
)

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
CHUNK_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
QUERY_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"
LLM_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_responses.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "streamlit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_RUNTIME_DIR = OUTPUT_DIR / "uploaded_corpora"
UPLOADED_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
UPLOAD_WIDGET_KEY_STATE = "uploaded_documents_widget_nonce"
UPLOAD_CLEAR_NOTICE_STATE = "uploaded_documents_cleared_notice"
CHAT_HISTORY_STATE = "chat_history"
CHAT_CLEAR_NOTICE_STATE = "chat_cleared_notice"

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


def ensure_upload_widget_state() -> None:
    """Initialize Streamlit session keys used by the upload widget."""
    if UPLOAD_WIDGET_KEY_STATE not in st.session_state:
        st.session_state[UPLOAD_WIDGET_KEY_STATE] = 0
    if UPLOAD_CLEAR_NOTICE_STATE not in st.session_state:
        st.session_state[UPLOAD_CLEAR_NOTICE_STATE] = False


def ensure_chat_state() -> None:
    """Initialize Streamlit session keys used by the chat interface."""
    if CHAT_HISTORY_STATE not in st.session_state:
        st.session_state[CHAT_HISTORY_STATE] = []
    if CHAT_CLEAR_NOTICE_STATE not in st.session_state:
        st.session_state[CHAT_CLEAR_NOTICE_STATE] = False


def build_uploaded_file_payloads(uploaded_files: list[Any] | None) -> list[UploadedFilePayload]:
    """Convert Streamlit uploaded files into plain payloads for the ingestion helper."""
    payloads: list[UploadedFilePayload] = []
    for uploaded_file in uploaded_files or []:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            continue
        payloads.append(
            UploadedFilePayload(
                file_name=uploaded_file.name,
                content=file_bytes,
            )
        )
    return payloads


def format_optional_year(value: object) -> str:
    if value is None or pd.isna(value) or value in {"", 0}:
        return "N/A"
    return str(int(value))


def build_uploaded_files_summary_df(uploaded_files: list[Any] | None) -> pd.DataFrame:
    """Build a lightweight dataframe describing uploaded files."""
    rows: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files or []:
        size_bytes = int(getattr(uploaded_file, "size", 0) or 0)
        rows.append(
            {
                "file_name": uploaded_file.name,
                "type": Path(uploaded_file.name).suffix.lower().lstrip(".") or "unknown",
                "size_kb": round(size_bytes / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_knowledge_views():
    return get_knowledge_artifacts(chunks_path=CHUNKS_PATH)


def render_retrieval_results(retrieval_results: list[dict], key_prefix: str = "retrieval") -> None:
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
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(
                "Source",
                "Uploaded" if row.get("document_source") == "uploaded" else "Built-in",
            )
            col2.metric("Company", row.get("company", "N/A"))
            col3.metric("Fiscal year", format_optional_year(row.get("fiscal_year")))
            col4.metric("Final score", f"{row.get('final_score', row.get('score', 0.0)):.4f}")
            col5.metric("Knowledge", f"{row.get('knowledge_score', 0.0):.4f}")

            if row.get("file_name"):
                st.caption(f"File: {row.get('file_name')}")

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
                key=f"{key_prefix}_chunk_text_{idx}",
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


def render_session_documents_sidebar() -> list[Any]:
    """Render the upload controls for session-scoped retrieval documents."""
    ensure_upload_widget_state()

    st.subheader("Session Documents")
    st.caption(
        "Add PDFs, text files, or markdown notes. They are merged into retrieval only for "
        "this Streamlit session."
    )

    if st.session_state.get(UPLOAD_CLEAR_NOTICE_STATE, False):
        st.success("Session documents cleared.")
        st.session_state[UPLOAD_CLEAR_NOTICE_STATE] = False

    uploaded_files = st.file_uploader(
        "Add documents to retrieval",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="These files are added to the Ask tab retrieval for the current session.",
        key=f"session_documents_{st.session_state[UPLOAD_WIDGET_KEY_STATE]}",
    )

    uploaded_summary_df = build_uploaded_files_summary_df(uploaded_files)
    active_count = len(uploaded_summary_df)

    metric1, metric2 = st.columns(2)
    metric1.metric("Active docs", active_count)
    metric2.metric("Formats", "pdf/txt/md")

    if active_count:
        st.info("Session corpus active for the Ask tab.")
        st.dataframe(uploaded_summary_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No session documents loaded.")

    clear_button = st.button(
        "Clear session documents",
        use_container_width=True,
        disabled=active_count == 0,
    )
    if clear_button:
        st.session_state[UPLOAD_WIDGET_KEY_STATE] += 1
        st.session_state[UPLOAD_CLEAR_NOTICE_STATE] = True
        st.rerun()

    st.caption("Explorer tabs keep using the precomputed base corpus views.")
    return uploaded_files or []


def run_assistant_query(
    question: str,
    assistant_mode: str,
    llm_model: str,
    embedding_model: str,
    top_k: int,
    retrieval_mode: str,
    company_filter: str | None,
    year_filter: int | None,
    uploaded_files: list[Any],
) -> dict[str, Any]:
    """Run one assistant turn and return the full result payload."""
    uploaded_payloads = build_uploaded_file_payloads(uploaded_files)
    runtime_chunks_path = CHUNKS_PATH
    runtime_chunk_embeddings_path = CHUNK_EMBEDDINGS_PATH
    persistent_index_mode = "auto"
    uploaded_runtime_corpus = None

    if uploaded_payloads:
        uploaded_runtime_corpus = prepare_uploaded_runtime_corpus(
            base_chunks_path=CHUNKS_PATH,
            base_chunk_embeddings_path=CHUNK_EMBEDDINGS_PATH,
            uploaded_files=uploaded_payloads,
            embedding_model=embedding_model,
            output_dir=UPLOADED_RUNTIME_DIR,
        )
        runtime_chunks_path = uploaded_runtime_corpus.chunks_path
        runtime_chunk_embeddings_path = uploaded_runtime_corpus.chunk_embeddings_path
        persistent_index_mode = "source"

    if assistant_mode == "Agent Analyst":
        result = run_financial_analyst_agent(
            question=question.strip(),
            llm_model=llm_model,
            llm_cache_path=LLM_CACHE_PATH,
            chunks_path=runtime_chunks_path,
            chunk_embeddings_path=runtime_chunk_embeddings_path,
            query_embeddings_path=QUERY_EMBEDDINGS_PATH,
            knowledge_chunks_path=CHUNKS_PATH,
            embedding_model=embedding_model,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            company_filter=company_filter,
            fiscal_year_filter=year_filter,
            persistent_index_mode=persistent_index_mode,
            uploaded_documents_available=bool(uploaded_payloads),
        )
    else:
        result = generate_rag_answer(
            question=question.strip(),
            chunks_path=runtime_chunks_path,
            chunk_embeddings_path=runtime_chunk_embeddings_path,
            query_embeddings_path=QUERY_EMBEDDINGS_PATH,
            embedding_model=embedding_model,
            llm_model=llm_model,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            llm_cache_path=LLM_CACHE_PATH,
            company_filter=company_filter,
            fiscal_year_filter=year_filter,
            persistent_index_mode=persistent_index_mode,
        )

    if uploaded_runtime_corpus is not None:
        result["uploaded_documents"] = uploaded_runtime_corpus.documents

    output_path = OUTPUT_DIR / f"latest_{result.get('mode', 'assistant')}_result.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def render_assistant_result_details(
    result: dict[str, Any],
    show_retrieval: bool,
    key_prefix: str,
) -> None:
    """Render citations and technical details for one assistant answer."""
    with st.expander("Sources et details", expanded=False):
        st.caption(f"Mode: {result.get('mode', 'assistant')}")
        st.caption(f"Retrieval mode: {result.get('retrieval_mode', 'unknown')}")

        if result.get("citations"):
            st.markdown("**Citations**")
            for citation in result["citations"]:
                st.markdown(f"- `{format_citation(citation)}`")
        else:
            st.caption("Aucune citation validee retournee.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top-k", result.get("top_k", "N/A"))
        col2.metric("Cache LLM", "Yes" if result.get("llm_from_cache") else "No")
        col3.metric("Company", result.get("company_filter") or "None")
        col4.metric("Year", result.get("fiscal_year_filter") or "None")

        uploaded_documents = result.get("uploaded_documents", [])
        if uploaded_documents:
            st.markdown("**Documents de session utilises**")
            st.dataframe(
                pd.DataFrame(uploaded_documents),
                use_container_width=True,
                hide_index=True,
            )

        render_safety_flags(result.get("safety_flags", []))
        render_tool_trace(result.get("tool_calls", []))

        if show_retrieval:
            render_retrieval_results(
                result.get("retrieval_results", []),
                key_prefix=f"{key_prefix}_retrieval",
            )

        st.download_button(
            label="Download answer as JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"{result.get('mode', 'assistant')}_result.json",
            mime="application/json",
            key=f"{key_prefix}_download_json",
        )


def main() -> None:
    ensure_upload_widget_state()
    ensure_chat_state()

    st.set_page_config(
        page_title="Financial RAG Demo",
        page_icon="F",
        layout="wide",
    )

    st.title("Financial Intelligence Demo")
    st.write(
        "Ask questions over the 10-K corpus, add your own documents for the current session, "
        "inspect extracted knowledge, and review competitor evidence with source pages."
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

        st.markdown("---")
        uploaded_files = render_session_documents_sidebar()

    company_filter = company_label_to_value(company_label)
    year_filter = year_label_to_value(year_label)

    ask_tab, knowledge_tab, competitors_tab = st.tabs(
        ["Ask", "Knowledge Explorer", "Competitors"]
    )

    with ask_tab:
        header_col, action_col = st.columns([5, 1])
        with header_col:
            st.caption(
                "Conversation grounded on the local corpus"
                + (" plus session documents." if uploaded_files else ".")
            )
            filter_parts = []
            if company_filter:
                filter_parts.append(f"company={company_filter}")
            if year_filter:
                filter_parts.append(f"fiscal_year={year_filter}")
            if filter_parts:
                st.caption("Active retrieval filters: " + " | ".join(filter_parts))
        with action_col:
            clear_chat = st.button("New chat", use_container_width=True)
            if clear_chat:
                st.session_state[CHAT_HISTORY_STATE] = []
                st.session_state[CHAT_CLEAR_NOTICE_STATE] = True
                st.rerun()

        if st.session_state.get(CHAT_CLEAR_NOTICE_STATE, False):
            st.success("Conversation cleared.")
            st.session_state[CHAT_CLEAR_NOTICE_STATE] = False

        if uploaded_files:
            st.caption(
                f"{len(uploaded_files)} session document(s) will also be searched alongside "
                "the base corpus."
            )

        if not st.session_state[CHAT_HISTORY_STATE]:
            st.info("Pose une question en bas de l'ecran pour commencer la conversation.")

        for message_index, message in enumerate(st.session_state[CHAT_HISTORY_STATE]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "result" in message:
                    render_assistant_result_details(
                        result=message["result"],
                        show_retrieval=show_retrieval,
                        key_prefix=f"chat_{message_index}",
                    )

        prompt = st.chat_input(
            "Pose une question sur les documents...",
        )

        if prompt:
            clean_prompt = prompt.strip()
            if not clean_prompt:
                st.stop()

            user_message = {"role": "user", "content": clean_prompt}
            st.session_state[CHAT_HISTORY_STATE].append(user_message)

            with st.chat_message("user"):
                st.markdown(clean_prompt)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Recherche et generation en cours..."):
                        result = run_assistant_query(
                            question=clean_prompt,
                            assistant_mode=assistant_mode,
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                            top_k=top_k,
                            retrieval_mode=retrieval_mode,
                            company_filter=company_filter,
                            year_filter=year_filter,
                            uploaded_files=uploaded_files,
                        )

                    answer_text = result.get("answer") or "No answer returned."
                    st.markdown(answer_text)
                    render_assistant_result_details(
                        result=result,
                        show_retrieval=show_retrieval,
                        key_prefix=f"chat_{len(st.session_state[CHAT_HISTORY_STATE])}",
                    )

                st.session_state[CHAT_HISTORY_STATE].append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "result": result,
                    }
                )
            except Exception as exc:
                with st.chat_message("assistant"):
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
