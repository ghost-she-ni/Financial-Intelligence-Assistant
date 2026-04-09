from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.agent.tools import (
    AgentRuntimeConfig,
    build_local_tool_registry,
    build_local_tool_specs,
    parse_tool_arguments,
)
from src.common.guardrails import build_safety_flags
from src.common.grounded_qa import build_context_block, normalize_citations
from src.common.prompting import build_agent_system_prompt, build_agent_user_prompt
from src.generation.rag_answer import DEFAULT_LLM_MODEL
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def summarize_tool_output(tool_name: str, tool_result: dict[str, Any]) -> str:
    """Create a compact human-readable trace summary for Streamlit and JSON logs."""
    if tool_name == "search_financial_corpus":
        return (
            f"Retrieved {tool_result.get('n_results', 0)} chunks "
            f"with mode={tool_result.get('retrieval_mode', 'unknown')}."
        )
    if tool_name == "lookup_knowledge_graph":
        return (
            f"Found {tool_result.get('entity_rows', 0)} entity rows and "
            f"{tool_result.get('triplet_rows', 0)} triplet rows."
        )
    if tool_name == "get_competitor_evidence":
        return (
            f"Returned {len(tool_result.get('summary_rows', []))} competitor summaries and "
            f"{len(tool_result.get('evidence_rows', []))} evidence rows."
        )
    return "Tool executed."


def parse_final_agent_json(response_text: str) -> dict[str, Any]:
    """Parse the agent's final JSON answer, with a defensive fallback."""
    clean_text = response_text.strip()
    if clean_text == "":
        return {
            "answer": "I cannot provide a grounded answer from the available local evidence.",
            "citations": [],
            "safety_flags": ["empty_model_output"],
        }

    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError:
        return {
            "answer": clean_text,
            "citations": [],
            "safety_flags": ["non_json_agent_output"],
        }

    if not isinstance(parsed, dict):
        return {
            "answer": clean_text,
            "citations": [],
            "safety_flags": ["non_object_agent_output"],
        }

    return parsed


def run_financial_analyst_agent(
    question: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_cache_path: Path | str = PROJECT_ROOT / "data" / "cache" / "llm_responses.jsonl",
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet",
    chunk_embeddings_path: Path = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet",
    query_embeddings_path: Path = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    retrieval_mode: str = "improved",
    top_k: int = 5,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    knowledge_chunks_path: Path | None = None,
    persistent_index_mode: str = "auto",
    uploaded_documents_available: bool = False,
    max_tool_calls: int = 3,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Run a local tool-using analyst agent with bounded function calling."""
    runtime_config = AgentRuntimeConfig(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        query_embeddings_path=query_embeddings_path,
        knowledge_chunks_path=knowledge_chunks_path,
        embedding_model=embedding_model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        persistent_index_mode=persistent_index_mode,
    )
    tool_specs = build_local_tool_specs()
    tool_registry = build_local_tool_registry(runtime_config)

    if llm_client is None:
        llm_client = LLMClient(
            model=llm_model,
            temperature=0.0,
            max_output_tokens=900,
            cache_path=llm_cache_path,
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_agent_system_prompt(max_tool_calls=max_tool_calls)},
        {
            "role": "user",
            "content": build_agent_user_prompt(
                question=question,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
                uploaded_documents_available=uploaded_documents_available,
            ),
        },
    ]

    tool_call_trace: list[dict[str, Any]] = []
    retrieval_results_df = pd.DataFrame()
    retrieved_context = ""
    last_llm_result: dict[str, Any] | None = None

    for turn_index in range(max_tool_calls + 1):
        last_llm_result = llm_client.chat(
            messages=messages,
            tools=tool_specs,
            tool_choice="auto",
            task_name="financial_analyst_agent",
            metadata={
                "question": question,
                "turn_index": turn_index + 1,
                "company_filter": company_filter,
                "fiscal_year_filter": fiscal_year_filter,
                "retrieval_mode": retrieval_mode,
            },
        )
        messages.append(last_llm_result["assistant_message"])

        tool_calls = last_llm_result.get("tool_calls", []) or []
        if tool_calls:
            remaining_calls = max_tool_calls - len(tool_call_trace)
            if remaining_calls <= 0:
                break

            for tool_call in tool_calls[:remaining_calls]:
                tool_name = tool_call["function"]["name"]
                tool_args = parse_tool_arguments(tool_call["function"]["arguments"])

                try:
                    if tool_name not in tool_registry:
                        raise ValueError(f"Unknown local tool: {tool_name}")

                    tool_result = tool_registry[tool_name](**tool_args)
                    tool_trace_entry = {
                        "step": len(tool_call_trace) + 1,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "status": "success",
                        "summary": summarize_tool_output(tool_name, tool_result),
                        "result": tool_result,
                    }

                    if tool_name == "search_financial_corpus":
                        retrieval_results_df = pd.DataFrame(tool_result.get("retrieval_results", []))
                        retrieved_context = str(tool_result.get("retrieved_context", "") or "")

                except Exception as exc:
                    tool_result = {"error": str(exc)}
                    tool_trace_entry = {
                        "step": len(tool_call_trace) + 1,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "status": "error",
                        "summary": f"Tool failed: {exc}",
                        "result": tool_result,
                    }

                tool_call_trace.append(tool_trace_entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            continue

        parsed_final = parse_final_agent_json(last_llm_result.get("response_text", ""))
        answer = str(parsed_final.get("answer", "") or "").strip()
        citations = normalize_citations(
            citations=parsed_final.get("citations", []),
            retrieval_results_df=retrieval_results_df,
        )
        model_flags = parsed_final.get("safety_flags", [])
        if not isinstance(model_flags, list):
            model_flags = []
        safety_flags = sorted(
            {
                *[str(flag) for flag in model_flags if str(flag).strip()],
                *build_safety_flags(
                    question=question,
                    answer=answer,
                    citations=citations,
                    tool_calls=tool_call_trace,
                ),
            }
        )

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "company_filter": company_filter,
            "fiscal_year_filter": fiscal_year_filter,
            "retrieval_results": (
                json.loads(retrieval_results_df.to_json(orient="records", force_ascii=False))
                if not retrieval_results_df.empty
                else []
            ),
            "retrieved_context": retrieved_context
            or (build_context_block(retrieval_results_df) if not retrieval_results_df.empty else ""),
            "llm_request_hash": last_llm_result.get("request_hash"),
            "llm_from_cache": bool(last_llm_result.get("from_cache", False)),
            "llm_created_at": last_llm_result.get("created_at"),
            "mode": "agent_analyst",
            "tool_calls": tool_call_trace,
            "safety_flags": safety_flags,
        }

    fallback_answer = (
        "I cannot provide a grounded answer from the available local evidence after the "
        "allowed tool calls."
    )
    fallback_flags = build_safety_flags(
        question=question,
        answer=fallback_answer,
        citations=[],
        tool_calls=tool_call_trace,
    )
    fallback_flags = sorted({*fallback_flags, "max_tool_calls_reached"})
    final_context = retrieved_context or (
        build_context_block(retrieval_results_df) if not retrieval_results_df.empty else ""
    )

    return {
        "question": question,
        "answer": fallback_answer,
        "citations": [],
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "company_filter": company_filter,
        "fiscal_year_filter": fiscal_year_filter,
        "retrieval_results": (
            json.loads(retrieval_results_df.to_json(orient="records", force_ascii=False))
            if not retrieval_results_df.empty
            else []
        ),
        "retrieved_context": final_context,
        "llm_request_hash": last_llm_result.get("request_hash") if last_llm_result else None,
        "llm_from_cache": bool(last_llm_result.get("from_cache", False)) if last_llm_result else False,
        "llm_created_at": last_llm_result.get("created_at") if last_llm_result else None,
        "mode": "agent_analyst",
        "tool_calls": tool_call_trace,
        "safety_flags": fallback_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local financial analyst agent.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="improved",
        choices=["classical_ml", "naive", "improved"],
        help="Retrieval mode used by the search tool.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks for tool retrieval.")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="Agent model.")
    parser.add_argument("--company_filter", type=str, default=None, help="Optional company filter.")
    parser.add_argument(
        "--fiscal_year_filter",
        type=int,
        default=None,
        help="Optional fiscal year filter.",
    )
    args = parser.parse_args()

    result = run_financial_analyst_agent(
        question=args.question,
        llm_model=args.llm_model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        company_filter=args.company_filter,
        fiscal_year_filter=args.fiscal_year_filter,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
