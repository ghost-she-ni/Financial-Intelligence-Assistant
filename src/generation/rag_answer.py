from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.common.guardrails import build_safety_flags
from src.common.grounded_qa import (
    build_context_block,
    build_system_prompt,
    build_user_prompt,
    normalize_citations,
)
from src.embeddings.cache import (
    append_records_to_cache,
    build_query_embedding_record,
    get_missing_query_rows,
    load_query_embedding_cache,
    resolve_text_hash,
    save_cache,
)
from src.llm.client import LLMClient
from src.retrieval.retrieve import (
    CLASSICAL_ML_RETRIEVAL_MODE,
    IMPROVED_RETRIEVAL_MODE,
    NAIVE_RETRIEVAL_MODE,
    normalize_retrieval_mode,
    retrieve_top_k_with_mode,
)

# Configure module logger
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def load_sentence_transformer(model_name: str, device: str | None = None) -> SentenceTransformer:
    """Load a sentence-transformers model."""
    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def ensure_query_embeddings_cached(
    query_texts: list[str],
    query_embeddings_path: Path,
    embedding_model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    verbose: bool = False,
) -> None:
    """Ensure that all query embeddings exist in the cache."""
    if not query_texts:
        return

    queries_df = pd.DataFrame({"query_text": query_texts}).drop_duplicates().reset_index(drop=True)

    if verbose:
        logger.info(f"Checking query embeddings for {len(queries_df)} unique queries.")

    cache_df = load_query_embedding_cache(query_embeddings_path)

    missing_queries_df = get_missing_query_rows(
        queries_df=queries_df,
        cache_df=cache_df,
        embedding_model=embedding_model_name,
        query_text_column="query_text",
        query_id_column="query_id",
    )

    if missing_queries_df.empty:
        if verbose:
            logger.info("All query embeddings already cached.")
        return

    if verbose:
        logger.info(f"Computing embeddings for {len(missing_queries_df)} missing queries.")

    model = load_sentence_transformer(embedding_model_name, device=device)

    new_records: list[dict[str, Any]] = []
    texts = missing_queries_df["query_text"].fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for row, embedding in zip(missing_queries_df.itertuples(index=False), embeddings):
        text_hash = resolve_text_hash(
            text_hash=getattr(row, "text_hash", None),
            text=row.query_text,
        )

        new_records.append(
            build_query_embedding_record(
                query_id=row.query_id,
                query_text=row.query_text,
                embedding_model=embedding_model_name,
                text_hash=text_hash,
                embedding=embedding.tolist(),
            )
        )

    updated_cache_df = append_records_to_cache(
        cache_df=cache_df,
        new_records=new_records,
        key_columns=["query_id", "embedding_model"],
    )
    save_cache(updated_cache_df, query_embeddings_path)

    if verbose:
        logger.info(f"Saved {len(new_records)} new query embeddings to cache.")


def generate_rag_answer(
    question: str,
    chunks_path: Path,
    chunk_embeddings_path: Path,
    query_embeddings_path: Path,
    embedding_model: str,
    llm_model: str,
    top_k: int = 5,
    llm_cache_path: Path | str = "data/cache/llm_responses.jsonl",
    device: str | None = None,
    verbose: bool = False,
    skip_llm: bool = False,
    retrieval_mode: str = IMPROVED_RETRIEVAL_MODE,
    enable_lexical_rerank: bool = True,
    system_prompt_path: Path | str | None = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    enable_metadata_filters: bool = True,
) -> dict[str, Any]:
    """
    Full RAG generation pipeline.

    Args:
        ...
        system_prompt_path: Optional path to a file containing the system prompt.
                           If provided, the file content is used; otherwise a default prompt is used.
    """
    if verbose:
        logger.info("Starting RAG generation for question: %s", question[:50])

    retrieval_mode = normalize_retrieval_mode(retrieval_mode)

    # Ensure query embeddings are cached
    ensure_query_embeddings_cached(
        query_texts=[question],
        query_embeddings_path=query_embeddings_path,
        embedding_model_name=embedding_model,
        device=device,
        verbose=verbose,
    )

    # Retrieve top-k chunks
    if verbose:
        logger.info("Retrieving top-%d chunks...", top_k)

    retrieval_results_df = retrieve_top_k_with_mode(
        chunks_path=chunks_path,
        chunk_embeddings_path=chunk_embeddings_path,
        query_embeddings_path=query_embeddings_path,
        embedding_model=embedding_model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        query_text=question,
        enable_noise_filter=True,
        enable_lexical_rerank=enable_lexical_rerank,
        enable_metadata_filters=enable_metadata_filters,
        company_filter=company_filter,
        fiscal_year_filter=fiscal_year_filter,
        verbose=verbose,
    )

    if verbose:
        logger.info(f"Retrieved {len(retrieval_results_df)} chunks.")

    retrieved_context = build_context_block(retrieval_results_df) if not retrieval_results_df.empty else ""
    serialized_retrieval_results = json.loads(
        retrieval_results_df.to_json(orient="records", force_ascii=False)
    ) if not retrieval_results_df.empty else []

    if skip_llm:
        return {
            "question": question,
            "answer": "",
            "citations": [],
            "top_k": top_k,
            "embedding_model": embedding_model,
            "retrieval_mode": retrieval_mode,
            "llm_model": llm_model,
            "company_filter": company_filter,
            "fiscal_year_filter": fiscal_year_filter,
            "retrieval_results": serialized_retrieval_results,
            "retrieved_context": retrieved_context,
            "llm_request_hash": None,
            "llm_from_cache": False,
            "llm_created_at": None,
            "mode": "direct_rag",
            "tool_calls": [],
            "safety_flags": build_safety_flags(
                question=question,
                answer="",
                citations=[],
                tool_calls=[],
            ),
            "skip_llm": True,
        }

    # Build system prompt (either from file or default)
    if system_prompt_path is not None:
        prompt_file = Path(system_prompt_path)
        try:
            system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            if not system_prompt:
                logger.warning("System prompt file is empty. Using default prompt.")
                system_prompt = build_system_prompt()
            else:
                if verbose:
                    logger.info(f"Loaded system prompt from {prompt_file}")
        except Exception as e:
            logger.warning(f"Failed to load system prompt from {prompt_file}: {e}. Using default.")
            system_prompt = build_system_prompt()
    else:
        system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(question, retrieval_results_df)

    if verbose:
        logger.info("Calling LLM...")

    llm_client = LLMClient(
        model=llm_model,
        temperature=0.0,
        max_output_tokens=800,
        cache_path=llm_cache_path,
    )

    try:
        llm_result = llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_name="rag_answer_generation",
            metadata={
                "question": question,
                "top_k": top_k,
                "embedding_model": embedding_model,
                "retrieval_mode": retrieval_mode,
                "company_filter": company_filter,
                "fiscal_year_filter": fiscal_year_filter,
            },
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise

    parsed_json = llm_result.get("parsed_json", {})
    answer = str(parsed_json.get("answer", "")).strip()
    citations = normalize_citations(
        citations=parsed_json.get("citations", []),
        retrieval_results_df=retrieval_results_df,
    )

    result = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "company_filter": company_filter,
        "fiscal_year_filter": fiscal_year_filter,
        "retrieval_results": serialized_retrieval_results,
        "retrieved_context": retrieved_context,
        "llm_request_hash": llm_result["request_hash"],
        "llm_from_cache": llm_result["from_cache"],
        "llm_created_at": llm_result["created_at"],
        "mode": "direct_rag",
        "tool_calls": [],
        "safety_flags": build_safety_flags(
            question=question,
            answer=answer,
            citations=citations,
            tool_calls=[],
        ),
    }

    if verbose:
        logger.info("RAG generation completed. Answer length: %d chars", len(answer))

    return result


def print_rag_result(result: dict[str, Any]) -> None:
    """Print a readable RAG answer using logging (INFO level)."""
    logger.info("RAG answer completed.")
    logger.info(f"Question: {result['question']}")
    logger.info("Answer:")
    logger.info(result["answer"] or "[EMPTY ANSWER]")
    logger.info("Citations:")
    if result["citations"]:
        for citation in result["citations"]:
            logger.info(f"- {citation['doc_id']} | page {citation['page']}")
    else:
        logger.info("- No validated citations returned.")
    logger.info(f"LLM from cache: {result['llm_from_cache']}")
    logger.info(f"LLM request hash: {result['llm_request_hash']}")


def save_rag_result(result: dict[str, Any], output_path: Path) -> None:
    """Save the RAG result to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved result to {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RAG answer from retrieved chunks.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/processed/chunks.parquet",
        help="Path to chunks file.",
    )
    parser.add_argument(
        "--chunk_embeddings_path",
        type=str,
        default="data/embeddings/chunk_embeddings.parquet",
        help="Path to chunk embeddings cache.",
    )
    parser.add_argument(
        "--query_embeddings_path",
        type=str,
        default="data/embeddings/query_embeddings.parquet",
        help="Path to query embeddings cache.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM model name.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved chunks used as context.",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default=IMPROVED_RETRIEVAL_MODE,
        choices=[
            CLASSICAL_ML_RETRIEVAL_MODE,
            NAIVE_RETRIEVAL_MODE,
            IMPROVED_RETRIEVAL_MODE,
        ],
        help="Retrieval mode to use for the answer: classical ML baseline, naive dense baseline, or improved retriever.",
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default="data/cache/llm_responses.jsonl",
        help="Path to the LLM JSONL cache.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for embedding model, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Optional path to a file containing the system prompt.",
    )
    parser.add_argument(
        "--company_filter",
        type=str,
        default=None,
        help="Optional explicit company filter for retrieval.",
    )
    parser.add_argument(
        "--fiscal_year_filter",
        type=int,
        default=None,
        help="Optional explicit fiscal year filter for retrieval.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run retrieval only, skip LLM call.",
    )
    parser.add_argument(
        "--disable_lexical_rerank",
        action="store_true",
        help="Disable lexical reranking in retrieval.",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        result = generate_rag_answer(
            question=args.question,
            chunks_path=Path(args.chunks_path),
            chunk_embeddings_path=Path(args.chunk_embeddings_path),
            query_embeddings_path=Path(args.query_embeddings_path),
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            llm_cache_path=args.llm_cache_path,
            device=args.device,
            verbose=args.verbose,
            skip_llm=args.dry_run,
            enable_lexical_rerank=not args.disable_lexical_rerank,
            system_prompt_path=args.system_prompt_path,
            company_filter=args.company_filter,
            fiscal_year_filter=args.fiscal_year_filter,
        )
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        sys.exit(1)

    print_rag_result(result)

    if args.output_path:
        save_rag_result(result, Path(args.output_path))


if __name__ == "__main__":
    main()
