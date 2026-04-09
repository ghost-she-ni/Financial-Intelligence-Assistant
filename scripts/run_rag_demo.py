from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.generation.rag_answer import (
    ensure_query_embeddings_cached,
    generate_rag_answer,
)

QUESTIONS = [
    "What are Adobe's main risk factors in 2024?",
    "How did Lockheed Martin describe competition in 2023?",
    "What financial metrics are highlighted in Pfizer's 2024 annual report?",
    "Did Adobe mention AI-related opportunities or risks in 2024?",
    "What are the main business segments discussed by Pfizer in 2023?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small RAG demo in classical_ml, naive, or improved mode."
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="improved",
        choices=["classical_ml", "naive", "improved"],
        help="Retrieval mode to use in the demo.",
    )
    args = parser.parse_args()

    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
    chunk_embeddings_path = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
    query_embeddings_path = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"
    output_dir = PROJECT_ROOT / "outputs" / "rag_demo" / args.retrieval_mode

    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "gpt-4o-mini"

    ensure_query_embeddings_cached(
        query_texts=QUESTIONS,
        query_embeddings_path=query_embeddings_path,
        embedding_model_name=embedding_model,
    )

    summary_rows = []

    for i, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 100)
        print(f"RAG DEMO QUESTION {i}")
        print(question)
        print("=" * 100)

        result = generate_rag_answer(
            question=question,
            chunks_path=chunks_path,
            chunk_embeddings_path=chunk_embeddings_path,
            query_embeddings_path=query_embeddings_path,
            embedding_model=embedding_model,
            llm_model=llm_model,
            top_k=5,
            retrieval_mode=args.retrieval_mode,
        )

        print("\nAnswer:")
        print(result["answer"])
        print("\nCitations:")
        if result["citations"]:
            for citation in result["citations"]:
                print(f"- {citation['doc_id']} | page {citation['page']}")
        else:
            print("- No validated citations returned.")

        output_path = output_dir / f"rag_answer_{i}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_rows.append(
            {
                "question": question,
                "retrieval_mode": args.retrieval_mode,
                "answer_preview": result["answer"][:200],
                "n_citations": len(result["citations"]),
                "llm_from_cache": result["llm_from_cache"],
                "output_file": output_path.name,
            }
        )

    summary_path = output_dir / "rag_demo_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRAG demo completed.")
    print(f"Saved results in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
