from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.retrieval.retrieve import extract_query_filters, retrieve_top_k_with_mode

TOP_K_VALUES = [3, 5]
RETRIEVAL_MODES = ["classical_ml", "naive", "improved"]


def load_queries(queries_path: Path) -> pd.DataFrame:
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    if queries_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(queries_path)
    elif queries_path.suffix.lower() == ".csv":
        df = pd.read_csv(queries_path)
    else:
        raise ValueError("Queries file must be .parquet or .csv")

    if "query_text" not in df.columns:
        raise ValueError("Queries file must contain a 'query_text' column")

    df = df.copy()
    df["query_text"] = df["query_text"].fillna("").astype(str).str.strip()
    df = df[df["query_text"] != ""].reset_index(drop=True)

    return df


def build_summary_row(
    query_text: str,
    top_k: int,
    retrieval_mode: str,
    results_df: pd.DataFrame,
) -> dict:
    filters = extract_query_filters(query_text)

    expected_company = filters["company"]
    expected_year = filters["fiscal_year"]

    top1_company = results_df.iloc[0]["company"] if not results_df.empty else None
    top1_year = int(results_df.iloc[0]["fiscal_year"]) if not results_df.empty else None
    top1_score = float(results_df.iloc[0]["score"]) if not results_df.empty else None
    top1_final_score = float(results_df.iloc[0]["final_score"]) if not results_df.empty else None

    company_match = None
    if expected_company is not None and top1_company is not None:
        company_match = (expected_company == top1_company)

    year_match = None
    if expected_year is not None and top1_year is not None:
        year_match = (expected_year == top1_year)

    return {
        "query_text": query_text,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "expected_company": expected_company,
        "expected_year": expected_year,
        "top1_company": top1_company,
        "top1_year": top1_year,
        "top1_score": top1_score,
        "top1_final_score": top1_final_score,
        "top1_company_match": company_match,
        "top1_year_match": year_match,
        "n_results": len(results_df),
        "manual_relevance": "",   # to fill manually: good / partial / poor
        "notes": "",              # to fill manually
    }


def main() -> None:
    queries_path = PROJECT_ROOT / "data" / "queries" / "queries.parquet"
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
    chunk_embeddings_path = PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.parquet"
    query_embeddings_path = PROJECT_ROOT / "data" / "embeddings" / "query_embeddings.parquet"

    output_dir = PROJECT_ROOT / "outputs" / "retrieval_tests"
    detailed_dir = output_dir / "detailed_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)

    queries_df = load_queries(queries_path)

    summary_rows = []

    for query_idx, query_text in enumerate(queries_df["query_text"].tolist(), start=1):
        for retrieval_mode in RETRIEVAL_MODES:
            for top_k in TOP_K_VALUES:
                print("=" * 100)
                print(f"QUERY {query_idx} | mode={retrieval_mode} | top_k={top_k}")
                print(query_text)
                print("=" * 100)

                results_df = retrieve_top_k_with_mode(
                    chunks_path=chunks_path,
                    chunk_embeddings_path=chunk_embeddings_path,
                    query_embeddings_path=query_embeddings_path,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    query_text=query_text,
                    enable_metadata_filters=True,
                    enable_noise_filter=True,
                    enable_lexical_rerank=True,
                )

                print(
                    results_df[
                        [
                            "score",
                            "lexical_score",
                            "section_score",
                            "final_score",
                            "chunk_id",
                            "company",
                            "fiscal_year",
                            "page_start",
                            "page_end",
                        ]
                    ].to_string(index=False)
                )
                print()

                file_name = f"query_{query_idx}_{retrieval_mode}_topk_{top_k}.csv"
                results_df.to_csv(detailed_dir / file_name, index=False)

                summary_rows.append(
                    build_summary_row(
                        query_text=query_text,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        results_df=results_df,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "retrieval_summary.csv", index=False)

    print("\nRetrieval tests completed.")
    print(f"Summary saved to: {(output_dir / 'retrieval_summary.csv').resolve()}")
    print(f"Detailed results saved to: {detailed_dir.resolve()}")


if __name__ == "__main__":
    main()
