from __future__ import annotations

import numpy as np
import pandas as pd


class SimpleVectorIndex:
    """
    Simple in-memory vector index using cosine similarity.

    The index stores normalized vectors and a corresponding metadata DataFrame.
    Search returns the top‑k most similar rows with similarity scores.

    Assumption:
    Vectors should be normalized for optimal performance, but the index will
    re‑normalize them again during initialization and for each query.
    """

    def __init__(self, vectors: np.ndarray, metadata_df: pd.DataFrame) -> None:
        """
        Initialize the vector index.

        Args:
            vectors: 2D numpy array of shape (n_vectors, embedding_dim).
            metadata_df: DataFrame with one row per vector, containing metadata
                         (e.g., chunk_id, doc_id, page_start, etc.).

        Raises:
            ValueError: If vectors and metadata_df have different lengths,
                        or if vectors is not 2D.
        """
        if len(vectors) != len(metadata_df):
            raise ValueError(
                f"vectors length ({len(vectors)}) does not match metadata_df length ({len(metadata_df)})"
            )

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D array, got {vectors.ndim}D")

        # Ensure float32 for efficiency
        self.vectors = self._normalize(vectors.astype(np.float32))
        self.metadata_df = metadata_df.reset_index(drop=True).copy()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length (L2 norm) in place.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self.vectors)

    def __repr__(self) -> str:
        """Return a string representation of the index."""
        return f"SimpleVectorIndex(n_vectors={len(self)})"

    def search(self, query_vector: np.ndarray | list[float], top_k: int = 5) -> pd.DataFrame:
        """
        Return the top‑k most similar rows using cosine similarity.

        Args:
            query_vector: 1D numpy array or list of floats representing the query embedding.
            top_k: Number of top results to return.

        Returns:
            DataFrame with the top‑k rows from metadata_df, with an extra column 'score'
            containing the cosine similarity (higher is better). Returns an empty DataFrame
            if the index is empty.

        Raises:
            ValueError: If the query vector dimension does not match the index vectors.
        """
        if len(self) == 0:
            # Empty index
            return pd.DataFrame()

        # Convert list to numpy if needed
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        elif isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32)
        else:
            raise TypeError(f"query_vector must be list or numpy array, got {type(query_vector)}")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim != 2:
            raise ValueError(f"query_vector must be 1D or 2D, got {query_vector.ndim}D")

        if query_vector.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.vectors.shape[1]})"
            )

        # Normalize query vector (important for cosine similarity)
        query_vector = self._normalize(query_vector)

        # Compute dot products (cosine similarity because both are normalized)
        scores = np.dot(self.vectors, query_vector[0])

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]

        results = self.metadata_df.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        results = results.sort_values("score", ascending=False).reset_index(drop=True)

        return results