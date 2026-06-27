"""Maximal Marginal Relevance (MMR) re-ranking.

Pure, dependency-light numeric helper shared by the embedding service to support
the ``mmr`` search type across any vector-store backend.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def maximal_marginal_relevance(
    query: list[float],
    candidates: list[list[float]],
    *,
    k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Return indices of the selected candidates, MMR-ordered.

    Args:
        query: the query embedding.
        candidates: candidate document embeddings.
        k: number of results to select.
        lambda_mult: trade-off between relevance (1.0) and diversity (0.0).
    """
    if not candidates or k <= 0:
        return []

    query_vec = _normalize(np.array([query], dtype=np.float64))[0]
    docs = _normalize(np.array(candidates, dtype=np.float64))
    relevance = docs @ query_vec

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    target = min(k, len(candidates))

    while len(selected) < target:
        best_index = -1
        best_score = -np.inf
        for index in remaining:
            if not selected:
                score = relevance[index]
            else:
                redundancy = max(float(docs[index] @ docs[s]) for s in selected)
                score = lambda_mult * relevance[index] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(best_index)
        remaining.discard(best_index)

    return selected
