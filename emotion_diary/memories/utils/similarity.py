from typing import Iterable, List, Sequence, Tuple

import numpy as np


def to_float32_array(
    vectors: Iterable
) -> np.ndarray:
    """
    Convert vectors into optimized
    float32 NumPy array.
    """

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def normalize_vectors(
    vectors: np.ndarray
) -> np.ndarray:
    """
    Normalize vectors for cosine similarity.

    Converts vectors to unit vectors.
    """

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    norms = np.clip(
        norms,
        a_min=1e-12,
        a_max=None
    )

    return vectors / norms


def compute_similarity_scores(
    query_embedding: Sequence[float],
    memory_embeddings: Iterable[Sequence[float]]
) -> np.ndarray:
    """
    Compute cosine similarity scores
    using optimized vectorized math.

    Returns:
        np.ndarray of similarity scores.
    """

    memory_embeddings = list(
        memory_embeddings
    )

    if not memory_embeddings:
        return np.array(
            [],
            dtype=np.float32
        )

    # Convert to arrays
    query_vector = to_float32_array(
        query_embedding
    ).reshape(1, -1)

    memory_vectors = to_float32_array(
        memory_embeddings
    )

    # Normalize vectors
    query_vector = normalize_vectors(
        query_vector
    )

    memory_vectors = normalize_vectors(
        memory_vectors
    )

    # Cosine similarity via dot product
    similarity_scores = np.dot(
        memory_vectors,
        query_vector.T
    ).flatten()

    return similarity_scores.astype(
        np.float32
    )


def rank_memories_by_similarity(
    similarity_scores: np.ndarray,
    memories: Sequence
) -> List[Tuple[float, object]]:
    """
    Pair memories with similarity scores
    and sort descending.
    """

    if len(similarity_scores) != len(memories):
        raise ValueError(
            "Similarity scores and memories "
            "must have same length."
        )

    scored_memories = [
        (
            float(score),
            memory
        )
        for score, memory in zip(
            similarity_scores,
            memories
        )
    ]

    scored_memories.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_memories


def filter_relevant_memories(
    scored_memories: List[
        Tuple[float, object]
    ],
    min_similarity: float = 0.3,
    top_k: int = 5
) -> List[Tuple[float, object]]:
    """
    Filter highly relevant memories.

    Args:
        min_similarity:
            Minimum cosine similarity threshold.

        top_k:
            Maximum number of memories to return.
    """

    if top_k <= 0:
        return []

    filtered_memories = [
        (score, memory)
        for score, memory in scored_memories
        if score >= min_similarity
    ]

    return filtered_memories[:top_k]