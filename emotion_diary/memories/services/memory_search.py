import logging

from typing import List, Optional

import numpy as np

from memories.ai.embeddings import create_embedding
from memories.ai.summarizer import summarize_text

from memories.utils.constants import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
)

from memories.utils.crypto import safe_decrypt

from memories.utils.similarity import (
    compute_similarity_scores,
    filter_relevant_memories,
    rank_memories_by_similarity,
)

from memories.utils.text import (
    Memory,
    combine_memory_texts,
    extract_emotions,
)


logger = logging.getLogger(__name__)






def collect_memory_embeddings(
    memories
):
    """
    Collect valid memory embeddings.

    Returns:
        List of:
            (
                memory_object,
                embedding_vector
            )
    """

    valid_memories = []

    for memory in memories:

        if not memory.embedding:
            continue

        try:

            embedding_vector = np.asarray(
                memory.embedding,
                dtype=np.float32
            )

            # Skip invalid vectors
            if embedding_vector.size == 0:
                continue

            valid_memories.append(
                (
                    memory,
                    embedding_vector
                )
            )

        except Exception as error:

            logger.warning(
                "Invalid embedding skipped "
                "| memory_id=%s | error=%s",
                getattr(memory, "id", None),
                error
            )

    return valid_memories


def decrypt_memory_data(
    memory
) -> Optional[Memory]:
    """
    Safely decrypt memory fields.
    """

    try:

        text = safe_decrypt(
            memory.text_content,
            fallback=""
        )

        if not text:
            return None

        emotion = safe_decrypt(
            memory.emotion_label,
            fallback="neutral"
        )

        return Memory(
            memory_id=memory.id,
            text=text,
            emotion=emotion
        )

    except Exception as error:

        logger.warning(
            "Memory decryption failed "
            "| memory_id=%s | error=%s",
            getattr(memory, "id", None),
            error
        )

        return None


def search_memories(
    *,
    query: str,
    user,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY
) -> dict:
    """
    Search semantically relevant memories.
    """

    empty_result = {
        "summary":"",
        "matches":[]
    }

    # Input validation
    if (
        not query
        or not query.strip()
        or not user
    ):
        return empty_result

    try:

        from memories.models import Memory as MemoryModel

        cleaned_query = query.strip()

        # Create query embedding
        query_embedding = create_embedding(
            cleaned_query
        )

        # Fetch only required fields
        memories = (
            MemoryModel.objects
            .filter(user=user)
            .only(
                "id",
                "embedding",
                "text_content",
                "emotion_label"
            )
        )

        # Collect valid embeddings
        memory_embeddings = (
            collect_memory_embeddings(
                memories
            )
        )

        if not memory_embeddings:

            return {
                "summary":"No related memories found.",
                "matches":[]
            }

        # Separate memory objects + vectors
        memory_objects = [
            memory
            for memory, _
            in memory_embeddings
        ]

        memory_vectors = [
            vector
            for _, vector
            in memory_embeddings
        ]

        # Compute similarities
        similarity_scores = (
            compute_similarity_scores(
                query_embedding,
                memory_vectors
            )
        )

        # Rank memories
        ranked_memories = (
            rank_memories_by_similarity(
                similarity_scores,
                memory_objects
            )
        )

        # Filter relevant memories
        relevant_memories = (
            filter_relevant_memories(
                ranked_memories,
                min_similarity=min_similarity,
                top_k=top_k
            )
        )

        if not relevant_memories:

            return {
                "summary":"No related memories found.",
                "matches":[]
            }

        matched_memories = []

        for _, memory in relevant_memories:

            decrypted_memory = (
                decrypt_memory_data(
                    memory
                )
            )

            if decrypted_memory:
                matched_memories.append(
                    decrypted_memory
                )

        if not matched_memories:

            return {
                "summary":"No related memories found.",
                "matches":[]
            }

        # Combine memory text
        combined_text = (
            combine_memory_texts(
                matched_memories
            )
        )

        # Extract emotions
        emotions = extract_emotions(
            matched_memories
        )

        # Generate summary
        summary = summarize_text(
            combined_text,
            emotions
        )

        return{
            "summary": summary,
            "matches": matched_memories,
        }

    except Exception as error:

        logger.exception(
            "Memory search failed: %s",
            error
        )

        return empty_result