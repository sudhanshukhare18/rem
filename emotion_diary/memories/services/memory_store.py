import logging
from typing import Any, Dict, Optional

from django.db import transaction
from django.utils import timezone

from memories.ai.embeddings import create_embedding
from memories.ai.emotion import (
    detect_emotion,
)
from memories.ai.tags import extract_tags
from memories.encryption import encrypt_aes


logger = logging.getLogger(__name__)


def build_memory_payload(
    text: str
) -> Dict[str, Any]:
    """
    Generate AI metadata for memory storage.
    """

    cleaned_text = text.strip()

    # Emotion analysis
    emotion_result = detect_emotion(
        cleaned_text
    )

    # Embedding generation
    embedding = create_embedding(
        cleaned_text
    ).tolist()

    # Tag extraction
    tags = extract_tags(
        cleaned_text
    )

    return {
        "embedding": embedding,
        "emotion": emotion_result.get(
            "emotion",
            "neutral"
        ),
        "confidence": emotion_result.get(
            "confidence",
            0.0
        ),
        "sentiment": emotion_result.get(
            "sentiment",
            "neutral"
        ),
        "tags": tags,
    }


@transaction.atomic
def save_memory(
    *,
    user,
    text: str,
    payload: Dict[str, Any],
    media=None
):
    """
    Save processed memory into database.
    """

    from memories.models import Memory

    encrypted_text = encrypt_aes(
        text.strip()
    )

    encrypted_emotion = encrypt_aes(
        payload["emotion"]
    )

    encrypted_sentiment = encrypt_aes(
        payload["sentiment"]
    )

    memory = Memory.objects.create(
        user=user,
        text_content=encrypted_text,
        emotion_label=encrypted_emotion,
        sentiment=encrypted_sentiment,
        embedding=payload["embedding"],
        tags=payload["tags"],
        media=media,
        created_at=timezone.now(),
    )

    logger.info(
        "Memory saved successfully | id=%s user=%s",
        memory.id,
        user.id
    )

    return memory


def process_and_store_text(
    *,
    text: str,
    user,
    media=None
) -> Optional[object]:
    """
    Main AI memory processing pipeline.

    Steps:
    1. Generate embeddings
    2. Detect emotion
    3. Extract tags
    4. Encrypt sensitive fields
    5. Store memory
    """

    # Input validation
    if not text or not text.strip():

        logger.warning(
            "Empty memory text received."
        )

        return None

    try:

        cleaned_text = text.strip()

        payload = build_memory_payload(
            cleaned_text
        )

        memory = save_memory(
            user=user,
            text=cleaned_text,
            payload=payload,
            media=media
        )

        return memory

    except Exception as error:

        logger.exception(
            "Memory storage pipeline failed: %s",
            error
        )

        return None