from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(slots=True)
class Memory:
    """
    Structured memory object.
    """

    memory_id: int
    text: str
    emotion: Optional[str] = None


def normalize_text(text: str) -> str:
    """
    Normalize memory text by removing
    extra spaces and line breaks.
    """

    return " ".join(
        str(text).strip().split()
    )


def normalize_emotion(
    emotion: Optional[str]
) -> Optional[str]:
    """
    Normalize emotion label.
    """

    if not emotion:
        return None

    cleaned = emotion.lower().strip()

    return cleaned or None


def combine_memory_texts(
    memories: Iterable[Memory]
) -> str:
    """
    Combine memories into a clean,
    summarization-friendly string.

    Example:
        I enjoyed the vacation.
        [emotion: joy]

        Work felt stressful.
        [emotion: anger]
    """

    combined_parts = []

    for memory in memories:

        # Skip invalid memory objects
        if not memory:
            continue

        text = normalize_text(
            memory.text
        )

        # Skip empty text
        if not text:
            continue

        emotion = normalize_emotion(
            memory.emotion
        )

        # Structured formatting improves
        # summarization and LLM parsing
        if emotion:

            formatted_memory = (
                f"{text}\n"
                f"[emotion: {emotion}]"
            )

        else:
            formatted_memory = text

        combined_parts.append(
            formatted_memory
        )

    return "\n\n".join(combined_parts)


def extract_emotions(
    memories: Iterable[Memory]
) -> List[str]:
    """
    Extract normalized emotion labels
    from memories.
    """

    emotions = []

    for memory in memories:

        if not memory:
            continue

        emotion = normalize_emotion(
            memory.emotion
        )

        if emotion:
            emotions.append(emotion)

    return emotions