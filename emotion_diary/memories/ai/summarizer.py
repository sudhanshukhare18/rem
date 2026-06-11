import logging
from collections import Counter
from typing import List

from .loader import get_summarizer


logger = logging.getLogger(__name__)


# Emotion → summary tone mapping
TONE_MAP = {
    "joy": "cheerful",
    "happiness": "uplifting",
    "love": "warm",
    "admiration": "inspiring",
    "gratitude": "heartfelt",
    "optimism": "hopeful",

    "sadness": "nostalgic",
    "grief": "emotional",
    "fear": "thoughtful",
    "nervousness": "reflective",

    "anger": "intense",
    "annoyance": "serious",
    "disappointment": "melancholic",

    "surprise": "curious",
    "confusion": "analytical",
    "neutral": "balanced",
}


def detect_dominant_emotion(
    emotions: List[str]
) -> str:
    """
    Detect the most common emotion.
    """

    if not emotions:
        return "neutral"

    normalized_emotions = [
        emotion.lower().strip()
        for emotion in emotions
        if emotion and isinstance(emotion, str)
    ]

    if not normalized_emotions:
        return "neutral"

    return Counter(
        normalized_emotions
    ).most_common(1)[0][0]


def build_summary_prompt(
    text: str,
    tone: str
) -> str:
    """
    Create emotionally-aware summarization prompt.
    """

    return (
        f"Summarize the following conversation "
        f"in a concise, clear, and emotionally "
        f"{tone} tone.\n\n"
        f"{text}"
    )


def clean_summary(summary: str) -> str:
    """
    Clean generated summary output.
    """

    if not summary:
        return ""

    return " ".join(
        summary.strip().split()
    )


def summarize_text(
    combined_text: str,
    emotions: List[str],
    max_length: int = 150,
    min_length: int = 40
) -> str:
    """
    Generate emotionally-aware summary.
    """

    # Empty input protection
    if not combined_text or not combined_text.strip():
        return ""

    try:

        dominant_emotion = (
            detect_dominant_emotion(
                emotions
            )
        )

        tone = TONE_MAP.get(
            dominant_emotion,
            "balanced"
        )

        prompt = build_summary_prompt(
            combined_text,
            tone
        )

        summarizer = get_summarizer()

        # Protect against transformer limits
        prompt = prompt[:4000]

        result = summarizer(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )

        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], dict)
        ):

            summary = result[0].get(
                "summary_text",
                ""
            )

            cleaned_summary = clean_summary(
                summary
            )

            if cleaned_summary:
                return cleaned_summary

    except Exception as error:

        logger.exception(
            "Summarization failed: %s",
            error
        )

    # Safe fallback
    fallback = combined_text[:300].strip()

    if len(combined_text) > 300:
        fallback += "..."

    return fallback