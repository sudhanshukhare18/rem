import logging 
from typing import Dict

from .loader import get_emotion_classifier

logger = logging.getLogger(__name__)

EMOTION_SENTIMENT_MAP={
    "joy": "positive",
    "happiness": "positive",
    "love": "positive",
    "admiration": "positive",
    "approval": "positive",
    "caring": "positive",
    "gratitude": "positive",
    "optimism": "positive",
    "pride": "positive",
    "excitement": "positive",
    "amusement": "positive",

    "surprise": "neutral",
    "curiosity": "neutral",
    "confusion": "neutral",
    "neutral": "neutral",
    "realization": "neutral",

    "sadness": "negative",
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "fear": "negative",
    "grief": "negative",
    "nervousness": "negative",
    "remorse": "negative",
}

def detect_emotion(text:str) ->Dict:
    
    default_response={
        "emotion":"neutral",
        "confidence":0.0,
        "sentiment":"neutral"
    }

    if not text or not text.strip():
        return default_response
    
    try:
        classifier=get_emotion_classifier()

        prediction=classifier(
            text[:512],
            truncation=True
        )

        if not prediction:
            return default_response
        
        if isinstance(predictions[0], list):
            predictions = predictions[0]

        # Get highest confidence prediction
        best_prediction = max(
            predictions,
            key=lambda x: x.get("score", 0)
        )

        emotion = best_prediction.get(
            "label",
            "neutral"
        ).lower()

        confidence = float(
            best_prediction.get("score", 0.0)
        )

        sentiment = map_sentiment(emotion)

    except Exception as error:
        logger.exception(
            "Emotion Detection Failed"
        )
        return default_response
    
def map_sentiment(emotion:str) -> str :

    if not emotion:
        return "neutral"
    
    return EMOTION_SENTIMENT_MAP.get(
        emotion.lower(),
        "neutral"
    )

