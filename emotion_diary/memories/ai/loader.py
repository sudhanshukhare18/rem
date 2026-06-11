import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class AIModels:


    embedder=None
    summarizer = None
    emotion_classifier = None
    nlp = None

def get_embedder():
    """
    Load embedding model once.
    """

    if AIModels.embedder is None:

        AIModels.embedder = (
            SentenceTransformer(
                "BAAI/bge-small-en-v1.5"
            )
        )

    return AIModels.embedder


def get_summarizer():
    """
    Load summarization model once.
    """

    if AIModels.summarizer is None:

        AIModels.summarizer = pipeline(
            task="summarization",
            model=(
                "philschmid/"
                "bart-large-cnn-samsum"
            ),
            device=-1
        )

    return AIModels.summarizer


def get_emotion_classifier():
    """
    Load emotion classifier once.
    """

    if (
        AIModels.emotion_classifier
        is None
    ):

        AIModels.emotion_classifier = (
            pipeline(
                task="text-classification",
                model=(
                    "j-hartmann/"
                    "emotion-english-"
                    "distilroberta-base"
                ),
                top_k=None,
                device=-1
            )
        )

    return AIModels.emotion_classifier


def get_nlp():
    """
    Load spaCy NLP model once.
    """

    if AIModels.nlp is None:

        AIModels.nlp = spacy.load(
            "en_core_web_sm"
        )

    return AIModels.nlp