import re
import base64
import numpy as np
from typing import Dict, Any
from django.utils import timezone
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from .encryption import encrypt_aes, decrypt_aes


# ---------------------- LAZY LOADERS ----------------------

_embedder = None
_summarizer = None
_emotion_pipe = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",  
            device=0 if False else -1  # we can change to torch.cuda.is_available() if using GPU
        )
    return _summarizer


def get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        _emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    return _emotion_pipe


# ---------------------- EMBEDDING ----------------------

def get_embedding(text: str):
    return get_embedder().encode(text)


# ---------------------- HELPERS ----------------------

def extract_tags(text):
    """Simple keyword extraction"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        "the", "and", "but", "are", "you", "for", "this",
        "that", "was", "have", "with"
    }
    return list(set([w for w in words if w not in stopwords]))[:10]


def detect_emotion(text: str) -> str:
    try:
        pipe = get_emotion_pipe()
        res = pipe(text[:512])
        if res and isinstance(res, list):
            return res[0]["label"]
    except Exception:
        pass
    return "neutral"


def looks_like_aes_ciphertext(value: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    if len(value) < 16:
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
        return len(decoded) > 16
    except Exception:
        return False


# ---------------------- STORE MEMORY ----------------------

def process_and_store_text(text, user, media=None):
    from .models import Memory

    try:
        embedding = get_embedding(text).tolist()
        emotion = detect_emotion(text)
        tags = extract_tags(text)

        emotion_to_sentiment = {
            "joy": "positive",
            "happiness": "positive",
            "love": "positive",
            "surprise": "neutral",
            "neutral": "neutral",
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "disgust": "negative",
        }

        sentiment = emotion_to_sentiment.get(emotion.lower(), "neutral")

        memory = Memory.objects.create(
            user=user,
            text_content=encrypt_aes(text),
            emotion_label=encrypt_aes(emotion),
            sentiment=encrypt_aes(sentiment),
            embedding=embedding,
            tags=tags,
            media=media,
            created_at=timezone.now(),
        )

        print(f"✅ Saved memory {memory.id}")
        return memory

    except Exception as e:
        print(f"❌ Error saving memory: {e}")
        return None


# ---------------------- SEARCH + SUMMARIZE ----------------------

def search_and_summarize(query: str, user, top_k: int = 1, min_similarity: float = 0.1) -> Dict[str, Any]:
    from .models import Memory

    result = {"summary": "", "matches": []}

    if not query or not user:
        return result

    try:
        q_emb = np.array(get_embedding(query)).reshape(1, -1)
        memories = Memory.objects.filter(user=user)

        emb_list = []
        for m in memories:
            if m.embedding:
                try:
                    emb = np.array(m.embedding, dtype=float)
                    emb_list.append((m, emb))
                except Exception:
                    continue

        if not emb_list:
            result["summary"] = "No embedded memories found."
            return result

        mem_vectors = np.stack([v for (_, v) in emb_list])
        sims = cosine_similarity(mem_vectors, q_emb).flatten()

        scored = [(float(s), emb_list[i][0]) for i, s in enumerate(sims)]
        filtered = sorted(scored, key=lambda x: x[0], reverse=True)

        filtered = [(s, m) for s, m in filtered if s >= min_similarity][:top_k]

        if not filtered:
            result["summary"] = "No related memories found."
            return result

        matched_memories = []

        for _, m in filtered:
            try:
                text_val = decrypt_aes(m.text_content) if looks_like_aes_ciphertext(m.text_content) else m.text_content
            except Exception:
                text_val = "[Decryption Failed]"

            try:
                emotion_val = decrypt_aes(m.emotion_label) if looks_like_aes_ciphertext(m.emotion_label) else m.emotion_label
            except Exception:
                emotion_val = "neutral"

            matched_memories.append((m, text_val, emotion_val))

        combined_text = "\n".join(
            [f"{text} ({emotion})" for _, text, emotion in matched_memories]
        )

        # Dominant emotion
        emotions = [e for _, _, e in matched_memories if e]
        dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"

        tone_map = {
            "joy": "cheerful",
            "happiness": "uplifting",
            "sadness": "nostalgic",
            "anger": "intense",
            "fear": "thoughtful",
            "love": "warm",
            "surprise": "curious",
            "neutral": "calm",
        }

        tone = tone_map.get(dominant_emotion.lower(), "balanced")

        prompt = f"{combined_text}\n\nSummarize emotionally in a {tone} tone."

        try:
            summarizer = get_summarizer()
            out = summarizer(prompt, max_length=150, min_length=40, do_sample=False)
            summary = out[0]["summary_text"].strip()
            result["summary"] = summary
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            result["summary"] = combined_text[:300] + "..."

        result["matches"] = matched_memories
        return result

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return result