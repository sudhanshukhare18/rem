"""
Global constants for the memories app.
"""

# =========================
# Similarity Search
# =========================

# Number of top semantic matches
DEFAULT_TOP_K = 5

# Minimum cosine similarity threshold
DEFAULT_MIN_SIMILARITY = 0.30

# Maximum similarity score
MAX_SIMILARITY_SCORE = 1.0

# Minimum similarity score
MIN_SIMILARITY_SCORE = -1.0


# =========================
# Embeddings
# =========================

# BGE-small embedding dimension
EMBEDDING_DIMENSION = 384

# Embedding dtype
EMBEDDING_DTYPE = "float32"

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 32


# =========================
# Summarization
# =========================

DEFAULT_SUMMARY_MAX_LENGTH = 150

DEFAULT_SUMMARY_MIN_LENGTH = 40

MAX_SUMMARY_INPUT_LENGTH = 4000


# =========================
# Emotion Detection
# =========================

DEFAULT_EMOTION = "neutral"

DEFAULT_SENTIMENT = "neutral"

MIN_EMOTION_CONFIDENCE = 0.50


# =========================
# Memory Processing
# =========================

# Max tags extracted per memory
MAX_TAGS = 10

# Minimum tag length
MIN_TAG_LENGTH = 3

# Max text preview length
TEXT_PREVIEW_LENGTH = 40

# Max fallback summary chars
FALLBACK_SUMMARY_LENGTH = 300


# =========================
# Encryption
# =========================

# Minimum AES decoded bytes
MIN_AES_BYTES = 16


# =========================
# Pagination
# =========================

DEFAULT_PAGE_SIZE = 25

MAX_PAGE_SIZE = 100


# =========================
# File Uploads
# =========================

MEMORY_MEDIA_UPLOAD_PATH = (
    "memory_media/"
)


# =========================
# Logging
# =========================

LOGGER_NAME = "memories"


# =========================
# Supported Sentiments
# =========================

POSITIVE = "positive"

NEUTRAL = "neutral"

NEGATIVE = "negative"

SUPPORTED_SENTIMENTS = {
    POSITIVE,
    NEUTRAL,
    NEGATIVE,
}