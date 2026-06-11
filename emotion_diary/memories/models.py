from django.conf import settings
from django.db import models


class Memory(models.Model):
    """
    Stores encrypted user memories with:
    - embeddings
    - emotions
    - sentiment
    - semantic tags
    - optional media
    """

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

    SENTIMENT_CHOICES = [
        (POSITIVE, "Positive"),
        (NEUTRAL, "Neutral"),
        (NEGATIVE, "Negative"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="memories",
        db_index=True,
    )

    # Encrypted memory text
    text_content = models.TextField()

    # Encrypted emotion label
    emotion_label = models.CharField(
        max_length=100
    )

    # Encrypted sentiment
    sentiment = models.CharField(
        max_length=20,
        choices=SENTIMENT_CHOICES,
        default=NEUTRAL,
        db_index=True,
    )

    # Vector embedding
    embedding = models.JSONField(
        null=True,
        blank=True,
    )

    # Semantic tags
    tags = models.JSONField(
        default=list,
        blank=True,
    )

    # Optional media
    media = models.FileField(
        upload_to="memory_media/",
        null=True,
        blank=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        ordering = ["-created_at"]

        indexes = [
            models.Index(
                fields=["user", "-created_at"]
            ),
            models.Index(
                fields=["user", "sentiment"]
            ),
        ]

    def __str__(self) -> str:
        """
        Human-readable representation.
        """

        preview = (
            self.text_content[:40]
            if self.text_content
            else "Empty Memory"
        )

        return (
            f"Memory("
            f"id={self.id}, "
            f"user={self.user_id}, "
            f"sentiment={self.sentiment}, "
            f"text='{preview}'"
            f")"
        )

    @property
    def has_embedding(self) -> bool:
        """
        Check whether memory contains embedding.
        """

        return bool(self.embedding)

    @property
    def has_media(self) -> bool:
        """
        Check whether memory has media attached.
        """

        return bool(self.media)

    @property
    def tag_count(self) -> int:
        """
        Number of semantic tags.
        """

        return len(self.tags or [])