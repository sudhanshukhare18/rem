from django.contrib import admin

from memories.models import Memory


@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    """
    Admin configuration for Memory model.
    """

    # List page
    list_display = (
        "id",
        "user",
        "sentiment",
        "has_embedding",
        "has_media",
        "tag_count",
        "created_at",
    )

    list_filter = (
        "sentiment",
        "created_at",
    )

    search_fields = (
        "text_content",
        "emotion_label",
        "user__username",
        "user__email",
    )

    ordering = (
        "-created_at",
    )

    date_hierarchy = "created_at"

    list_per_page = 25

    # Performance optimization
    list_select_related = (
        "user",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
        "has_embedding",
        "has_media",
        "tag_count",
    )

    # Cleaner grouped sections
    fieldsets = (
        (
            "User Information",
            {
                "fields": (
                    "user",
                )
            },
        ),

        (
            "Memory Content",
            {
                "fields": (
                    "text_content",
                    "media",
                )
            },
        ),

        (
            "AI Metadata",
            {
                "fields": (
                    "emotion_label",
                    "sentiment",
                    "tags",
                    "embedding",
                )
            },
        ),

        (
            "Memory Insights",
            {
                "fields": (
                    "has_embedding",
                    "has_media",
                    "tag_count",
                )
            },
        ),

        (
            "Timestamps",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )

    def has_embedding(self, obj):
        """
        Check whether memory has embedding.
        """

        return bool(obj.embedding)

    has_embedding.boolean = True

    def has_media(self, obj):
        """
        Check whether memory has media.
        """

        return bool(obj.media)

    has_media.boolean = True

    def tag_count(self, obj):
        """
        Number of semantic tags.
        """

        return len(obj.tags or [])