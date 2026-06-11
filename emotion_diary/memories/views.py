import logging

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render

from memories.models import Memory

from memories.services.memory_search import (
    search_memories,
    
)

from memories.services.memory_store import (
    process_and_store_text,
)

from memories.utils.crypto import (
    safe_decrypt,
)


logger = logging.getLogger(__name__)


@login_required
def home_view(request):
    """
    Home page for memories app.
    """

    return render(
        request,
        "index.html"
    )


@login_required
def store_memory_view(request):
    """
    Store a new emotional memory.
    """

    if request.method != "POST":

        return render(
            request,
            "memories/store.html"
        )

    try:

        text = (
            request.POST.get("text", "")
            .strip()
        )

        media = request.FILES.get(
            "media"
        )

        # Validate input
        if not text:

            return JsonResponse(
                {
                    "success": False,
                    "message": (
                        "Text is required."
                    ),
                },
                status=400
            )

        # Store memory
        memory = process_and_store_text(
            text=text,
            user=request.user,
            media=media
        )

        if not memory:

            return JsonResponse(
                {
                    "success": False,
                    "message": (
                        "Failed to save memory."
                    ),
                },
                status=500
            )

        logger.info(
            "Memory stored successfully "
            "| memory_id=%s | user_id=%s",
            memory.id,
            request.user.id
        )

        return JsonResponse(
            {
                "success": True,
                "message": (
                    "Memory saved successfully."
                ),
                "memory_id": memory.id,
            }
        )

    except Exception as error:

        logger.exception(
            "Memory storage view failed: %s",
            error
        )

        return JsonResponse(
            {
                "success": False,
                "message": (
                    "Unexpected server error."
                ),
            },
            status=500
        )


@login_required
def search_page_view(request):
    """
    Render semantic search page.
    """

    return render(
        request,
        "search.html"
    )

@login_required
def search_memory_view(request):
    """
    Search emotional memories.
    """

    query = request.GET.get(
        "q",
        ""
    ).strip()

    result = {
        "summary": "",
        "matches": [],
    }

    if query:
        result = search_memories(
            query=query,
            user=request.user
        )

    return render(
        request,
        "search_summary.html",
        {
            "query": query,
            "summary": result["summary"],
            "matches": result["matches"],
        }
    )

@login_required
def memory_list_view(request):
    """
    Show all user memories.
    """

    try:

        memories = (
            Memory.objects
            .filter(user=request.user)
            .only(
                "id",
                "text_content",
                "emotion_label",
                "sentiment",
                "tags",
                "created_at",
            )
            .order_by("-created_at")
        )

        formatted_memories = []

        for memory in memories:

            decrypted_text = safe_decrypt(
                memory.text_content,
                fallback="[Unavailable]"
            )

            decrypted_emotion = safe_decrypt(
                memory.emotion_label,
                fallback="neutral"
            )

            formatted_memories.append(
                {
                    "id": memory.id,
                    "text": decrypted_text,
                    "emotion": (
                        decrypted_emotion
                    ),
                    "sentiment": (
                        memory.sentiment
                    ),
                    "tags": memory.tags,
                    "created_at": (
                        memory.created_at
                    ),
                    "has_media": bool(
                        memory.media
                    ),
                }
            )

        return render(
            request,
            "memories/list.html",
            {
                "memories": (
                    formatted_memories
                )
            }
        )

    except Exception as error:

        logger.exception(
            "Memory list view failed: %s",
            error
        )

        return render(
            request,
            "memories/list.html",
            {
                "memories": [],
                "error": (
                    "Failed to load memories."
                ),
            }
        )