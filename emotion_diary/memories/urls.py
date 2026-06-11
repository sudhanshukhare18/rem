from django.urls import path

from memories.views import (
    home_view,
    memory_list_view,
    search_memory_view,
    store_memory_view,
    search_page_view
)


app_name = "memories"


urlpatterns = [

    # Home
    path(
        "", 
        home_view,
        name="home",
    ), 


    # Store new memory
    path(
        "store/",

        store_memory_view,
        name="store_memory",
    ),
    path( 
    "search-page/",
    search_page_view,
    name="search_page",
    ),
    # Semantic memory search
    path(
        "search/",
        search_memory_view,
        name="search_memory",
    ),

    # User memory timeline
    path(
        "list/",
        memory_list_view,
        name="memory_list",
    ),
]