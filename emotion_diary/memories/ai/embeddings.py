from .loader import get_embedder

def create_embedding(text:str):
    embedder = get_embedder()

    embedding=  embedder.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False
    )
    return embedding.astype("float32")