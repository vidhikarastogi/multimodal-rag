from .embeddings import embed_text


def retrieve_multimodal(
    query,
    vector_store,
    k=5
):

    """
    Unified retrieval using CLIP
    embeddings for both text and images.
    """

    # Convert query into CLIP embedding
    query_embedding = embed_text(
        query
    )

    # Search FAISS
    results = (
        vector_store
        .similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
    )

    return results