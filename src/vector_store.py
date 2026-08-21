import numpy as np

from langchain_community.vectorstores import FAISS


def create_vector_store(
    all_docs,
    all_embeddings
):

    # Convert embeddings to NumPy array
    embeddings_array = np.array(
        all_embeddings
    )

    # Create FAISS vector store
    vector_store = FAISS.from_embeddings(

        text_embeddings=[
            (
                doc.page_content,
                embedding
            )

            for doc, embedding in zip(
                all_docs,
                embeddings_array
            )
        ],

        # We already created embeddings
        embedding=None,

        # Store metadata
        metadatas=[
            doc.metadata
            for doc in all_docs
        ]
    )

    return vector_store