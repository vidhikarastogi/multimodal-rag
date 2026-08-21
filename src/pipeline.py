from .retrieval import (
    retrieve_multimodal
)

from .message import (
    create_multimodal_message
)


def multimodal_pdf_rag_pipeline(
    query,
    vector_store,
    image_data_store,
    llm,
    k=5
):

    # ==================================
    # RETRIEVE
    # ==================================

    context_docs = retrieve_multimodal(
        query,
        vector_store,
        k=k
    )

    # ==================================
    # CREATE MULTIMODAL MESSAGE
    # ==================================

    message = create_multimodal_message(

        query,

        context_docs,

        image_data_store

    )

    # ==================================
    # GET LLM RESPONSE
    # ==================================

    response = llm.invoke(
        [message]
    )

    # ==================================
    # PRINT RETRIEVED CONTEXT
    # ==================================

    print(
        f"\nRetrieved "
        f"{len(context_docs)} documents:"
    )

    for doc in context_docs:

        doc_type = doc.metadata.get(
            "type",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        if doc_type == "text":

            preview = (
                doc.page_content[:100]
                + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )

            print(
                f"  - Text from page "
                f"{page}: {preview}"
            )

        else:

            print(
                f"  - Image from page {page}"
            )

    print("\n")

    return response.content