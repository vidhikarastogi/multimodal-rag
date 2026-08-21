from .config import PDF_PATH

from .pdf_processor import (
    process_pdf
)

from .vector_store import (
    create_vector_store
)

from .llm import (
    initialize_llm
)

from .pipeline import (
    multimodal_pdf_rag_pipeline
)


def main():

    # ==================================
    # STEP 1: PROCESS PDF
    # ==================================

    print("Processing PDF...")

    (
        all_docs,
        all_embeddings,
        image_data_store
    ) = process_pdf(
        PDF_PATH
    )

    print(
        f"Processed "
        f"{len(all_docs)} documents."
    )

    # ==================================
    # STEP 2: CREATE FAISS
    # ==================================

    print(
        "Creating FAISS vector store..."
    )

    vector_store = create_vector_store(

        all_docs,

        all_embeddings

    )

    print(
        "FAISS vector store created."
    )

    # ==================================
    # STEP 3: INITIALIZE NVIDIA VLM
    # ==================================

    print(
        "Initializing NVIDIA VLM..."
    )

    llm = initialize_llm()

    print(
        "LLM initialized."
    )

    # ==================================
    # STEP 4: INTERACTIVE QUESTIONS
    # ==================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Multimodal RAG is ready!"
    )

    print(
        "Ask questions about your PDF."
    )

    print(
        "Type 'exit' to stop."
    )

    print(
        "=" * 70
    )

    while True:

        query = input(
            "\nAsk a question: "
        )

        # Exit condition
        if query.lower().strip() == "exit":
            print(
                "\nExiting Multimodal RAG..."
            )
            break

        # Don't process empty questions
        if not query.strip():
            print(
                "Please enter a question."
            )
            continue

        # ==================================
        # STEP 5: RUN RAG
        # ==================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Query: {query}"
        )

        print(
            "-" * 50
        )

        try:

            answer = (
                multimodal_pdf_rag_pipeline(

                    query,

                    vector_store,

                    image_data_store,

                    llm

                )
            )

            print(
                f"Answer: {answer}"
            )

        except Exception as e:

            print(
                f"Error: {e}"
            )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()