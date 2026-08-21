from langchain_core.messages import HumanMessage


def create_multimodal_message(
    query,
    retrieved_docs,
    image_data_store
):
    content = []

    # ==================================
    # ADD QUESTION
    # ==================================

    content.append({

        "type": "text",

        "text": (
            f"Question: {query}"
            "\n\nContext:\n"
        )
    })

    # ==================================
    # SEPARATE TEXT AND IMAGES
    # ==================================

    text_docs = [

        doc

        for doc in retrieved_docs

        if doc.metadata.get(
            "type"
        ) == "text"

    ]

    image_docs = [

        doc

        for doc in retrieved_docs

        if doc.metadata.get(
            "type"
        ) == "image"

    ]

    # ==================================
    # ADD TEXT
    # ==================================

    if text_docs:

        text_context = "\n\n".join(

            [

                (
                    f"[Page "
                    f"{doc.metadata['page']}]: "
                    f"{doc.page_content}"
                )

                for doc in text_docs

            ]

        )

        content.append({

            "type": "text",

            "text": (
                "Text excerpts:\n"
                f"{text_context}\n"
            )

        })

    # ==================================
    # ADD IMAGES
    # ==================================

    for doc in image_docs:

        image_id = (
            doc.metadata.get(
                "image_id"
            )
        )

        if (
            image_id
            and image_id in image_data_store
        ):

            # Image label
            content.append({

                "type": "text",

                "text": (
                    f"\n[Image from page "
                    f"{doc.metadata['page']}]:\n"
                )

            })

            # Actual image
            content.append({

                "type": "image_url",

                "image_url": {

                    "url": (
                        "data:image/png;base64,"
                        f"{image_data_store[image_id]}"
                    )

                }

            })

    # ==================================
    # FINAL INSTRUCTION
    # ==================================

    content.append({

        "type": "text",

        "text": (
            "\n\nPlease answer the question "
            "based on the provided text "
            "and images."
        )

    })

    return HumanMessage(
        content=content
    )