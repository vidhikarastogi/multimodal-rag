import fitz
import io
import base64

from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .embeddings import embed_text, embed_image

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def process_pdf(pdf_path):

    # Open PDF
    doc = fitz.open(pdf_path)

    # Store documents
    all_docs = []

    # Store embeddings
    all_embeddings = []

    # Store actual images
    image_data_store = {}

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Loop through pages
    for i, page in enumerate(doc):

        # ==================================
        # PROCESS TEXT
        # ==================================

        text = page.get_text()

        if text.strip():

            # Create temporary document
            temp_doc = Document(
                page_content=text,
                metadata={
                    "page": i,
                    "type": "text"
                }
            )

            # Split text
            text_chunks = splitter.split_documents(
                [temp_doc]
            )

            # Embed each text chunk
            for chunk in text_chunks:

                embedding = embed_text(
                    chunk.page_content
                )

                all_embeddings.append(
                    embedding
                )

                all_docs.append(
                    chunk
                )

        # ==================================
        # PROCESS IMAGES
        # ==================================

        for img_index, img in enumerate(
            page.get_images(full=True)
        ):

            try:

                # Get image reference
                xref = img[0]

                # Extract image
                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image["image"]

                # Convert to PIL
                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                # Unique image ID
                image_id = (
                    f"page_{i}_img_{img_index}"
                )

                # ==================================
                # STORE IMAGE AS BASE64
                # ==================================

                buffered = io.BytesIO()

                pil_image.save(
                    buffered,
                    format="PNG"
                )

                img_base64 = base64.b64encode(
                    buffered.getvalue()
                ).decode()

                image_data_store[
                    image_id
                ] = img_base64

                # ==================================
                # CREATE IMAGE EMBEDDING
                # ==================================

                embedding = embed_image(
                    pil_image
                )

                all_embeddings.append(
                    embedding
                )

                # ==================================
                # CREATE IMAGE DOCUMENT
                # ==================================

                image_doc = Document(
                    page_content=(
                        f"[Image: {image_id}]"
                    ),
                    metadata={
                        "page": i,
                        "type": "image",
                        "image_id": image_id
                    }
                )

                all_docs.append(
                    image_doc
                )

            except Exception as e:

                print(
                    f"Error processing image "
                    f"{img_index} on page {i}: {e}"
                )

                continue

    # Close PDF
    doc.close()

    return (
        all_docs,
        all_embeddings,
        image_data_store
    )