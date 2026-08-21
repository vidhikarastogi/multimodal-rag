# Multimodal RAG

A **Multimodal Retrieval-Augmented Generation (RAG)** system that processes PDF documents containing visual content such as charts, graphs, and images. The system uses **CLIP embeddings** for visual retrieval, **FAISS** for similarity search, and **NVIDIA Nemotron Vision** for multimodal question answering.

Users can ask questions about the contents of a PDF directly from the terminal, and the system retrieves the most relevant visual information before generating an answer.

---

## 🚀 Project Overview

Traditional RAG systems primarily retrieve and process text. This project extends the RAG approach to handle **visual information contained inside PDF documents**.

The system can work with:

* 📊 Bar charts
* 🥧 Pie charts
* 📈 Graphs
* 🖼️ Images
* 📄 Image-based PDF pages
* 📑 Other visual elements

### Core Pipeline

```text
                 PDF Document
                      │
                      ▼
                   PyMuPDF
                      │
                      ▼
                Extract Images
                      │
                      ▼
                  CLIP Model
                      │
                      ▼
              Image Embeddings
                      │
                      ▼
                 FAISS Index
                      │
                      │
User Question ────────┤
       │              │
       ▼              ▼
 CLIP Text        Similarity Search
 Embedding              │
                        ▼
                Relevant Images
                        │
                        ▼
              NVIDIA Nemotron VL
                        │
                        ▼
                  Final Answer
```

---

## ✨ Features

* PDF image extraction
* Multimodal document processing
* CLIP-based image embeddings
* CLIP-based text/query embeddings
* FAISS vector similarity search
* Top-K relevant image retrieval
* NVIDIA multimodal LLM integration
* Base64 image processing
* Interactive terminal-based question answering
* Environment-variable based API key configuration
* Visual question answering over PDF content

---

## 🛠️ Technologies Used

| Technology                | Purpose                             |
| ------------------------- | ----------------------------------- |
| Python                    | Core programming language           |
| PyMuPDF                   | PDF processing and image extraction |
| Hugging Face Transformers | Loading CLIP model                  |
| OpenAI CLIP               | Image and text embeddings           |
| PyTorch                   | Deep learning backend               |
| FAISS                     | Vector similarity search            |
| LangChain                 | RAG and vector-store framework      |
| NVIDIA AI Endpoints       | Multimodal LLM API                  |
| NVIDIA Nemotron Nano VL   | Image understanding and generation  |
| python-dotenv             | Environment variable management     |
| NumPy                     | Numerical operations                |
| Pillow                    | Image processing                    |

---

## 📁 Project Structure

```text
multimodal-rag/
│
├── .gitignore
├── requirements.txt
├── run.py
├── multimodal_sample.pdf
├── .env
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── embeddings.py
    ├── llm.py
    ├── main.py
    ├── message.py
    ├── pdf_processor.py
    ├── pipeline.py
    ├── retrieval.py
    └── vector_store.py
```

> `.env` should remain local and should not be committed to GitHub.

---

## 📌 Important Files

### `src/pdf_processor.py`

Processes the PDF, extracts images, and prepares the information required for embedding and retrieval.

### `src/embeddings.py`

Loads the CLIP model and generates image and text/query embeddings.

### `src/vector_store.py`

Creates and manages the FAISS vector store using the generated embeddings.

### `src/retrieval.py`

Converts the user's question into a CLIP text embedding and retrieves the most relevant images from FAISS.

### `src/llm.py`

Initializes the NVIDIA multimodal language model.

### `src/pipeline.py`

Connects the retrieval system with the NVIDIA vision model to generate answers based on retrieved images.

### `src/main.py`

Runs the complete application and provides an interactive terminal interface for asking questions.

### `run.py`

Provides the project entry point for running the application.

---

# ⚙️ Installation

## 1. Create a Virtual Environment

Open **Windows PowerShell** inside the project directory:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, you should see:

```text
(.venv) PS C:\...\multimodal-rag>
```

---

## 2. Install Dependencies

Install all required Python packages:

```powershell
python -m pip install -r requirements.txt
```

---

# 🔑 NVIDIA API Key

The project uses NVIDIA AI Endpoints to access the multimodal language model.

Create an NVIDIA API key through the NVIDIA API catalog.

Then create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### Important

Never place the API key directly inside Python source files.

The `.env` file should remain local and should **not** be uploaded to GitHub.

Your `.gitignore` should contain:

```text
.env
.venv/
__pycache__/
```

---

# ⚙️ Configuration

The main configuration is available in:

```text
src/config.py
```

### NVIDIA Model

```python
NVIDIA_MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
```

### CLIP Model

```python
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
```

### PDF File

```python
PDF_PATH = "multimodal_sample.pdf"
```

---

# 🧪 Testing the Project

Before running the complete application, individual components can be tested to verify that the environment and dependencies are working correctly.

---

## 1. Check Python

```powershell
python --version
```

Check pip:

```powershell
python -m pip --version
```

---

## 2. Check NVIDIA API Key

Run:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('NVIDIA API key loaded:', bool(os.getenv('NVIDIA_API_KEY')))"
```

Expected output:

```text
NVIDIA API key loaded: True
```

---

## 3. Test NVIDIA API

Run:

```powershell
python -c "import requests, os; from dotenv import load_dotenv; load_dotenv(); r=requests.post('https://integrate.api.nvidia.com/v1/chat/completions', headers={'Authorization':'Bearer '+os.getenv('NVIDIA_API_KEY'),'Content-Type':'application/json'}, json={'model':'nvidia/llama-3.1-nemotron-nano-vl-8b-v1','messages':[{'role':'user','content':'Say OK.'}],'max_tokens':5,'temperature':0}, timeout=30); print('STATUS:',r.status_code); print(r.text[:1000])"
```

Expected:

```text
STATUS: 200
```

---

## 4. Test PDF Processing

Run:

```powershell
python -c "import fitz; doc=fitz.open('multimodal_sample.pdf'); print('Pages:',len(doc)); [(print('Page',i+1,'images:',len(page.get_images(full=True)),'text:',len(page.get_text()))) for i,page in enumerate(doc)]; doc.close()"
```

For the provided sample PDF, the expected output is:

```text
Pages: 4
Page 1 images: 1
Page 2 images: 1
Page 3 images: 1
Page 4 images: 1
```

---

## 5. Test CLIP

Run:

```powershell
python -c "from transformers import CLIPProcessor, CLIPModel; m=CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); p=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP loaded successfully')"
```

Expected:

```text
CLIP loaded successfully
```

> The first run downloads the CLIP model, so it may take some time.

---

## 6. Test PDF Processing Module

Run:

```powershell
python -c "from src.pdf_processor import process_pdf; d,e,i=process_pdf('multimodal_sample.pdf'); print('Documents:',len(d)); print('Embeddings:',len(e)); print('Images:',len(i))"
```

Expected:

```text
Documents: 4
Embeddings: 4
Images: 4
```

---

## 7. Test FAISS Vector Store

Run:

```powershell
python -c "from src.pdf_processor import process_pdf; from src.vector_store import create_vector_store; d,e,i=process_pdf('multimodal_sample.pdf'); vs=create_vector_store(d,e); print('FAISS vector store created successfully')"
```

Expected:

```text
FAISS vector store created successfully
```

---

## 8. Test Retrieval

Run:

```powershell
python -c "from src.pdf_processor import process_pdf; from src.vector_store import create_vector_store; from src.retrieval import retrieve_multimodal; d,e,i=process_pdf('multimodal_sample.pdf'); vs=create_vector_store(d,e); results=retrieve_multimodal('What information is shown in this document?',vs,k=3); print('Retrieved:',len(results)); [print(j+1,r.metadata,r.page_content[:100]) for j,r in enumerate(results)]"
```

Expected:

```text
Retrieved: 3
```

The retrieved results should contain image-related metadata such as:

```text
'type': 'image'
'image_id': 'page_0_img_0'
```

---

# ▶️ Run the Complete Project

After all individual tests pass, run:

```powershell
python -m src.main
```

You should see something similar to:

```text
Processing PDF...
Processed 4 documents.

Creating FAISS vector store...
FAISS vector store created.

Initializing NVIDIA VLM...
LLM initialized.

======================================================================
Multimodal RAG is ready!
Ask questions about your PDF.
Type 'exit' to stop.
======================================================================

Ask a question:
```

---

# 💬 Ask Questions

Once the application starts, you can ask questions directly from the terminal.

For example:

```text
Ask a question: What does the chart on page 1 show?
```

Other examples:

```text
Ask a question: What are the main findings of the document?
```

```text
Ask a question: What visual elements are present in the document?
```

```text
Ask a question: Summarize the information shown in the charts.
```

You can continue asking questions without restarting the application.

To exit:

```text
exit
```

---

# 📊 Sample Questions

For the provided sample PDF, the system can answer questions related to:

### Market Trends

```text
What does the chart on page 1 show?
```

The system retrieves the relevant chart and uses the multimodal model to understand the visual information.

### Document Summary

```text
Summarize the main findings of the document.
```

The system retrieves relevant visual information and generates a natural-language summary.

### Visual Analysis

```text
What visual elements are present in the document?
```

The system can identify visual elements such as:

* Bar graphs
* Pie charts
* Market segmentation charts
* Other graphical information

---

# 🔄 How the RAG Pipeline Works

## 1. PDF Processing

PyMuPDF opens the PDF document and extracts the images contained within the pages.

```text
PDF → PyMuPDF → Images
```

---

## 2. Image Embedding

The extracted images are passed through the CLIP model.

CLIP converts each image into a numerical vector representation called an **embedding**.

```text
Image → CLIP → Image Embedding
```

---

## 3. Vector Storage

The generated embeddings are stored in a FAISS vector index.

```text
Image Embeddings → FAISS
```

FAISS allows the system to perform efficient similarity searches.

---

## 4. User Query

When a user enters a question, the question is converted into a CLIP text embedding.

```text
User Question → CLIP → Text Embedding
```

---

## 5. Similarity Search

The query embedding is compared with the image embeddings stored in FAISS.

The system retrieves the most relevant images based on similarity.

```text
Query Embedding
       │
       ▼
 FAISS Similarity Search
       │
       ▼
Relevant Images
```

---

## 6. Multimodal Generation

The retrieved images are passed to the NVIDIA multimodal language model.

The model analyzes the visual information and the user's question.

```text
Question + Retrieved Images
            │
            ▼
      NVIDIA Nemotron VL
```

---

## 7. Final Answer

The NVIDIA multimodal model generates a natural-language answer based on the retrieved visual information.

```text
Retrieved Visual Context
          +
      User Question
          │
          ▼
    Final Answer
```

---

# 🧠 Why Multimodal RAG?

Traditional RAG systems generally focus on retrieving text chunks from documents.

However, many real-world documents contain important information inside:

* Charts
* Graphs
* Tables
* Diagrams
* Images
* Infographics

A text-only RAG system may fail to capture this information effectively.

Multimodal RAG addresses this limitation by allowing visual information to participate in the retrieval and generation process.

---

# 🎯 Project Goal

The primary goal of this project is to demonstrate how **Multimodal Retrieval-Augmented Generation** can retrieve and reason over visual information contained within documents.

Instead of relying exclusively on textual content, the system combines:

```text
Visual Retrieval
      +
Vector Similarity Search
      +
Multimodal LLM
      =
Visual Question Answering
```

This approach can be useful for documents containing:

* Charts
* Graphs
* Diagrams
* Images
* Infographics
* Visual reports

---

# ⚠️ Common Warnings

## PyMuPDF Warning

You may encounter a warning related to the `fitz` API being deprecated.

This is currently a warning and does not necessarily prevent the application from running.

---

## Hugging Face Warning

You may see:

```text
You are sending unauthenticated requests to the HF Hub
```

The CLIP model can still download successfully without a Hugging Face token.

---

## FAISS / LangChain Warning

You may see a warning similar to:

```text
embedding_function is expected to be an Embeddings object
```

This can occur because of LangChain compatibility or deprecation changes.

It does not necessarily prevent the current vector-store and retrieval pipeline from working.

---

# 🔐 Security

Never commit your NVIDIA API key to GitHub.

Keep the following files and folders out of version control:

```text
.env
.venv/
__pycache__/
```

The repository should contain:

```text
requirements.txt
src/
run.py
multimodal_sample.pdf
.gitignore
```

But it should **not** contain:

```text
.env
.venv/
__pycache__/
```

---

# 🚀 Future Improvements

Possible improvements for the project include:

* Support for text + image hybrid retrieval
* Table extraction and understanding
* OCR for scanned PDFs
* Page-level multimodal retrieval
* Support for multiple PDF documents
* Persistent FAISS indexes
* Improved metadata filtering
* Web-based user interface
* Conversation history
* Source/page citations in answers
* GPU acceleration
* Support for additional vision-language models

---

# 📌 Conclusion

This project demonstrates a complete **Multimodal RAG pipeline** for retrieving and understanding visual information from PDF documents.

By combining **PyMuPDF, CLIP, FAISS, LangChain, and NVIDIA Nemotron Vision**, the system can retrieve relevant visual content and use a multimodal LLM to answer questions about the document.

The project provides a practical foundation for building AI systems capable of understanding documents beyond traditional text-based retrieval.
