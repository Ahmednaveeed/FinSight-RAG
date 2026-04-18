# Multimodal Financial RAG Pipeline

**Description:**
An end-to-end multimodal RAG pipeline to chat with financial PDFs and charts. Built with LangChain, Mistral-7B, EasyOCR, and Gradio. Users can ask questions via text or by uploading image snippets to search and analyze document data. Features include local 4-bit LLM inference, embedding visualization, evaluation metrics, and source citations.

---

## 🚀 Features

* **Multimodal Data Extraction:** Parses complex financial PDFs, extracting both standard text and data from embedded charts/graphs using EasyOCR.
* **Text & Image Queries:** Users can query the system using natural language text or by uploading an image (e.g., a screenshot of a table or chart).
* **Advanced RAG Architecture:** Utilizes ChromaDB for vector storage and retrieval, ensuring accurate semantic search across both text and OCR-processed image chunks.
* **Local LLM Integration:** Powered by `Mistral-7B-Instruct-v0.1` running locally with 4-bit quantization, connected via LangChain.
* **Chain-of-Thought (CoT) Prompting:** Instructs the LLM to think step-by-step as a financial analyst for accurate, grounded reasoning.
* **Interactive UI:** A responsive, Gradio-based web interface for seamless text and image interactions.
* **Analytics & Evaluation:** Includes Principal Component Analysis (PCA) visualizations of the semantic embedding space, along with ROUGE and BLEU scoring for response evaluation.

## 🛠️ Tech Stack

* **Document Processing:** `PyMuPDF` (fitz), `EasyOCR`, `Pillow`, `NumPy`
* **Embeddings & Vector DB:** `sentence-transformers/all-MiniLM-L6-v2`, `ChromaDB`
* **LLM & Orchestration:** `Mistral-7B-Instruct-v0.1` (Hugging Face / bitsandbytes 4-bit quantization), `LangChain`
* **User Interface:** `Gradio`
* **Evaluation:** `NLTK` (BLEU), `rouge-score`, `scikit-learn` (PCA / Cosine Similarity), `Matplotlib`

## 🧩 Pipeline Overview

1. **Extraction:** The system reads through PDF files. Standard text is grouped into documents. Images (like bar charts) are run through EasyOCR, and the structural data is extracted as `[IMAGE CONTENT: ...]`.
2. **Chunking & Embedding:** Both text and image-text chunks are split using LangChain's `RecursiveCharacterTextSplitter` and embedded using the lightweight `all-MiniLM-L6-v2` model.
3. **Retrieval:** If a user types a query, it is vector-searched. If a user uploads an image, the system automatically uses EasyOCR to extract the text from the uploaded image and dynamically searches the vector database for related financial data.
4. **Generation:** The top 3 most relevant context chunks are injected into a Chain-of-Thought prompt, guiding Mistral-7B to deduce the answer strictly based on the provided financial context. The final response includes source citations.
5. **Evaluation:** ROUGE and BLEU metrics are calculated to score the generative quality of the pipeline against reference answers.

## 🏃 Setup and Usage

This project was built to run effectively in a Google Colab / Jupyter environment with GPU support.

1. Clone this repository.
2. Open `FinSight.ipynb` in a Jupyter/Colab environment.
3. Place your financial PDF files in a `/data/` directory accessible by the notebook.
4. Run the notebook cells sequentially to:
   - Extract data and build the ChromaDB vector store.
   - Load the Mistral-7B model (requires a Hugging Face token).
   - Launch the Gradio web interface.

## 📊 Visualizations

The pipeline includes a built-in PCA reduction tool to visualize the 384-dimensional embedding space. It plots document chunks vs. user query vectors to demonstrate semantic proximity and retrieval behavior visually.

---
*Developed as an exploration into Advanced Prompting, Multimodal RAG architectures, and efficient local LLM deployment.*
