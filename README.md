# Learn RAG (Retrieval-Augmented Generation)

A step-by-step tutorial project to understand and build a complete Retrieval-Augmented Generation (RAG) system locally using **LangChain**, **ChromaDB**, **HuggingFace**, and **Ollama**.

This project provides four Python scripts, each progressively building a more advanced stage of a RAG pipeline.

---

## Features

- **100% Local Processing**: Uses local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and a local LLM via Ollama (`phi3` by default). No API keys or external services are required!
- **Persistent Vector Database**: Embeds and stores documents locally using ChromaDB.
- **Progressive Learning**: Four distinct scripts that take you from simple data ingestion to a fully interactive, history-aware chat application over your data.

## Prerequisites

1. **Python 3.8+**
2. **[Ollama](https://ollama.com/)**: Installed and running locally.
3. The model `phi3` pulled via Ollama (or edit the scripts to use your preferred model like `llama3.2`).
   ```bash
   ollama pull phi3
   ```

## Installation

1. Clone or download this repository.
2. Navigate to the project directory:
   ```bash
   cd Learn-RAG
   ```
3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure & Usage

### 1. Ingestion Pipeline (`1_ingestion_pipeline.py`)
This script loads `.txt` documents from a `docs/` folder, splits them into manageable chunks (with overlap), embeds them using a HuggingFace model, and saves the vectors locally into `db/chroma_db`.

**Usage**:
1. Create a directory named `docs/` in the project root if it doesn't exist.
2. Add your text files (`.txt`) containing the knowledge base into the `docs/` folder.
3. Run the script:
   ```bash
   python 1_ingestion_pipeline.py
   ```

### 2. Retrieval Pipeline (`2_retrieval_pipeline.py`)
Once your documents are ingested, this script demonstrates how to search the Chroma vector database to retrieve the top *K* most relevant document chunks based on a raw user query.

**Usage**:
```bash
python 2_retrieval_pipeline.py
```

### 3. Answer Generation (`3_answer_generation.py`)
This script combines the retrieval step with generation. It retrieves relevant chunks from the database and passes them to a local LLM through Ollama as context. It instructs the model to answer the query *only* based on the provided documents.

**Usage**:
Ensure your Ollama server is running, then execute:
```bash
python 3_answer_generation.py
```

### 4. History-Aware Generation (`4_history_aware_generation.py`)
A complete, interactive terminal chatbot. It maintains conversation history, automatically rewrites follow-up questions to represent standalone queries, retrieves context, and generates history-aware answers.

**Usage**:
1. Make sure the Ollama server is running (`ollama serve`).
2. Run the script:
   ```bash
   python 4_history_aware_generation.py
   ```
3. Type your questions directly into the interactive prompt. Type `quit` to exit or `clear` to reset the conversation history.

---

## Troubleshooting
- **Ollama errors**: Ensure the Ollama app is running in the background. If the model isn't found, ensure you have pulled it: `ollama pull phi3`.
- **Database errors**: If you encounter issues with ChromaDB or update your documents, you can simply delete the active `db/` folder and re-run the `1_ingestion_pipeline.py` script.
