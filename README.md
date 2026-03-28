# Learn RAG - Retrieval-Augmented Generation Pipeline

A learning project to understand how RAG (Retrieval-Augmented Generation) systems work by building a document ingestion and retrieval pipeline.

## What is RAG?

RAG combines:
- **Retrieval**: Finding relevant information from documents
- **Generation**: Using that information to answer questions

This project focuses on the retrieval part - loading documents, creating embeddings, and searching for relevant content.

## Features

- Load multiple text documents
- Smart document chunking with overlap
- Free local embeddings using HuggingFace models
- Vector storage with ChromaDB
- Semantic similarity search

## Installation

```bash
# Clone the repository
git clone https://github.com/raghava-ram/Learn-rag.git
cd Learn-rag

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt