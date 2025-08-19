# AI Exam Grader – PDF Embedding Pipeline

This module creates a searchable knowledge base from textbooks (PDFs) for use in AI-based exam grading.

## 🧠 Features
- Extracts and chunks text from multiple PDFs
- Generates semantic embeddings using MiniLM
- Saves a FAISS index + metadata for offline retrieval

## 🚀 How to Use

1. Place your PDF files in the `pdfs/` folder.
2. Run the embedding script:
```bash
python scripts/generate_embeddings.py
