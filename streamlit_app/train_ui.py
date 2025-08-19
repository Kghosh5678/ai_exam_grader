import os
import streamlit as st
import fitz  # PyMuPDF
import json
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config
PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

model = load_model()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text()
        pages.append((page_num + 1, text))
    return pages

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_pdfs(uploaded_files):
    all_chunks = []
    metadata = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages = extract_text_from_pdf(file_path)
        for page_number, text in pages:
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    metadata.append({
                        "pdf": uploaded_file.name,
                        "page": page_number,
                        "text": chunk
                    })
    return all_chunks, metadata

def generate_and_save_embeddings(chunks, metadata):
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(OUTPUT_FOLDER, "faiss_index.index"))

    # Save metadata
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return len(chunks)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📚 AI Exam Grader – PDF Trainer")
st.markdown("Upload one or more textbooks (PDF) to train the model on domain-specific knowledge.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded. Ready to process.")
    if st.button("🧠 Train the Model"):
        with st.spinner("Processing and embedding..."):
            chunks, metadata = process_pdfs(uploaded_files)
            total = generate_and_save_embeddings(chunks, metadata)
        st.success(f"✅ Model trained with {total} chunks from {len(uploaded_files)} PDF(s).")
