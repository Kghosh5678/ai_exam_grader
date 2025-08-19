import os
import streamlit as st
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

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

def process_pdfs(uploaded_files, model):
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

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, os.path.join(OUTPUT_FOLDER, "faiss_index.index"))
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return len(all_chunks)

@st.cache_data(show_spinner=False)
def load_faiss_index():
    index_path = os.path.join(OUTPUT_FOLDER, "faiss_index.index")
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        return None

@st.cache_data(show_spinner=False)
def load_metadata():
    meta_path = os.path.join(OUTPUT_FOLDER, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

model = load_model()

st.title("📚 AI Exam Grader")

tab1, tab2 = st.tabs(["Train Model (Upload PDFs)", "Query Knowledge Base"])

with tab1:
    st.header("Upload PDF files to train the model")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded. Ready to train.")
        if st.button("🧠 Train the Model"):
            with st.spinner("Processing and embedding PDFs..."):
                total_chunks = process_pdfs(uploaded_files, model)
            st.success(f"✅ Model trained with {total_chunks} chunks from {len(uploaded_files)} PDF(s).")

with tab2:
    st.header("Query the trained knowledge base")
    query = st.text_area("Enter your question or student answer here", height=150)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter some text to search.")
        else:
            index = load_faiss_index()
            metadata = load_metadata()

            if index is None or len(metadata) == 0:
                st.warning("No trained model found. Please upload PDFs and train first.")
            else:
                query_embedding = model.encode([query])
                D, I = index.search(np.array(query_embedding).astype('float32'), TOP_K)

                st.write(f"Top {TOP_K} results:")
                for rank, idx in enumerate(I[0]):
                    item = metadata[idx]
                    st.markdown(f"**Rank {rank + 1}** — _{item['pdf']} (Page {item['page']})_")
                    st.write(item['text'])
                    st.write("---")
