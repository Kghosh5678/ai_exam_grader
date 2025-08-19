import os
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# ---------- CONFIG ----------
PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
# ----------------------------

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

def build_embeddings():
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []
    metadata = []

    print("📚 Processing PDFs...")
    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)
        pages = extract_text_from_pdf(pdf_path)

        for page_number, text in pages:
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    metadata.append({
                        "pdf": filename,
                        "page": page_number,
                        "text": chunk
                    })

    print(f"🧠 Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    print("💾 Saving FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_FOLDER, "faiss_index.index"))

    print("📝 Saving metadata...")
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Done! Your offline AI knowledge base is ready.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    build_embeddings()
