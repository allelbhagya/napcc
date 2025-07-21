from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_pdf(path, chunk_size=500, overlap=50):
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

    if not text.strip():
        raise ValueError("no extractable text")

    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size].strip())

    return chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, model

def build_index(embeddings):
    embeddings = np.atleast_2d(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def prepare_rag_inputs(pdf_path):
    chunks = load_pdf(pdf_path)
    embeddings, model = embed_chunks(chunks)
    index = build_index(embeddings)
    return chunks, embeddings, model, index
