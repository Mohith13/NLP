from config import *

from pipeline.pdf_loader import load_pdfs
from pipeline.text_splitter import split_documents
from pipeline.embeddings import get_embeddings
from pipeline.vector_store import create_vector_store


def build_index():
    print("\nLoading PDFs...")
    documents = load_pdfs(DATA_DIR)

    if len(documents) == 0:
        print("No PDF files found in Data folder.")
        return

    print(f"Loaded {len(documents)} PDF pages.")

    print("Splitting documents...")
    chunks = split_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print(f"Created {len(chunks)} chunks.")

    print("Loading HuggingFace embedding model...")
    embeddings = get_embeddings(EMBEDDING_MODEL)

    print("Creating FAISS vector store...")
    create_vector_store(
        chunks,
        embeddings,
        VECTOR_STORE_DIR
    )

    print("Vector store created successfully.")


if __name__ == "__main__":
    build_index()