import os
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings, save_path):
    os.makedirs(save_path, exist_ok=True)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(save_path)

    return vector_store


def load_vector_store(save_path, embeddings):
    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(
            "Vector store not found. Run build_index.py first."
        )

    vector_store = FAISS.load_local(
        folder_path=save_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store