from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings