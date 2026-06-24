from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_pdfs(data_dir):
    loader = PyPDFDirectoryLoader(
        path=data_dir,
        glob="**/*.pdf",
        recursive=True
    )

    documents = loader.load()
    return documents