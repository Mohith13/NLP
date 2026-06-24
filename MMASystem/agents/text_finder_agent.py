class TextFinderAgent:
    def __init__(self, vector_store, top_k):
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

    def run(self, query):
        docs = self.retriever.invoke(query)

        results = []

        for doc in docs:
            results.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "text": doc.page_content
            })

        return results