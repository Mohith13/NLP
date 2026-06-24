from config import *

from pipeline.embeddings import get_embeddings
from pipeline.vector_store import load_vector_store

from agents.text_finder_agent import TextFinderAgent
from agents.answer_agent import QAAgent
from agents.summarizer_agent import SummarizerAgent

from Orchestrator.query_router import QueryRouter


def show_sources(docs):
    print("\nSources:")
    for doc in docs:
        print(f"- {doc['source']} | Page {doc['page']}")


def main():
    print("\nLoading embedding model...")
    embeddings = get_embeddings(EMBEDDING_MODEL)

    print("Loading vector store...")
    vector_store = load_vector_store(
        VECTOR_STORE_DIR,
        embeddings
    )

    router = QueryRouter()

    text_finder = TextFinderAgent(vector_store, TOP_K)
    qa_agent = QAAgent(LLM_MODEL)
    summarizer_agent = SummarizerAgent(LLM_MODEL)

    print("\nMulti-Agentic RAG System Ready")

    while True:
        query = input("\nAsk something, or type exit: ")

        if query.lower() == "exit":
            print("Exiting.")
            break

        selected_agent = router.route(query)

        print(f"\nSelected Agent: {selected_agent}")

        docs = text_finder.run(query)

        if selected_agent == "text_finder":
            print("\nRelevant Text Found:")

            for index, doc in enumerate(docs, start=1):
                print(f"\nResult {index}")
                print(f"Source: {doc['source']} | Page: {doc['page']}")
                print(doc["text"][:1200])

        elif selected_agent == "summarizer":
            summary = summarizer_agent.run(query, docs)

            print("\nSummary:")
            print(summary)

            show_sources(docs)

        else:
            answer = qa_agent.run(query, docs)

            print("\nAnswer:")
            print(answer)

            show_sources(docs)


if __name__ == "__main__":
    main()