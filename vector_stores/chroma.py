"""
Chroma Vector Store Example - Using Chroma as a vector database.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def chroma_basic_example():
    """Example of basic Chroma vector store usage."""
    print_section("Chroma Vector Store Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language.", metadata={"category": "programming"}),
        Document(page_content="JavaScript is used for web development.", metadata={"category": "programming"}),
        Document(page_content="Machine learning is a subset of AI.", metadata={"category": "AI"}),
        Document(page_content="Deep learning uses neural networks.", metadata={"category": "AI"}),
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="langchain_examples"
    )

    print_result("Vector Store Created", "Chroma with 4 documents")

    # Similarity search
    query = "What programming languages are used for web development?"
    results = vectorstore.similarity_search(query, k=2)

    print_result(f"Similarity Search: '{query}'", "")
    print_documents(results)

    # Similarity search with scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

    print_result("\nSearch with Scores", "")
    for doc, score in results_with_scores:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")

    # Max marginal relevance search (for diversity)
    mmr_results = vectorstore.max_marginal_relevance_search(query, k=2, fetch_k=10)

    print_result("\nMMR Search (Diverse Results)", "")
    print_documents(mmr_results)

    # Clean up
    vectorstore.delete_collection()


def chroma_persistence_example():
    """Example of persisting Chroma vector store to disk."""
    print_section("Chroma Persistence Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create and persist vector store
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        documents=[Document(page_content="LangChain is a framework for LLM applications.")],
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print_result("Vector Store Persisted", f"Directory: {persist_directory}")

    # Load from disk
    loaded_vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print_result("Vector Store Loaded", "Successfully loaded from disk")

    # Test loaded vector store
    results = loaded_vectorstore.similarity_search("What is LangChain?", k=1)
    print_documents(results)

    # Clean up
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)


if __name__ == "__main__":
    chroma_basic_example()
    chroma_persistence_example()
