"""
Hybrid Search Example - Combining keyword and semantic search.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def hybrid_search_example():
    """Example of hybrid search combining BM25 and vector search."""
    print_section("Hybrid Search Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language with dynamic semantics."),
        Document(page_content="JavaScript is a dynamic programming language for web development."),
        Document(page_content="Java is a class-based, object-oriented programming language."),
        Document(page_content="C++ is a powerful language for system programming."),
        Document(page_content="Go is a statically typed language developed by Google."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    print_result("Vector Store Created", "Chroma with 5 documents")

    # Semantic search (vector-based)
    query = "programming language for web"
    semantic_results = vectorstore.similarity_search(query, k=3)

    print_result(f"Semantic Search: '{query}'", "")
    print_documents(semantic_results)

    # Note: True hybrid search requires BM25 or similar keyword search
    # This example demonstrates the concept
    print("\n--- Hybrid Search Concept ---")
    print("Hybrid search combines:")
    print("1. Semantic search (vector embeddings)")
    print("2. Keyword search (BM25, TF-IDF)")
    print("\nTo implement full hybrid search, you would:")
    print("- Use a keyword search engine (Elasticsearch, BM25)")
    print("- Combine results with vector search")
    print("- Apply score fusion or reciprocal rank fusion")

    # Clean up
    vectorstore.delete_collection()


def weighted_search_example():
    """Example of weighted search combining different retrieval methods."""
    print_section("Weighted Search Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Machine learning uses algorithms to learn from data."),
        Document(page_content="Deep learning is a subset of machine learning using neural networks."),
        Document(page_content="Natural language processing deals with human language."),
        Document(page_content="Computer vision enables machines to see and understand images."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    print_result("Vector Store Created", "Chroma with 4 documents")

    # MMR search (combines relevance and diversity)
    query = "machine learning"
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=10
    )

    print_result(f"MMR Search: '{query}'", "Balances relevance and diversity")
    print_documents(mmr_results)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    hybrid_search_example()
    weighted_search_example()
