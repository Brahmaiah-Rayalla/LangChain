"""
ANN (Approximate Nearest Neighbors) Retrieval Example - Fast approximate search.
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def ann_retrieval_example():
    """Example of ANN retrieval using Chroma."""
    print_section("ANN Retrieval Example (Approximate Nearest Neighbors)")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn patterns in data.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning is about learning optimal actions through trial and error.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning finds patterns in unlabeled data without explicit guidance.",
        "Transfer learning adapts pre-trained models to new tasks with limited data."
    ]

    # Create documents with metadata
    from langchain.schema import Document
    docs = [Document(page_content=doc, metadata={"category": "AI/ML"}) for doc in documents]

    print_result("Number of Documents", len(docs))

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create Chroma vector store (uses ANN/HNSW for fast approximate search)
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="ann_collection"
    )

    # Perform ANN search (faster than exact KNN for large datasets)
    query = "What is deep learning?"
    k = 3

    results = vectorstore.similarity_search(query, k=k)

    print_result(f"Top {k} Approximate Nearest Neighbors for: '{query}'", "")
    print_documents(results, max_docs=k)

    # ANN with MMR (Maximal Marginal Relevance) for diversity
    mmr_results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=10)

    print_result("\nMMR Results (Diverse results)", "")
    print_documents(mmr_results, max_docs=k)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    ann_retrieval_example()
