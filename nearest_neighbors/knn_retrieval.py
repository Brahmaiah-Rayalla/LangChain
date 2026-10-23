"""
KNN (K-Nearest Neighbors) Retrieval Example - Finding exact nearest neighbors.
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def knn_retrieval_example():
    """Example of KNN retrieval using FAISS."""
    print_section("KNN Retrieval Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is primarily used for web development and interactive websites.",
        "Java is a class-based, object-oriented programming language.",
        "C++ is a powerful language for system programming and game development.",
        "Go is a statically typed language developed by Google for simplicity and efficiency.",
        "Rust is a systems programming language focused on safety and performance.",
        "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.",
        "Swift is Apple's programming language for iOS and macOS development."
    ]

    # Create documents with metadata
    from langchain.schema import Document
    docs = [Document(page_content=doc, metadata={"index": i}) for i, doc in enumerate(documents)]

    print_result("Number of Documents", len(docs))

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS index (supports exact KNN search)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Perform KNN search
    query = "What programming language is used for iOS development?"
    k = 3  # Number of nearest neighbors

    results = vectorstore.similarity_search(query, k=k)

    print_result(f"Top {k} Nearest Neighbors for: '{query}'", "")
    print_documents(results, max_docs=k)

    # Search with scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    print_result("\nResults with Similarity Scores", "")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"\n--- Result {i + 1} (Score: {score:.4f}) ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    knn_retrieval_example()
