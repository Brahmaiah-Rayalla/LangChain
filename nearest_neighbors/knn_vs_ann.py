"""
KNN vs ANN Comparison Example - Comparing exact vs approximate nearest neighbor search.
"""
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def knn_vs_ann_comparison():
    """Compare KNN (exact) vs ANN (approximate) retrieval."""
    print_section("KNN vs ANN Comparison")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create a larger dataset for meaningful comparison
    documents = [
        f"Document {i}: This is sample text about topic {i % 10} with some additional context."
        for i in range(1000)
    ]

    docs = [Document(page_content=doc, metadata={"index": i}) for i, doc in enumerate(documents)]

    print_result("Dataset Size", len(docs))

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS index (exact KNN)
    print("\nCreating FAISS index (KNN)...")
    start_time = time.time()
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_time = time.time() - start_time
    print_result("FAISS Index Creation Time", f"{faiss_time:.2f} seconds")

    # Create Chroma index (ANN with HNSW)
    print("\nCreating Chroma index (ANN)...")
    start_time = time.time()
    chroma_store = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="comparison_collection"
    )
    chroma_time = time.time() - start_time
    print_result("Chroma Index Creation Time", f"{chroma_time:.2f} seconds")

    # Query for comparison
    query = "Document about topic 5"

    # KNN search with FAISS
    print("\nPerforming KNN search (FAISS)...")
    start_time = time.time()
    knn_results = faiss_store.similarity_search(query, k=5)
    knn_search_time = time.time() - start_time
    print_result("KNN Search Time", f"{knn_search_time:.4f} seconds")

    # ANN search with Chroma
    print("\nPerforming ANN search (Chroma)...")
    start_time = time.time()
    ann_results = chroma_store.similarity_search(query, k=5)
    ann_search_time = time.time() - start_time
    print_result("ANN Search Time", f"{ann_search_time:.4f} seconds")

    # Summary
    print_result("\nComparison Summary", "")
    print(f"Index Creation: FAISS (KNN) = {faiss_time:.2f}s, Chroma (ANN) = {chroma_time:.2f}s")
    print(f"Search Time: FAISS (KNN) = {knn_search_time:.4f}s, Chroma (ANN) = {ann_search_time:.4f}s")
    print("\nNote: For small datasets, KNN may be faster. For large datasets (100K+), ANN is typically faster.")

    # Clean up
    chroma_store.delete_collection()


if __name__ == "__main__":
    knn_vs_ann_comparison()
