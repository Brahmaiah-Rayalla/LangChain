"""
Vector Store Retriever Example - Using vector stores as retrievers.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def vector_retriever_example():
    """Example of using a vector store as a retriever."""
    print_section("Vector Store Retriever Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language.", metadata={"source": "doc1"}),
        Document(page_content="JavaScript is used for web development.", metadata={"source": "doc2"}),
        Document(page_content="Java is object-oriented and platform-independent.", metadata={"source": "doc3"}),
        Document(page_content="C++ is used for system programming.", metadata={"source": "doc4"}),
        Document(page_content="Go is developed by Google for efficiency.", metadata={"source": "doc5"}),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    print_result("Retriever Created", "Similarity search with k=3")

    # Use the retriever
    query = "What programming language is used for web development?"
    results = retriever.invoke(query)

    print_result(f"Query: '{query}'", "")
    print_documents(results)

    # Test with different search types
    print("\n--- Using MMR Search ---")
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    mmr_results = mmr_retriever.invoke(query)
    print_documents(mmr_results)

    # Test with similarity score threshold
    print("\n--- Using Similarity Score Threshold ---")
    threshold_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )
    threshold_results = threshold_retriever.invoke(query)
    print_documents(threshold_results)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    vector_retriever_example()
