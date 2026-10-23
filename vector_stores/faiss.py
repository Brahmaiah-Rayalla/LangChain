"""
FAISS Vector Store Example - Using FAISS for efficient similarity search.
"""
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def faiss_basic_example():
    """Example of basic FAISS vector store usage."""
    print_section("FAISS Vector Store Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a versatile programming language."),
        Document(page_content="JavaScript is essential for web development."),
        Document(page_content="Java is used for enterprise applications."),
        Document(page_content="C++ is powerful for system programming."),
        Document(page_content="Go is efficient for concurrent programming."),
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)

    print_result("FAISS Index Created", "FAISS with 5 documents")

    # Similarity search
    query = "What language is used for web development?"
    results = vectorstore.similarity_search(query, k=2)

    print_result(f"Similarity Search: '{query}'", "")
    print_documents(results)

    # Similarity search with scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

    print_result("\nSearch with Scores", "")
    for doc, score in results_with_scores:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}\n")

    # Max marginal relevance search
    mmr_results = vectorstore.max_marginal_relevance_search(query, k=2, fetch_k=10)

    print_result("\nMMR Search", "")
    print_documents(mmr_results)


def faiss_persistence_example():
    """Example of saving and loading FAISS index."""
    print_section("FAISS Persistence Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS index
    documents = [
        Document(page_content="LangChain is a framework for LLM applications."),
        Document(page_content="FAISS provides efficient similarity search."),
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save to disk
    save_path = "./faiss_index"
    vectorstore.save_local(save_path)

    print_result("FAISS Index Saved", f"Path: {save_path}")

    # Load from disk
    loaded_vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

    print_result("FAISS Index Loaded", "Successfully loaded from disk")

    # Test loaded index
    results = loaded_vectorstore.similarity_search("What is LangChain?", k=1)
    print_documents(results)

    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def faiss_merge_example():
    """Example of merging multiple FAISS indexes."""
    print_section("FAISS Merge Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create first index
    docs1 = [Document(page_content=f"Document {i} from set 1") for i in range(3)]
    vectorstore1 = FAISS.from_documents(docs1, embeddings)

    # Create second index
    docs2 = [Document(page_content=f"Document {i} from set 2") for i in range(3)]
    vectorstore2 = FAISS.from_documents(docs2, embeddings)

    print_result("Indexes Created", "2 separate FAISS indexes")

    # Merge indexes
    vectorstore1.merge_from(vectorstore2)

    print_result("Indexes Merged", "Combined into single index")

    # Test merged index
    results = vectorstore1.similarity_search("Document", k=6)
    print_result("Total Documents After Merge", len(results))
    print_documents(results)


if __name__ == "__main__":
    faiss_basic_example()
    faiss_persistence_example()
    faiss_merge_example()
