"""
Record Manager Example - Managing document updates and deletions in index.
"""
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def record_manager_example():
    """Example of using RecordManager to track document changes."""
    print_section("Record Manager Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize record manager
    namespace = "langchain_example"
    record_manager = SQLRecordManager(
        namespace=namespace,
        db_url="sqlite:///record_manager_cache.sql"
    )

    # Create schema if it doesn't exist
    record_manager.create_schema()

    print_result("Record Manager Initialized", f"Namespace: {namespace}")

    # Create sample documents
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason about their responses.
    """

    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    # Load and split documents
    loader = TextLoader("sample.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(embedding_function=embeddings)

    # Index documents with record manager
    result = index(
        docs=splits,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="full",
        source_id_key="source"
    )

    print_result("Indexing Result", f"Added: {result['num_added']}, Deleted: {result['num_deleted']}, Skipped: {result['num_skipped']}")

    # Test retrieval
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query, k=1)

    print_result("Search Results", f"Found {len(results)} documents")

    # Clean up
    os.remove("sample.txt")
    if os.path.exists("record_manager_cache.sql"):
        os.remove("record_manager_cache.sql")


if __name__ == "__main__":
    record_manager_example()
