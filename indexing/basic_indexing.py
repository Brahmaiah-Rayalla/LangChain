"""
Basic Indexing Example - Creating a document index for retrieval.
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


def basic_indexing_example():
    """Example of creating a basic document index."""
    print_section("Basic Indexing Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason about their responses.

    LangChain provides two main abstractions:
    1. Components - Modular building blocks (LLMs, prompt templates, etc.)
    2. Chains - Sequences of components for specific use cases

    The framework supports various use cases including:
    - Document analysis and Q&A
    - Chatbots
    - Agents
    - Code generation
    """

    # Save to file
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    # Load documents
    loader = TextLoader("sample.txt", encoding="utf-8")
    documents = loader.load()

    print_result("Loaded Documents", len(documents))

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(documents)

    print_result("Number of Splits", len(splits))

    # Create embeddings and index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create vector store (index)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print_result("Index Created", "Successfully created Chroma vector store")

    # Test retrieval
    query = "What are the main abstractions in LangChain?"
    results = vectorstore.similarity_search(query, k=2)

    print_result("Search Results", f"Found {len(results)} relevant documents")
    print_documents(results)

    # Clean up
    os.remove("sample.txt")
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")


if __name__ == "__main__":
    basic_indexing_example()
