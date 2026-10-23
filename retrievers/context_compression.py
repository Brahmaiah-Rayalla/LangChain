"""
Context Compression Retriever Example - Compressing retrieved documents.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def context_compression_example():
    """Example of using ContextualCompressionRetriever to compress retrieved documents."""
    print_section("Context Compression Retriever Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents with additional context
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including object-oriented, functional, and procedural programming."),
        Document(page_content="JavaScript is a dynamic programming language primarily used for web development. It was created by Brendan Eich in 1995. JavaScript can be used for both client-side and server-side development with Node.js."),
        Document(page_content="Java is a class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It was developed by James Gosling at Sun Microsystems and released in 1995."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Initialize LLM for compression
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Create context compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    print_result("Context Compression Retriever Created", "Compresses retrieved documents to relevant parts")

    # Test without compression
    query = "Tell me about Python's creation"
    print(f"\n--- Without Compression ---")
    uncompressed_results = base_retriever.invoke(query)
    print_documents(uncompressed_results)

    # Test with compression
    print(f"\n--- With Compression ---")
    compressed_results = compression_retriever.invoke(query)
    print_documents(compressed_results)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    context_compression_example()
