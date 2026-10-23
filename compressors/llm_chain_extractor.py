"""
LLM Chain Extractor Example - Extracting relevant parts of documents using LLM.
"""
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def llm_chain_extractor_example():
    """Example of using LLMChainExtractor to extract relevant document parts."""
    print_section("LLM Chain Extractor Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity. It was created by Guido van Rossum in 1991. Python is widely used for web development, data science, and automation."),
        Document(page_content="JavaScript is a dynamic programming language used for web development. It was created by Brendan Eich in 1995. JavaScript runs in browsers and can also be used on the server with Node.js."),
    ]

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    print_result("Compressor Created", "LLMChainExtractor")

    # Compress documents
    query = "What is Python used for?"

    print_result("\nOriginal Documents", "")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i + 1} ---")
        print(doc.page_content)

    print_result("\nCompressed Documents", "")
    compressed_docs = compressor.compress_documents(documents, query)

    for i, doc in enumerate(compressed_docs):
        print(f"\n--- Compressed Document {i + 1} ---")
        print(doc.page_content)


if __name__ == "__main__":
    llm_chain_extractor_example()
