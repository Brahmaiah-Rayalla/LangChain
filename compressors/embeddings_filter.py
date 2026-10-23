"""
Embeddings Filter Example - Filtering documents by embedding similarity.
"""
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.document_compressors import EmbeddingsFilter
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def embeddings_filter_example():
    """Example of using EmbeddingsFilter to filter documents by similarity."""
    print_section("Embeddings Filter Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="The capital of France is Paris."),
        Document(page_content="Machine learning is a subset of AI."),
        Document(page_content="I like pizza for dinner."),
        Document(page_content="The weather is nice today."),
    ]

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create embeddings filter with a similarity threshold
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.76
    )

    print_result("Filter Created", "EmbeddingsFilter with threshold=0.76")

    # Filter documents
    query = "What is Python?"

    print_result("\nOriginal Documents", f"{len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"{i + 1}. {doc.page_content}")

    print_result("\nFiltered Documents", "")
    filtered_docs = embeddings_filter.compress_documents(documents, query)

    for i, doc in enumerate(filtered_docs):
        print(f"{i + 1}. {doc.page_content}")


if __name__ == "__main__":
    embeddings_filter_example()
