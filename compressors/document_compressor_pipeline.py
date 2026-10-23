"""
Document Compressor Pipeline Example - Combining multiple compressors.
"""
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
    DocumentCompressorPipeline
)
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def document_compressor_pipeline_example():
    """Example of using DocumentCompressorPipeline to combine multiple compressors."""
    print_section("Document Compressor Pipeline Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity. It was created by Guido van Rossum in 1991. Python is widely used for web development, data science, and automation."),
        Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower and its rich history."),
        Document(page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
        Document(page_content="JavaScript is a dynamic programming language used for web development. It was created by Brendan Eich in 1995."),
    ]

    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create individual compressors
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.75
    )
    llm_extractor = LLMChainExtractor.from_llm(llm)

    # Create pipeline: first filter by similarity, then extract relevant parts
    pipeline = DocumentCompressorPipeline(
        transformers=[embeddings_filter, llm_extractor]
    )

    print_result("Pipeline Created", "EmbeddingsFilter -> LLMChainExtractor")

    # Compress documents
    query = "Tell me about programming languages"

    print_result("\nOriginal Documents", f"{len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"{i + 1}. {doc.page_content[:80]}...")

    print_result("\nCompressed Documents", "")
    compressed_docs = pipeline.compress_documents(documents, query)

    for i, doc in enumerate(compressed_docs):
        print(f"\n--- Compressed Document {i + 1} ---")
        print(doc.page_content)


if __name__ == "__main__":
    document_compressor_pipeline_example()
