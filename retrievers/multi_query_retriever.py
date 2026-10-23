"""
Multi-Query Retriever Example - Generating multiple queries for better retrieval.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def multi_query_retriever_example():
    """Example of using MultiQueryRetriever to generate multiple queries."""
    print_section("Multi-Query Retriever Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Machine learning enables computers to learn from data without explicit programming."),
        Document(page_content="Deep learning uses neural networks with multiple layers for complex pattern recognition."),
        Document(page_content="Natural language processing deals with human-computer language interaction."),
        Document(page_content="Computer vision allows machines to interpret visual information."),
        Document(page_content="Reinforcement learning teaches agents through reward-based feedback."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Initialize LLM for query generation
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    print_result("Multi-Query Retriever Created", "Generates multiple queries for better retrieval")

    # Use the retriever
    query = "How do machines learn from data?"
    results = retriever.invoke(query)

    print_result(f"Original Query: '{query}'", "")
    print_result("Number of Results", len(results))
    print_documents(results)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    multi_query_retriever_example()
