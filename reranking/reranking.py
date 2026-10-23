"""
Reranking Example - Re-ranking retrieved documents for better relevance.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents, OPENAI_API_KEY


def llm_reranking_example():
    """Example of using LLM to rerank retrieved documents."""
    print_section("LLM Reranking Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability."),
        Document(page_content="The capital of France is Paris, known for the Eiffel Tower."),
        Document(page_content="JavaScript is primarily used for web development and interactive websites."),
        Document(page_content="Java is a class-based, object-oriented programming language."),
        Document(page_content="The weather in Tokyo is pleasant today with clear skies."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Initial retrieval
    query = "What programming language is used for web development?"
    initial_results = vectorstore.similarity_search(query, k=5)

    print_result(f"Query: '{query}'", "")
    print_result("Initial Retrieval (Top 5)", f"{len(initial_results)} documents")
    print_documents(initial_results)

    # Rerank using LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create reranking prompt
    rerank_prompt = f"""
    Given the query: "{query}"
    And the following documents, rank them by relevance (1 = most relevant, 5 = least relevant).

    Documents:
    {chr(10).join([f'{i+1}. {doc.page_content}' for i, doc in enumerate(initial_results)])}

    Return only the ranking numbers in order.
    """

    print_result("\nReranking with LLM", "Using GPT-3.5-turbo to rerank")

    try:
        ranking = llm.invoke(rerank_prompt)
        print_result("LLM Ranking Response", ranking.content)

        # Parse ranking and reorder documents
        # Note: This is a simplified example
        print("\nNote: In production, you would parse the ranking and reorder documents accordingly.")

    except Exception as e:
        print(f"Error during reranking: {e}")

    # Clean up
    vectorstore.delete_collection()


def score_threshold_reranking():
    """Example of reranking using score threshold."""
    print_section("Score Threshold Reranking")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="JavaScript is for web development."),
        Document(page_content="Machine learning is a subset of AI."),
        Document(page_content="The weather is nice today."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Search with scores
    query = "programming language"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=4)

    print_result(f"Query: '{query}'", "")

    # Apply score threshold
    threshold = 0.75
    filtered_results = [(doc, score) for doc, score in results_with_scores if score < threshold]

    print_result(f"Initial Results", f"{len(results_with_scores)} documents")
    for doc, score in results_with_scores:
        print(f"  Score: {score:.4f} - {doc.page_content[:50]}...")

    print_result(f"\nFiltered Results (threshold < {threshold})", f"{len(filtered_results)} documents")
    for doc, score in filtered_results:
        print(f"  Score: {score:.4f} - {doc.page_content[:50]}...")

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    llm_reranking_example()
    score_threshold_reranking()
