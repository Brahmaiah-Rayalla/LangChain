"""
Retrieval Evaluator Example - Evaluating retrieval quality.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.evaluation import load_evaluator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def retrieval_relevance_example():
    """Example of evaluating retrieval relevance."""
    print_section("Retrieval Relevance Evaluation")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity."),
        Document(page_content="JavaScript is primarily used for web development."),
        Document(page_content="Java is a class-based, object-oriented programming language."),
        Document(page_content="C++ is powerful for system programming."),
        Document(page_content="Go is efficient for concurrent programming."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print_result("Retriever Created", "Chroma with k=3")

    # Load relevance evaluator
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    evaluator = load_evaluator("retrieval_relevance", llm=llm)

    print_result("Evaluator Loaded", "Retrieval Relevance Evaluator")

    # Evaluate retrieval
    query = "What programming language is used for web development?"
    retrieved_docs = retriever.invoke(query)

    print_result(f"Query: '{query}'", "")

    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i + 1} ---")
        print(f"Content: {doc.page_content}")

        # Evaluate relevance
        result = evaluator.evaluate(
            query=query,
            result=doc
        )

        print(f"Relevance Score: {result['score']}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")

    # Clean up
    vectorstore.delete_collection()


def retrieval_precision_example():
    """Example of evaluating retrieval precision at k."""
    print_section("Retrieval Precision Evaluation")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="JavaScript is for web development."),
        Document(page_content="Machine learning is a subset of AI."),
        Document(page_content="Deep learning uses neural networks."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print_result("Retriever Created", "Chroma with k=2")

    # Define ground truth (relevant document indices)
    query = "What is Python?"
    ground_truth = [0]  # Document at index 0 is relevant

    # Retrieve documents
    retrieved_docs = retriever.invoke(query)

    # Calculate precision@k
    retrieved_indices = [i for i, doc in enumerate(documents) if doc in retrieved_docs]
    relevant_retrieved = len(set(retrieved_indices) & set(ground_truth))
    precision = relevant_retrieved / len(retrieved_docs)

    print_result(f"Query: '{query}'", "")
    print(f"Ground Truth (relevant indices): {ground_truth}")
    print(f"Retrieved indices: {retrieved_indices}")
    print(f"Precision@{len(retrieved_docs)}: {precision:.2f}")

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    retrieval_relevance_example()
    retrieval_precision_example()
