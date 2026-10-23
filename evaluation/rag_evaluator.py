"""
RAG Evaluator Example - Evaluating Retrieval-Augmented Generation systems.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.evaluation import load_evaluator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def rag_evaluator_example():
    """Example of evaluating a RAG system."""
    print_section("RAG Evaluator Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models. It enables context-aware and reasoning applications."),
        Document(page_content="LangChain provides two main abstractions: Components (modular building blocks) and Chains (sequences of components)."),
        Document(page_content="LangChain supports various use cases including document analysis, chatbots, agents, and code generation."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    print_result("RAG Chain Created", "RetrievalQA with Chroma")

    # Load RAG evaluator
    evaluator = load_evaluator("context_qa", llm=llm)

    print_result("Evaluator Loaded", "Context QA Evaluator")

    # Evaluate RAG responses
    test_cases = [
        {
            "question": "What are the main abstractions in LangChain?",
            "context": documents[1].page_content
        },
        {
            "question": "What use cases does LangChain support?",
            "context": documents[2].page_content
        }
    ]

    for test_case in test_cases:
        # Get answer from RAG chain
        answer = qa_chain.run(test_case["question"])

        print(f"\n{'='*60}")
        print(f"Question: {test_case['question']}")
        print(f"Context: {test_case['context'][:100]}...")
        print(f"Answer: {answer}")
        print('='*60)

        # Evaluate answer
        result = evaluator.evaluate_strings(
            prediction=answer,
            reference=test_case["context"],
            input=test_case["question"]
        )

        print(f"Score: {result['score']}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    rag_evaluator_example()
