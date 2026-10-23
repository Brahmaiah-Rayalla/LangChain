"""
Retrieval QA Chain Example - Question answering with document retrieval.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def retrieval_qa_example():
    """Example of RetrievalQA chain for question answering."""
    print_section("Retrieval QA Chain Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create sample documents
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models. It enables context-aware and reasoning applications."),
        Document(page_content="LangChain provides two main abstractions: Components (modular building blocks) and Chains (sequences of components)."),
        Document(page_content="LangChain supports various use cases including document analysis, chatbots, agents, and code generation."),
        Document(page_content="LangChain integrates with many tools and services including OpenAI, Anthropic, Hugging Face, and various vector stores."),
        Document(page_content="LangChain Templates provide deployable reference architectures for common applications."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print_result("Chain Created", "RetrievalQA with source documents")

    # Ask questions
    questions = [
        "What are the main abstractions in LangChain?",
        "What use cases does LangChain support?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

        result = qa_chain.invoke({"query": question})

        print(f"\nAnswer: {result['result']}")

        print("\nSource Documents:")
        for i, doc in enumerate(result['source_documents']):
            print(f"\n--- Source {i + 1} ---")
            print(doc.page_content)

    # Clean up
    vectorstore.delete_collection()


if __name__ == "__main__":
    retrieval_qa_example()
