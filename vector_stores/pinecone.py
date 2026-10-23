"""
Pinecone Vector Store Example - Using Pinecone as a cloud vector database.
"""
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def pinecone_example():
    """Example of using Pinecone vector store."""
    print_section("Pinecone Vector Store Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("⚠️  PINECONE_API_KEY not found. Please set it in your environment or .env file.")
        print("This example requires a Pinecone account and API key.")
        return

    try:
        import pinecone
    except ImportError:
        print("⚠️  pinecone-client not installed. Install it with: pip install pinecone-client")
        return

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language.", metadata={"id": 1}),
        Document(page_content="JavaScript is used for web development.", metadata={"id": 2}),
        Document(page_content="Machine learning is a subset of AI.", metadata={"id": 3}),
    ]

    # Create or connect to index
    index_name = "langchain-example"
    if index_name not in pinecone.list_indexes():
        print(f"Creating new Pinecone index: {index_name}")
        pinecone.create_index(index_name, dimension=1536, metric="cosine")

    print_result("Pinecone Index", index_name)

    # Create vector store
    vectorstore = Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )

    print_result("Vector Store Created", "Pinecone with 3 documents")

    # Similarity search
    query = "What programming language is used for web development?"
    results = vectorstore.similarity_search(query, k=2)

    print_result(f"Similarity Search: '{query}'", "")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

    # Clean up (optional - comment out to keep the index)
    # pinecone.delete_index(index_name)
    # print_result("Index Deleted", index_name)


if __name__ == "__main__":
    pinecone_example()
