"""
Web Loader Example - Loading web pages into LangChain documents.
"""
from langchain_community.document_loaders import WebBaseLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents


def web_loader_example():
    """Example of loading web pages using WebBaseLoader."""
    print_section("Web Loader Example")

    # Load a web page
    url = "https://python.langchain.com/docs/get_started/introduction"
    loader = WebBaseLoader(url)

    try:
        documents = loader.load()

        print_result("Number of Documents Loaded", len(documents))
        print_result("Source URL", documents[0].metadata.get("source"))
        print_documents(documents)

    except Exception as e:
        print(f"⚠️  Error loading web page: {e}")
        print("Make sure you have internet access and the URL is accessible.")


if __name__ == "__main__":
    web_loader_example()
