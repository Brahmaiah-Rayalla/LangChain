"""
PDF Loader Example - Loading PDF files into LangChain documents.
"""
from langchain_community.document_loaders import PyPDFLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents


def pdf_loader_example():
    """Example of loading PDF files using PyPDFLoader."""
    print_section("PDF Loader Example")

    # Note: This example requires a PDF file to exist
    # You can provide your own PDF file path
    pdf_path = "sample.pdf"

    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF file not found at: {pdf_path}")
        print("Please provide a valid PDF file path to test this example.")
        return

    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print_result("Number of Pages Loaded", len(documents))
    print_result("First Page Content", documents[0].page_content)
    print_result("Metadata", documents[0].metadata)

    print_documents(documents, max_docs=2)


if __name__ == "__main__":
    pdf_loader_example()
