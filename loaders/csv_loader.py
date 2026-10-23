"""
CSV Loader Example - Loading CSV files into LangChain documents.
"""
from langchain_community.document_loaders import CSVLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents


def csv_loader_example():
    """Example of loading CSV files using CSVLoader."""
    print_section("CSV Loader Example")

    # Create a sample CSV file
    sample_file = "sample_data.csv"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""name,age,city
John,25,New York
Jane,30,San Francisco
Bob,35,Chicago
Alice,28,Boston
""")

    # Load the CSV file
    loader = CSVLoader(file_path=sample_file)
    documents = loader.load()

    print_result("Number of Rows Loaded", len(documents))
    print_result("First Row", documents[0].page_content)
    print_result("Metadata", documents[0].metadata)

    print_documents(documents, max_docs=3)

    # Clean up
    os.remove(sample_file)


if __name__ == "__main__":
    csv_loader_example()
