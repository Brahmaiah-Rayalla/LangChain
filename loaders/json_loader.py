"""
JSON Loader Example - Loading JSON files into LangChain documents.
"""
from langchain_community.document_loaders import JSONLoader
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents


def json_loader_example():
    """Example of loading JSON files using JSONLoader."""
    print_section("JSON Loader Example")

    # Create a sample JSON file
    sample_file = "sample_data.json"
    data = [
        {"id": 1, "name": "Product A", "price": 100, "description": "A great product"},
        {"id": 2, "name": "Product B", "price": 200, "description": "An even better product"},
        {"id": 3, "name": "Product C", "price": 150, "description": "A mid-range product"}
    ]

    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Load the JSON file
    # jq_schema specifies which part of the JSON to extract
    loader = JSONLoader(
        file_path=sample_file,
        jq_schema=".[]",
        text_content=False
    )
    documents = loader.load()

    print_result("Number of Items Loaded", len(documents))
    print_result("First Item", documents[0].page_content)
    print_result("Metadata", documents[0].metadata)

    print_documents(documents, max_docs=2)

    # Clean up
    os.remove(sample_file)


if __name__ == "__main__":
    json_loader_example()
