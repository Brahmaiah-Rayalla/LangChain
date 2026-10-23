"""
Common utility functions for LangChain examples.
"""
from typing import List, Any
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(label: str, result: Any):
    """Print a formatted result."""
    print(f"\n{label}:")
    print("-" * 40)
    print(result)
    print("-" * 40)


def measure_time(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n⏱️  Execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_documents(docs: List[Any], max_docs: int = 3):
    """Print document contents in a readable format."""
    for i, doc in enumerate(docs[:max_docs]):
        print(f"\n--- Document {i + 1} ---")
        print(f"Content: {truncate_text(doc.page_content)}")
        if hasattr(doc, 'metadata'):
            print(f"Metadata: {doc.metadata}")
