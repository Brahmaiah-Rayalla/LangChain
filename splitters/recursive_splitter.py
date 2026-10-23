"""
Recursive Character Text Splitter Example - Splitting text recursively by different separators.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, truncate_text


def recursive_splitter_example():
    """Example of splitting text using RecursiveCharacterTextSplitter."""
    print_section("Recursive Character Text Splitter Example")

    # Sample text with multiple paragraphs
    text = """
    LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason.

    The framework consists of several parts:
    - LangChain Libraries: The library itself
    - LangChain Templates: A collection of deployable reference architectures
    - LangServe: A deployment platform
    - LangSmith: A developer platform

    LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.
    """

    # Create the recursive splitter
    # It tries different separators in order: ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )

    # Split the text
    chunks = splitter.split_text(text)

    print_result("Number of Chunks", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} characters) ---")
        print(truncate_text(chunk, 120))


if __name__ == "__main__":
    recursive_splitter_example()
