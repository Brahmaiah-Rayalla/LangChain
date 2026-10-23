"""
Text Loader Example - Loading text files into LangChain documents.
"""
from langchain_community.document_loaders import TextLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, print_documents


def text_loader_example():
    """Example of loading text files using TextLoader."""
    print_section("Text Loader Example")

    # Create a sample text file
    sample_file = "sample_text.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""LangChain is a framework for developing applications powered by language models.

It enables applications that:
- Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- Reason: rely on a language model to reason (about what to answer, how to answer, etc.)

This framework consists of several parts:
- LangChain Libraries: The library itself
- LangChain Templates: A collection of deployable reference architectures
- LangServe: A deployment platform
- LangSmith: A developer platform
""")

    # Load the text file
    loader = TextLoader(sample_file, encoding="utf-8")
    documents = loader.load()

    print_result("Loaded Documents", documents)
    print_documents(documents)

    # Clean up
    os.remove(sample_file)


if __name__ == "__main__":
    text_loader_example()
