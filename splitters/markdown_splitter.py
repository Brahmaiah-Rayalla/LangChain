"""
Markdown Text Splitter Example - Splitting markdown text by headers.
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, truncate_text


def markdown_splitter_example():
    """Example of splitting markdown using MarkdownHeaderTextSplitter."""
    print_section("Markdown Text Splitter Example")

    # Sample markdown text
    markdown_text = """
# LangChain Introduction

LangChain is a framework for developing applications powered by language models.

## Key Features

### Context-Aware
Applications can connect a language model to sources of context.

### Reasoning
Applications can rely on a language model to reason about responses.

## Components

### LangChain Libraries
The library itself for building LLM applications.

### LangChain Templates
A collection of deployable reference architectures.

### LangServe
A deployment platform for LangChain applications.

### LangSmith
A developer platform for building production-grade LLM applications.
"""

    # Create the markdown splitter
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    # Split the markdown
    documents = splitter.split_text(markdown_text)

    print_result("Number of Documents", len(documents))

    for i, doc in enumerate(documents):
        print(f"\n--- Document {i + 1} ---")
        print(f"Content: {truncate_text(doc.page_content, 150)}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    markdown_splitter_example()
