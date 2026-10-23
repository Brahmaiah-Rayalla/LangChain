"""
Character Text Splitter Example - Splitting text by character count.
"""
from langchain_text_splitters import CharacterTextSplitter
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, truncate_text


def character_splitter_example():
    """Example of splitting text using CharacterTextSplitter."""
    print_section("Character Text Splitter Example")

    # Sample text
    text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason.
    The framework consists of several parts including LangChain Libraries,
    LangChain Templates, LangServe, and LangSmith.

    This is a longer paragraph that should be split into multiple chunks
    when the character limit is exceeded. The character splitter will break
    the text into smaller pieces based on the specified chunk size and overlap.
    """

    # Create the splitter
    splitter = CharacterTextSplitter(
        separator="\n",
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
    character_splitter_example()
