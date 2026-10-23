"""
Code Text Splitter Example - Splitting code by programming language syntax.
"""
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, truncate_text


def python_code_splitter_example():
    """Example of splitting Python code using PythonCodeTextSplitter."""
    print_section("Python Code Splitter Example")

    # Sample Python code
    python_code = """
def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    for i in range(10):
        print(f"Fibonacci({i}) = {calculate_fibonacci(i)}")

if __name__ == "__main__":
    main()
"""

    # Create the Python code splitter
    splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)

    # Split the code
    chunks = splitter.split_text(python_code)

    print_result("Number of Chunks", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} characters) ---")
        print(truncate_text(chunk, 150))


def generic_code_splitter_example():
    """Example of splitting code using RecursiveCharacterTextSplitter with language support."""
    print_section("Generic Code Splitter Example")

    # Sample JavaScript code
    js_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
"""

    # Create the code splitter for JavaScript
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=100,
        chunk_overlap=0
    )

    # Split the code
    chunks = splitter.split_text(js_code)

    print_result("Number of Chunks", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} characters) ---")
        print(truncate_text(chunk, 150))


if __name__ == "__main__":
    python_code_splitter_example()
    generic_code_splitter_example()
