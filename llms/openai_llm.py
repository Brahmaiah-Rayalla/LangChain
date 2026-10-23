"""
OpenAI LLM Example - Using OpenAI's language models.
"""
from langchain_openai import OpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def openai_llm_example():
    """Example of using OpenAI LLM."""
    print_section("OpenAI LLM Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize OpenAI LLM
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    print_result("LLM Initialized", "OpenAI GPT-3.5-turbo-instruct")

    # Simple invocation
    prompt = "What is LangChain?"
    result = llm.invoke(prompt)

    print_result("Prompt", prompt)
    print_result("Response", result)

    # Batch invocation
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Java?"
    ]

    print_result("\nBatch Invocation", f"{len(prompts)} prompts")
    results = llm.batch(prompts)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Response: {result[:100]}...")


def openai_llm_with_params():
    """Example of OpenAI LLM with various parameters."""
    print_section("OpenAI LLM with Parameters")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize with parameters
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5,      # Controls randomness (0-1)
        max_tokens=100,       # Maximum tokens in response
        top_p=0.9,           # Nucleus sampling
        frequency_penalty=0.5,  # Penalize frequent tokens
        presence_penalty=0.5   # Penalize repetition
    )

    print_result("LLM Parameters", "temperature=0.5, max_tokens=100, top_p=0.9")

    result = llm.invoke("Write a short poem about programming.")
    print_result("Response", result)


if __name__ == "__main__":
    openai_llm_example()
    openai_llm_with_params()
