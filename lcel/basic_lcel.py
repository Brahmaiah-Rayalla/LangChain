"""
Basic LCEL Example - Introduction to LangChain Expression Language.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def basic_lcel_example():
    """Example of basic LCEL chain composition."""
    print_section("Basic LCEL Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])

    # Create a simple LCEL chain using the pipe operator (|)
    chain = prompt | llm | StrOutputParser()

    print_result("Chain Created", "prompt | llm | StrOutputParser")

    # Invoke the chain
    query = "What is LangChain Expression Language?"
    result = chain.invoke({"input": query})

    print_result("Query", query)
    print_result("Response", result)


def custom_function_lcel():
    """Example of LCEL with custom functions."""
    print_section("LCEL with Custom Functions")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Define a custom function
    def uppercase(text: str) -> str:
        return text.upper()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create chain with custom function
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | llm | StrOutputParser() | uppercase

    print_result("Chain Created", "prompt | llm | StrOutputParser() | uppercase")

    # Invoke the chain
    result = chain.invoke({"topic": "programming"})

    print_result("Result", result)


if __name__ == "__main__":
    basic_lcel_example()
    custom_function_lcel()
