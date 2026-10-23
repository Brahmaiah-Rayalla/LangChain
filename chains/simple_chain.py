"""
Simple Chain Example - Basic LLM chain with prompt.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def simple_chain_example():
    """Example of a simple LLM chain."""
    print_section("Simple Chain Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        "Tell me a {adjective} joke about {topic}."
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    print_result("Chain Created", "LLMChain with prompt template")

    # Run the chain
    result = chain.run(adjective="funny", topic="programming")

    print_result("Result", result)


def simple_chain_with_memory():
    """Example of a simple chain with conversation memory."""
    print_section("Simple Chain with Memory")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.memory import ConversationBufferMemory

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create memory
    memory = ConversationBufferMemory()

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ])

    # Create chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    print_result("Chain Created", "LLMChain with ConversationBufferMemory")

    # First interaction
    result1 = chain.run(input="My name is Alice.")
    print_result("\nFirst Interaction", result1)

    # Second interaction (should remember the name)
    result2 = chain.run(input="What is my name?")
    print_result("\nSecond Interaction", result2)


if __name__ == "__main__":
    simple_chain_example()
    simple_chain_with_memory()
