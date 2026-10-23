"""
Conversation Buffer Memory Example - Storing conversation history.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def conversation_buffer_example():
    """Example of ConversationBufferMemory."""
    print_section("Conversation Buffer Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    print_result("Chain Created", "LLMChain with ConversationBufferMemory")

    # Simulate conversation
    messages = [
        "My name is Alice.",
        "What is my name?",
        "I like Python programming.",
        "What programming language do I like?"
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show memory state
    print_result("\nMemory State", "")
    print(memory.chat_memory.messages)


def conversation_buffer_window_example():
    """Example of ConversationBufferWindowMemory (limited window)."""
    print_section("Conversation Buffer Window Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.memory import ConversationBufferWindowMemory

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create memory with window size of 2 (keeps last 2 exchanges)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=2,
        return_messages=True
    )

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    print_result("Chain Created", "LLMChain with ConversationBufferWindowMemory (k=2)")

    # Simulate conversation
    messages = [
        "My name is Bob.",
        "I work as a developer.",
        "I live in New York.",
        "What is my name?",  # This might not be remembered due to window size
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show memory state
    print_result("\nMemory State (Last 2 exchanges)", "")
    print(memory.chat_memory.messages)


if __name__ == "__main__":
    conversation_buffer_example()
    conversation_buffer_window_example()
