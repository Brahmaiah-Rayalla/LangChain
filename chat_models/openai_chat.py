"""
OpenAI Chat Model Example - Using OpenAI's chat models.
"""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def openai_chat_example():
    """Example of using OpenAI Chat model."""
    print_section("OpenAI Chat Model Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize Chat model
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    print_result("Chat Model Initialized", "OpenAI GPT-3.5-turbo")

    # Simple message
    message = HumanMessage(content="What is LangChain?")
    result = chat.invoke([message])

    print_result("Message", message.content)
    print_result("Response", result.content)


def chat_with_system_message():
    """Example of chat with system message."""
    print_section("Chat with System Message")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create messages
    messages = [
        SystemMessage(content="You are a helpful programming assistant."),
        HumanMessage(content="What is Python?")
    ]

    result = chat.invoke(messages)

    print_result("System Message", "You are a helpful programming assistant.")
    print_result("Human Message", "What is Python?")
    print_result("AI Response", result.content)


def chat_conversation():
    """Example of multi-turn conversation."""
    print_section("Multi-turn Conversation")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Conversation history
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="My name is Alice."),
        AIMessage(content="Hello Alice! How can I help you today?"),
        HumanMessage(content="What is my name?")
    ]

    result = chat.invoke(messages)

    print_result("Conversation History", f"{len(messages)} messages")
    print_result("AI Response", result.content)


def chat_streaming():
    """Example of streaming chat responses."""
    print_section("Chat Streaming Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )

    print("Streaming response:")
    print("-" * 60)

    for chunk in chat.stream([HumanMessage(content="Tell me a short joke.")]):
        print(chunk.content, end="", flush=True)

    print("\n" + "-" * 60)


if __name__ == "__main__":
    openai_chat_example()
    chat_with_system_message()
    chat_conversation()
    chat_streaming()
