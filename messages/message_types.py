"""
Message Types Example - Different message types in LangChain.
"""
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result


def message_types_example():
    """Example of different message types."""
    print_section("Message Types Example")

    # Human message
    human_msg = HumanMessage(content="What is LangChain?")
    print_result("HumanMessage", human_msg.content)

    # AI message
    ai_msg = AIMessage(content="LangChain is a framework for LLM applications.")
    print_result("AIMessage", ai_msg.content)

    # System message
    system_msg = SystemMessage(content="You are a helpful assistant.")
    print_result("SystemMessage", system_msg.content)

    # Function message (for function calling)
    function_msg = FunctionMessage(
        name="calculator",
        content="The result is 42"
    )
    print_result("FunctionMessage", f"Name: {function_msg.name}, Content: {function_msg.content}")

    # Tool message (for tool responses)
    tool_msg = ToolMessage(
        tool_call_id="123",
        content="Weather is sunny"
    )
    print_result("ToolMessage", f"Tool Call ID: {tool_msg.tool_call_id}, Content: {tool_msg.content}")


def message_with_metadata():
    """Example of messages with metadata."""
    print_section("Messages with Metadata")

    human_msg = HumanMessage(
        content="Hello!",
        metadata={"user_id": "123", "timestamp": "2024-01-01"}
    )

    print_result("Message Content", human_msg.content)
    print_result("Message Metadata", human_msg.metadata)


def message_sequence():
    """Example of message sequence for conversation."""
    print_section("Message Sequence Example")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="My name is Alice."),
        AIMessage(content="Hello Alice! How can I help you?"),
        HumanMessage(content="What is my name?"),
    ]

    print_result("Conversation Sequence", f"{len(messages)} messages")

    for i, msg in enumerate(messages, 1):
        msg_type = msg.__class__.__name__
        print(f"\n{i}. {msg_type}: {msg.content[:50]}...")


if __name__ == "__main__":
    message_types_example()
    message_with_metadata()
    message_sequence()
