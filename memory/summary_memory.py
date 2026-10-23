"""
Summary Memory Example - Summarizing conversation history.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def conversation_summary_example():
    """Example of ConversationSummaryMemory."""
    print_section("Conversation Summary Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create summary memory
    memory = ConversationSummaryMemory(
        llm=llm,
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

    print_result("Chain Created", "LLMChain with ConversationSummaryMemory")

    # Simulate conversation
    messages = [
        "My name is Charlie.",
        "I'm a software engineer with 5 years of experience.",
        "I specialize in Python and JavaScript.",
        "What do you know about me?",
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show summary
    print_result("\nConversation Summary", memory.buffer)


def conversation_summary_buffer_example():
    """Example of ConversationSummaryBufferMemory (summary + recent messages)."""
    print_section("Conversation Summary Buffer Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.memory import ConversationSummaryBufferMemory

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create summary buffer memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100,
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

    print_result("Chain Created", "LLMChain with ConversationSummaryBufferMemory (max_token_limit=100)")

    # Simulate conversation
    messages = [
        "My name is Diana.",
        "I work at a tech company.",
        "I'm learning LangChain.",
        "What is my name?",
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show memory state
    print_result("\nMemory State", "")
    print(f"Summary: {memory.moving_summary_buffer}")
    print(f"Recent messages: {len(memory.chat_memory.messages)}")


if __name__ == "__main__":
    conversation_summary_example()
    conversation_summary_buffer_example()
