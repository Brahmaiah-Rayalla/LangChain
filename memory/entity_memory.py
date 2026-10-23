"""
Entity Memory Example - Extracting and remembering entities from conversation.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def conversation_entity_example():
    """Example of ConversationEntityMemory."""
    print_section("Conversation Entity Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create entity memory
    memory = ConversationEntityMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with a good memory for details about people."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    print_result("Chain Created", "LLMChain with ConversationEntityMemory")

    # Simulate conversation with entities
    messages = [
        "My name is Eve and I work at Google.",
        "I have a dog named Max.",
        "What's my dog's name?",
        "Where do I work?",
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show extracted entities
    print_result("\nExtracted Entities", "")
    print(memory.entity_store.store)


def conversation_kg_memory_example():
    """Example of ConversationKGMemory (Knowledge Graph memory)."""
    print_section("Conversation Knowledge Graph Memory Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.memory import ConversationKGMemory

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create KG memory
    memory = ConversationKGMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who remembers relationships between entities."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    print_result("Chain Created", "LLMChain with ConversationKGMemory")

    # Simulate conversation
    messages = [
        "Frank is a software engineer at Microsoft.",
        "Sarah is Frank's manager.",
        "What do you know about Frank?",
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = chain.run(input=message)
        print(f"Assistant: {response}")

    # Show knowledge graph
    print_result("\nKnowledge Graph", "")
    print(memory.kg.get_triples())


if __name__ == "__main__":
    conversation_entity_example()
    conversation_kg_memory_example()
