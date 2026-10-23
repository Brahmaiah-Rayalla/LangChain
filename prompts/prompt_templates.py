"""
Prompt Templates Example - Creating reusable prompt templates.
"""
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def prompt_template_example():
    """Example of basic PromptTemplate."""
    print_section("Prompt Template Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create a simple prompt template
    template = "Tell me a {adjective} joke about {topic}."
    prompt = PromptTemplate(
        template=template,
        input_variables=["adjective", "topic"]
    )

    print_result("Prompt Template Created", template)

    # Format the prompt
    formatted_prompt = prompt.format(adjective="funny", topic="programming")
    print_result("Formatted Prompt", formatted_prompt)


def chat_prompt_template_example():
    """Example of ChatPromptTemplate."""
    print_section("Chat Prompt Template Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful {role} assistant."),
        ("human", "{input}")
    ])

    print_result("Chat Prompt Template Created", "System + Human messages")

    # Format the prompt
    formatted = prompt.format_messages(role="technical", input="What is Python?")
    print_result("Formatted Messages", f"{len(formatted)} messages")

    for msg in formatted:
        print(f"\n{msg.type}: {msg.content}")


def partial_prompt_example():
    """Example of partial prompt templates."""
    print_section("Partial Prompt Template Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create a prompt template
    template = "Explain {concept} in {style} language."
    prompt = PromptTemplate(
        template=template,
        input_variables=["concept", "style"]
    )

    # Partial with specific style
    simple_prompt = prompt.partial(style="simple")
    print_result("Partial Prompt Created", "style='simple'")

    # Format the partial prompt
    formatted = simple_prompt.format(concept="machine learning")
    print_result("Formatted Prompt", formatted)


def composition_example():
    """Example of composing multiple prompt templates."""
    print_section("Prompt Composition Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create individual templates
    system_template = "You are a {role} assistant."
    human_template = "Help me with {task}."

    # Compose them
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    print_result("Composed Prompt", "System + Human templates")

    # Format
    formatted = prompt.format_messages(role="technical", task="debugging code")
    print_result("Formatted Messages", f"{len(formatted)} messages")


if __name__ == "__main__":
    prompt_template_example()
    chat_prompt_template_example()
    partial_prompt_example()
    composition_example()
