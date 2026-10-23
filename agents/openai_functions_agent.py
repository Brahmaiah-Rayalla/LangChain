"""
OpenAI Functions Agent Example - Using OpenAI function calling with agents.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_word_length(word: str) -> str:
    """Get the length of a word."""
    return str(len(word))


@tool
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def openai_functions_agent_example():
    """Example of an OpenAI Functions agent."""
    print_section("OpenAI Functions Agent Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Define tools using the @tool decorator
    tools = [calculator, get_word_length, reverse_string]

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the available tools to answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create OpenAI Functions agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    print_result("OpenAI Functions Agent Created", "Agent with Calculator, Word Length, and Reverse String tools")

    # Test the agent
    queries = [
        "What is 15 * 7?",
        "What is the length of the word 'LangChain'?",
        "Reverse the string 'hello world'",
        "Calculate 100 / 4 and then reverse the result"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = agent_executor.invoke({"input": query})
        print(f"\nResult: {result['output']}")


if __name__ == "__main__":
    openai_functions_agent_example()
