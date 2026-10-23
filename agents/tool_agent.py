"""
Tool Agent Example - Using tools with agents.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import ChatPromptTemplate
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def calculator_tool(x: str) -> str:
    """Simple calculator tool that evaluates mathematical expressions."""
    try:
        result = eval(x)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def search_tool(query: str) -> str:
    """Mock search tool."""
    return f"Search results for '{query}': This is a mock search result."


def tool_agent_example():
    """Example of an agent using tools."""
    print_section("Tool Agent Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Useful for mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'."
        ),
        Tool(
            name="Search",
            func=search_tool,
            description="Useful for searching information. Input should be a search query."
        )
    ]

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    print_result("Agent Created", "ZERO_SHOT_REACT_DESCRIPTION agent with Calculator and Search tools")

    # Test the agent
    queries = [
        "What is 25 * 4?",
        "What is the result of 100 / 5?",
        "Search for information about Python programming"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = agent.run(query)
        print(f"\nResult: {result}")


if __name__ == "__main__":
    tool_agent_example()
