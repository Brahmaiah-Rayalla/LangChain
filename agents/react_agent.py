"""
ReAct Agent Example - Reasoning and Acting agent.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(location: str) -> str:
    """Mock weather tool."""
    return f"The weather in {location} is sunny with a temperature of 25°C."


def react_agent_example():
    """Example of a ReAct agent."""
    print_section("ReAct Agent Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Define tools
    tools = [
        Tool(
            name="CurrentTime",
            func=get_current_time,
            description="Get the current date and time."
        ),
        Tool(
            name="Weather",
            func=get_weather,
            description="Get the weather for a specific location. Input should be the location name."
        )
    ]

    # Get the ReAct prompt from LangChain hub
    # Note: This requires internet access to fetch from hub
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception as e:
        print(f"⚠️  Could not fetch prompt from hub: {e}")
        print("Using a simple prompt instead.")

        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        )

    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    print_result("ReAct Agent Created", "Reasoning and Acting agent with Time and Weather tools")

    # Test the agent
    queries = [
        "What time is it now?",
        "What's the weather in Tokyo?",
        "What's the current time and weather in New York?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        try:
            result = agent_executor.invoke({"input": query})
            print(f"\nResult: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    react_agent_example()
