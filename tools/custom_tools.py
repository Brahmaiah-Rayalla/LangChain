"""
Custom Tools Example - Creating custom tools for agents.
"""
from langchain.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result


def simple_tool_example():
    """Example of creating a simple tool using @tool decorator."""
    print_section("Simple Tool Example")

    @tool
    def calculator(expression: str) -> str:
        """A simple calculator that evaluates mathematical expressions."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    print_result("Tool Created", "Calculator tool")

    # Use the tool
    result = calculator.invoke("2 + 2")
    print_result("Invoked with '2 + 2'", result)

    result = calculator.invoke("10 * 5")
    print_result("Invoked with '10 * 5'", result)


def structured_tool_example():
    """Example of creating a structured tool with input schema."""
    print_section("Structured Tool Example")

    # Define input schema
    class WeatherInput(BaseModel):
        location: str = Field(description="The city name")
        unit: str = Field(default="celsius", description="Temperature unit (celsius or fahrenheit)")

    # Define tool function
    def get_weather(input: WeatherInput) -> str:
        """Get the weather for a specific location."""
        return f"The weather in {input.location} is 25°{input.unit} and sunny."

    # Create structured tool
    weather_tool = StructuredTool.from_function(
        func=get_weather,
        name="Weather",
        description="Get the weather for a specific location",
        args_schema=WeatherInput
    )

    print_result("Tool Created", "Weather tool with structured input")

    # Use the tool
    result = weather_tool.invoke({"location": "New York", "unit": "celsius"})
    print_result("Invoked with location='New York'", result)


def multiple_tools_example():
    """Example of creating multiple tools."""
    print_section("Multiple Tools Example")

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Search results for: {query}"

    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            return str(eval(expression))
        except:
            return "Error in calculation"

    @tool
    def get_time() -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [search, calculate, get_time]

    print_result("Tools Created", f"{len(tools)} tools")

    for tool in tools:
        print(f"\n- {tool.name}: {tool.description}")

        # Test each tool
        if tool.name == "search":
            result = tool.invoke("Python programming")
        elif tool.name == "calculate":
            result = tool.invoke("5 * 10")
        elif tool.name == "get_time":
            result = tool.invoke("")

        print(f"  Result: {result}")


if __name__ == "__main__":
    simple_tool_example()
    structured_tool_example()
    multiple_tools_example()
