"""
Basic Output Parsers Example - Parsing LLM outputs into structured data.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
    DatetimeOutputParser
)
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def structured_output_parser_example():
    """Example of StructuredOutputParser."""
    print_section("Structured Output Parser Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Define response schemas
    response_schemas = [
        ResponseSchema(name="name", description="The name of the person"),
        ResponseSchema(name="age", description="The age of the person as a number"),
        ResponseSchema(name="occupation", description="The occupation of the person"),
    ]

    # Create parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get format instructions
    format_instructions = output_parser.get_format_instructions()
    print_result("Format Instructions", format_instructions)

    # Create prompt with format instructions
    template = """
    Extract information from the following text.
    {format_instructions}

    Text: {text}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create chain
    chain = prompt | llm | output_parser

    # Run the chain
    text = "John is a 30-year-old software engineer who loves Python."
    result = chain.invoke({
        "text": text,
        "format_instructions": format_instructions
    })

    print_result("\nParsed Output", result)
    print(f"Name: {result['name']}")
    print(f"Age: {result['age']}")
    print(f"Occupation: {result['occupation']}")


def comma_separated_parser_example():
    """Example of CommaSeparatedListOutputParser."""
    print_section("Comma Separated List Parser Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create parser
    parser = CommaSeparatedListOutputParser()

    # Get format instructions
    format_instructions = parser.get_format_instructions()
    print_result("Format Instructions", format_instructions)

    # Create prompt
    template = """
    List three programming languages.
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create chain
    chain = prompt | llm | parser

    # Run the chain
    result = chain.invoke({"format_instructions": format_instructions})

    print_result("\nParsed List", result)
    for i, item in enumerate(result, 1):
        print(f"{i}. {item}")


def datetime_parser_example():
    """Example of DatetimeOutputParser."""
    print_section("Datetime Parser Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create parser
    parser = DatetimeOutputParser()

    # Get format instructions
    format_instructions = parser.get_format_instructions()
    print_result("Format Instructions", format_instructions)

    # Create prompt
    template = """
    When was the first moon landing?
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create chain
    chain = prompt | llm | parser

    # Run the chain
    try:
        result = chain.invoke({"format_instructions": format_instructions})
        print_result("\nParsed Datetime", result)
    except Exception as e:
        print(f"Error parsing datetime: {e}")


if __name__ == "__main__":
    structured_output_parser_example()
    comma_separated_parser_example()
    datetime_parser_example()
