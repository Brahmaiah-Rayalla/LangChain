"""
Sequential Chain Example - Running multiple chains in sequence.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def simple_sequential_chain_example():
    """Example of SimpleSequentialChain."""
    print_section("Simple Sequential Chain Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # First chain: Generate a story outline
    outline_prompt = ChatPromptTemplate.from_template(
        "Write a brief outline for a {genre} story about {topic}."
    )
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt)

    # Second chain: Write the story based on the outline
    story_prompt = ChatPromptTemplate.from_template(
        "Write a short story based on this outline:\n{outline}"
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)

    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[outline_chain, story_chain],
        verbose=True
    )

    print_result("Chain Created", "Outline Chain -> Story Chain")

    # Run the chain
    result = overall_chain.run("A science fiction story about time travel")

    print_result("\nFinal Story", result)


def sequential_chain_example():
    """Example of SequentialChain with multiple inputs/outputs."""
    print_section("Sequential Chain with Multiple Inputs/Outputs")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.chains import SequentialChain

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Chain 1: Generate a product name
    name_prompt = ChatPromptTemplate.from_template(
        "Suggest a creative name for a {product_type} company."
    )
    name_chain = LLMChain(
        llm=llm,
        prompt=name_prompt,
        output_key="company_name"
    )

    # Chain 2: Generate a slogan
    slogan_prompt = ChatPromptTemplate.from_template(
        "Write a catchy slogan for a company called {company_name} that sells {product_type}."
    )
    slogan_chain = LLMChain(
        llm=llm,
        prompt=slogan_prompt,
        output_key="slogan"
    )

    # Chain 3: Write a marketing description
    desc_prompt = ChatPromptTemplate.from_template(
        "Write a marketing description for {company_name}: {slogan}"
    )
    desc_chain = LLMChain(
        llm=llm,
        prompt=desc_prompt,
        output_key="description"
    )

    # Combine chains
    overall_chain = SequentialChain(
        chains=[name_chain, slogan_chain, desc_chain],
        input_variables=["product_type"],
        output_variables=["company_name", "slogan", "description"],
        verbose=True
    )

    print_result("Chain Created", "Name Chain -> Slogan Chain -> Description Chain")

    # Run the chain
    result = overall_chain.run("coffee")

    print_result("\nCompany Name", result["company_name"])
    print_result("\nSlogan", result["slogan"])
    print_result("\nDescription", result["description"])


if __name__ == "__main__":
    simple_sequential_chain_example()
    sequential_chain_example()
