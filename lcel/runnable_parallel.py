"""
RunnableParallel Example - Running multiple chains in parallel.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def runnable_parallel_example():
    """Example of running multiple operations in parallel using RunnableParallel."""
    print_section("RunnableParallel Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create prompt templates
    joke_prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    poem_prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")
    fact_prompt = ChatPromptTemplate.from_template("Tell me an interesting fact about {topic}")

    # Create chains
    joke_chain = joke_prompt | llm | StrOutputParser()
    poem_chain = poem_prompt | llm | StrOutputParser()
    fact_chain = fact_prompt | llm | StrOutputParser()

    # Combine chains in parallel
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        poem=poem_chain,
        fact=fact_chain
    )

    print_result("Parallel Chain Created", "joke, poem, and fact chains run in parallel")

    # Invoke the parallel chain
    result = parallel_chain.invoke({"topic": "cats"})

    print_result("\nJoke", result["joke"])
    print_result("\nPoem", result["poem"])
    print_result("\nFact", result["fact"])


def runnable_branch_example():
    """Example of conditional branching with RunnableBranch."""
    print_section("RunnableBranch Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.schema.runnable import RunnableBranch

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Define branches
    def classify_sentiment(text: str) -> str:
        """Simple sentiment classification."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["happy", "good", "great", "love"]):
            return "positive"
        elif any(word in text_lower for word in ["sad", "bad", "terrible", "hate"]):
            return "negative"
        else:
            return "neutral"

    # Create response chains for each sentiment
    positive_chain = ChatPromptTemplate.from_template("Respond enthusiastically to: {input}") | llm | StrOutputParser()
    negative_chain = ChatPromptTemplate.from_template("Respond empathetically to: {input}") | llm | StrOutputParser()
    neutral_chain = ChatPromptTemplate.from_template("Respond neutrally to: {input}") | llm | StrOutputParser()

    # Create branch
    branch = RunnableBranch(
        (lambda x: classify_sentiment(x["input"]) == "positive", positive_chain),
        (lambda x: classify_sentiment(x["input"]) == "negative", negative_chain),
        neutral_chain
    )

    print_result("Branch Created", "Conditional routing based on sentiment")

    # Test with different inputs
    test_inputs = [
        "I love this product!",
        "This is terrible!",
        "It's okay, I guess."
    ]

    for test_input in test_inputs:
        result = branch.invoke({"input": test_input})
        print(f"\nInput: {test_input}")
        print(f"Sentiment: {classify_sentiment(test_input)}")
        print(f"Response: {result}")


if __name__ == "__main__":
    runnable_parallel_example()
    runnable_branch_example()
