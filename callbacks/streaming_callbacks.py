"""
Streaming Callbacks Example - Streaming outputs in real-time.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def streaming_callback_example():
    """Example of streaming LLM outputs."""
    print_section("Streaming Callback Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM with streaming enabled
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template("Write a short story about {topic}.")

    # Create chain
    chain = prompt | llm | StrOutputParser()

    print_result("Chain Created", "with streaming enabled")

    # Stream the output
    print("\nStreaming output:")
    print("-" * 60)

    for chunk in chain.stream({"topic": "a robot learning to love"}):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60)


def custom_streaming_handler():
    """Example of custom streaming handler."""
    print_section("Custom Streaming Handler Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.callbacks.base import BaseCallbackHandler

    class StreamingHandler(BaseCallbackHandler):
        def __init__(self):
            self.tokens = []

        def on_llm_new_token(self, token, **kwargs):
            self.tokens.append(token)
            print(token, end="", flush=True)

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template("Explain {concept} in simple terms.")

    # Create chain
    chain = prompt | llm | StrOutputParser()

    print_result("Chain Created", "with CustomStreamingHandler")

    # Stream with custom handler
    handler = StreamingHandler()
    print("\nStreaming with custom handler:")
    print("-" * 60)

    result = chain.invoke({"concept": "machine learning"}, config={"callbacks": [handler]})

    print("\n" + "-" * 60)
    print(f"\nTotal tokens: {len(handler.tokens)}")


if __name__ == "__main__":
    streaming_callback_example()
    custom_streaming_handler()
