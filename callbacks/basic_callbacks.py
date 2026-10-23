"""
Basic Callbacks Example - Handling events during chain execution.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def stdout_callback_example():
    """Example of using StdOutCallbackHandler."""
    print_section("StdOut Callback Handler Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create prompt
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")

    # Create chain with stdout callback
    chain = prompt | llm | StrOutputParser()

    print_result("Chain Created", "with StdOutCallbackHandler")

    # Invoke with callback
    handler = StdOutCallbackHandler()
    result = chain.invoke({"topic": "programming"}, config={"callbacks": [handler]})

    print_result("\nFinal Result", result)


def custom_callback_example():
    """Example of creating a custom callback handler."""
    print_section("Custom Callback Handler Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.callbacks.base import BaseCallbackHandler

    class CustomCallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            print(f"\n🚀 LLM started with prompts: {prompts}")

        def on_llm_end(self, response, **kwargs):
            print(f"✅ LLM finished. Tokens used: {response.llm_output.get('token_usage', {}).get('total_tokens', 'N/A')}")

        def on_chain_start(self, serialized, inputs, **kwargs):
            print(f"\n🔗 Chain started with inputs: {inputs}")

        def on_chain_end(self, outputs, **kwargs):
            print(f"✅ Chain finished with outputs: {outputs}")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create prompt
    prompt = ChatPromptTemplate.from_template("What is {concept}?")

    # Create chain
    chain = prompt | llm | StrOutputParser()

    print_result("Chain Created", "with CustomCallbackHandler")

    # Invoke with custom callback
    custom_handler = CustomCallbackHandler()
    result = chain.invoke({"concept": "LangChain"}, config={"callbacks": [custom_handler]})

    print_result("\nFinal Result", result)


if __name__ == "__main__":
    stdout_callback_example()
    custom_callback_example()
