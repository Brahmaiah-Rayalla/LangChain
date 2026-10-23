"""
RunnableSequence Example - Sequencing operations in LCEL.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def runnable_sequence_example():
    """Example of sequencing operations with RunnablePassthrough."""
    print_section("RunnableSequence Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create a chain that processes input through multiple steps
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question: {question}\n\nContext: {context}"
    )

    # Define a function to generate context
    def generate_context(question: str) -> str:
        return f"This is additional context for the question: {question}"

    # Create chain with RunnablePassthrough
    chain = (
        {
            "context": generate_context,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print_result("Chain Created", "generate_context | prompt | llm | StrOutputParser")

    # Invoke the chain
    result = chain.invoke("What is the capital of France?")

    print_result("Question", "What is the capital of France?")
    print_result("Response", result)


def router_example():
    """Example of routing inputs to different chains."""
    print_section("Router Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from langchain.schema.runnable import RunnableLambda

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create different chains for different tasks
    summarize_chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()
    translate_chain = ChatPromptTemplate.from_template("Translate to Spanish: {text}") | llm | StrOutputParser()
    rewrite_chain = ChatPromptTemplate.from_template("Rewrite more formally: {text}") | llm | StrOutputParser()

    # Define routing logic
    def route_function(inputs):
        task = inputs.get("task", "summarize")
        if task == "summarize":
            return summarize_chain
        elif task == "translate":
            return translate_chain
        elif task == "rewrite":
            return rewrite_chain
        else:
            return summarize_chain

    # Create router chain
    router = RunnableLambda(route_function)

    print_result("Router Created", "Routes to different chains based on task")

    # Test different tasks
    test_cases = [
        {"task": "summarize", "text": "LangChain is a framework for developing applications powered by language models."},
        {"task": "translate", "text": "Hello, how are you?"},
        {"task": "rewrite", "text": "Hey, what's up?"}
    ]

    for test_case in test_cases:
        chain = router.invoke(test_case)
        result = chain.invoke(test_case)
        print(f"\nTask: {test_case['task']}")
        print(f"Input: {test_case['text']}")
        print(f"Output: {result}")


if __name__ == "__main__":
    runnable_sequence_example()
    router_example()
