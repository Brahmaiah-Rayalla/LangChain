"""
Few-shot Prompting Example - Using examples to guide model behavior.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def few_shot_prompt_example():
    """Example of few-shot prompting with examples."""
    print_section("Few-shot Prompting Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Define examples
    examples = [
        {
            "input": "happy",
            "output": "sad"
        },
        {
            "input": "big",
            "output": "small"
        },
        {
            "input": "hot",
            "output": "cold"
        }
    ]

    # Define the example template
    example_template = """
Input: {input}
Output: {output}
"""

    # Create the few-shot prompt template
    example_prompt = PromptTemplate(
        template=example_template,
        input_variables=["input", "output"]
    )

    # Create the few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of the following word:",
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
        example_separator="\n"
    )

    print_result("Few-shot Prompt Created", "3 examples for antonym task")

    # Format the prompt
    formatted_prompt = few_shot_prompt.format(input="fast")
    print_result("Formatted Prompt", formatted_prompt)


def few_shot_chat_example():
    """Example of few-shot prompting in chat format."""
    print_section("Few-shot Chat Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Define examples
    examples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Tokyo."
        },
        {
            "question": "What is the capital of Brazil?",
            "answer": "The capital of Brazil is Brasília."
        }
    ]

    # Define the example template
    example_prompt = PromptTemplate(
        template="Question: {question}\nAnswer: {answer}",
        input_variables=["question", "answer"]
    )

    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Answer the following questions:",
        suffix="Question: {input}\nAnswer:",
        input_variables=["input"]
    )

    print_result("Few-shot Chat Prompt Created", "3 examples for QA task")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create chain
    chain = few_shot_prompt | llm

    # Test
    result = chain.invoke({"input": "What is the capital of Germany?"})
    print_result("Question", "What is the capital of Germany?")
    print_result("Answer", result.content)


def dynamic_few_shot():
    """Example of dynamic few-shot selection."""
    print_section("Dynamic Few-shot Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Create a larger pool of examples
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "big", "output": "small"},
        {"input": "hot", "output": "cold"},
        {"input": "light", "output": "dark"},
        {"input": "new", "output": "old"},
        {"input": "rich", "output": "poor"},
    ]

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        k=2  # Select top 2 most similar examples
    )

    print_result("Example Selector Created", "Selects 2 most similar examples")

    # Select examples dynamically
    selected_examples = example_selector.select_examples({"input": "bright"})

    print_result("Input", "bright")
    print_result("Selected Examples", f"{len(selected_examples)} examples")

    for i, example in enumerate(selected_examples, 1):
        print(f"{i}. {example}")


if __name__ == "__main__":
    few_shot_prompt_example()
    few_shot_chat_example()
    dynamic_few_shot()
