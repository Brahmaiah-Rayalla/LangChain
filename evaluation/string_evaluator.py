"""
String Evaluator Example - Evaluating string outputs.
"""
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def string_similarity_example():
    """Example of evaluating string similarity."""
    print_section("String Similarity Evaluation")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Load string similarity evaluator
    evaluator = load_evaluator("labeled_criteria", criteria="conciseness")

    print_result("Evaluator Loaded", "Labeled Criteria - Conciseness")

    # Evaluate different responses
    predictions = [
        "LangChain is a framework for developing applications powered by language models.",
        "LangChain is a framework.",
        "LangChain helps you build LLM-powered applications by providing components and chains for context-aware and reasoning applications."
    ]

    reference = "LangChain is a framework for LLM applications."

    for i, prediction in enumerate(predictions):
        result = evaluator.evaluate_strings(
            prediction=prediction,
            reference=reference,
            input="What is LangChain?"
        )

        print(f"\n--- Prediction {i + 1} ---")
        print(f"Text: {prediction}")
        print(f"Score: {result['score']}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")


def correctness_evaluator_example():
    """Example of evaluating correctness using LLM."""
    print_section("Correctness Evaluation")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Load correctness evaluator
    evaluator = load_evaluator("correctness", llm=llm)

    print_result("Evaluator Loaded", "Correctness Evaluator")

    # Evaluate responses
    test_cases = [
        {
            "input": "What is 2 + 2?",
            "prediction": "2 + 2 equals 4.",
            "reference": "4"
        },
        {
            "input": "What is 2 + 2?",
            "prediction": "2 + 2 equals 5.",
            "reference": "4"
        }
    ]

    for test_case in test_cases:
        result = evaluator.evaluate_strings(
            prediction=test_case["prediction"],
            reference=test_case["reference"],
            input=test_case["input"]
        )

        print(f"\nInput: {test_case['input']}")
        print(f"Prediction: {test_case['prediction']}")
        print(f"Reference: {test_case['reference']}")
        print(f"Score: {result['score']}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")


if __name__ == "__main__":
    string_similarity_example()
    correctness_evaluator_example()
