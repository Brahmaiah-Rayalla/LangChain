"""
Anthropic LLM Example - Using Anthropic's Claude models.
"""
from langchain_anthropic import AnthropicLLM
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, ANTHROPIC_API_KEY


def anthropic_llm_example():
    """Example of using Anthropic Claude LLM."""
    print_section("Anthropic LLM Example")

    if not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not found. Please set it in your environment or .env file.")
        return

    try:
        # Initialize Anthropic LLM
        llm = AnthropicLLM(
            model="claude-3-haiku-20240307",
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.7
        )

        print_result("LLM Initialized", "Anthropic Claude-3 Haiku")

        # Simple invocation
        prompt = "What is LangChain?"
        result = llm.invoke(prompt)

        print_result("Prompt", prompt)
        print_result("Response", result)

    except ImportError:
        print("⚠️  langchain-anthropic not installed. Install it with: pip install langchain-anthropic")
    except Exception as e:
        print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    anthropic_llm_example()
