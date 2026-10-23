"""
Common utilities and configuration for LangChain examples.
"""
from .config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DATA_DIR,
    VECTOR_STORE_DIR
)
from .utils import (
    print_section,
    print_result,
    measure_time,
    truncate_text,
    print_documents
)

__all__ = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DATA_DIR",
    "VECTOR_STORE_DIR",
    "print_section",
    "print_result",
    "measure_time",
    "truncate_text",
    "print_documents"
]
