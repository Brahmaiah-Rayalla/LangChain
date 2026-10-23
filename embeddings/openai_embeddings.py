"""
OpenAI Embeddings Example - Using OpenAI's embedding models.
"""
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def openai_embeddings_example():
    """Example of using OpenAI embeddings."""
    print_section("OpenAI Embeddings Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )

    print_result("Embeddings Initialized", "OpenAI text-embedding-ada-002")

    # Embed a single text
    text = "LangChain is a framework for LLM applications."
    embedding = embeddings.embed_query(text)

    print_result("Text", text)
    print_result("Embedding Dimensions", len(embedding))
    print_result("First 5 Values", embedding[:5])

    # Embed multiple texts
    texts = [
        "Python is a programming language.",
        "JavaScript is used for web development.",
        "Java is used for enterprise applications."
    ]

    print_result("\nBatch Embedding", f"{len(texts)} texts")
    embeddings_list = embeddings.embed_documents(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings_list)):
        print(f"\n{i+1}. {text}")
        print(f"   Dimensions: {len(emb)}")


def embedding_similarity():
    """Example of calculating similarity between embeddings."""
    print_section("Embedding Similarity Example")

    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        return

    from numpy import dot
    from numpy.linalg import norm

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Get embeddings for similar and dissimilar texts
    text1 = "Python is a programming language."
    text2 = "Python is used for software development."  # Similar
    text3 = "The weather is nice today."  # Dissimilar

    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    emb3 = embeddings.embed_query(text3)

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    print_result("Text 1", text1)
    print_result("Text 2", text2)
    print_result("Similarity (1-2)", f"{sim_12:.4f}")

    print_result("\nText 3", text3)
    print_result("Similarity (1-3)", f"{sim_13:.4f}")

    print("\nInterpretation: Higher similarity means texts are more semantically similar.")


if __name__ == "__main__":
    openai_embeddings_example()
    embedding_similarity()
