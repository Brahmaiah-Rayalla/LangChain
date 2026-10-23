# Quick Start Guide

Get started with LangChain examples in 5 minutes.

## 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

## 3. Run Your First Example

```bash
# Run a simple chain example
python chains/simple_chain.py

# Run a document loader example
python loaders/text_loader.py

# Run a text splitter example
python splitters/recursive_splitter.py
```

## 4. Explore Concepts

### Beginner
- Start with `loaders/` to understand data loading
- Move to `splitters/` to learn about chunking
- Try `prompts/` to understand prompt templates
- Explore `llms/` and `chat_models/` for model usage

### Intermediate
- Explore `chains/` for sequencing operations
- Learn `memory/` for conversation state
- Try `vector_stores/` for semantic search
- Use `embeddings/` for vector representations
- Build `tools/` for agent functionality

### Advanced
- Build `agents/` that can use tools
- Implement `retrievers/` for RAG systems
- Use `evaluation/` to measure performance
- Try `langgraph/` for stateful applications
- Explore `hybrid_search/` and `reranking/` for advanced retrieval

## 5. Common Patterns

### Loading and Splitting Documents
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)
```

### Creating a Simple Chain
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "LangChain"})
```

### Using Chat Models
```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")
result = chat.invoke([HumanMessage(content="Hello!")])
```

### Creating Tools
```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))
```

## 6. Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the root directory
- Verify your API key is correct
- Check that you have sufficient credits

### Import Errors
- Make sure you're in the repository root
- Install missing packages: `pip install <package>`
- Check Python version (requires 3.8+)

### Example Errors
- Some examples require specific files or API keys
- Check the example's error messages for specific requirements
- Modify examples to use your own data if needed

## 7. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore each concept folder for in-depth examples
- Check the `common/` folder for shared utilities
- Modify examples to fit your use case

## 8. Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
- [LangChain API Reference](https://python.langchain.com/docs/reference/)

Happy coding! 🚀
