# LangChain Learning Repository

A comprehensive collection of LangChain examples covering all major concepts and features. This repository is designed to help you learn and understand LangChain through practical, runnable examples.

## 📚 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Structure](#structure)
- [Concepts Covered](#concepts-covered)
- [Usage](#usage)
- [Configuration](#configuration)

## 🎯 Overview

This repository contains examples for all major LangChain concepts including:

- **Document Loaders**: Loading data from various sources (text, PDF, web, CSV, JSON)
- **Text Splitters**: Breaking documents into manageable chunks
- **Indexing**: Creating searchable document indexes
- **Nearest Neighbors**: KNN and ANN retrieval methods
- **LCEL**: LangChain Expression Language for declarative chain composition
- **Retrievers**: Fetching relevant documents from data sources
- **Compressors**: Compressing and filtering retrieved documents
- **Chains**: Sequencing calls to components
- **Agents**: Using LLMs to decide actions and take them
- **Vector Stores**: Storing and searching vector embeddings
- **Evaluation**: Measuring and assessing LLM application performance
- **Memory**: Storing and retrieving conversation state
- **Callbacks**: Handling events during chain execution
- **Prompts**: Creating reusable prompt templates
- **Output Parsers**: Parsing LLM outputs into structured data
- **LLMs**: Using language models directly
- **Chat Models**: Using chat-optimized language models
- **Embeddings**: Converting text to vector representations
- **Tools**: Creating custom tools for agents
- **Messages**: Different message types for conversations
- **Few-shot Prompting**: Using examples to guide model behavior
- **Hybrid Search**: Combining keyword and semantic search
- **Reranking**: Re-ranking retrieved documents for better relevance
- **LangGraph**: Building stateful applications with graphs

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key (for most examples)

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LangChain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## 📁 Structure

```
LangChain/
├── common/              # Shared utilities and configuration
│   ├── config.py       # Configuration settings
│   ├── utils.py        # Utility functions
│   └── __init__.py
├── loaders/            # Document loading examples
├── splitters/          # Text splitting examples
├── indexing/           # Document indexing examples
├── nearest_neighbors/  # KNN and ANN retrieval examples
├── lcel/              # LangChain Expression Language examples
├── retrievers/       # Retrieval strategy examples
├── compressors/       # Document compression examples
├── chains/            # Chain composition examples
├── agents/            # Agent implementation examples
├── vector_stores/     # Vector database examples
├── evaluation/        # Evaluation and metrics examples
├── memory/            # Conversation memory examples
├── callbacks/         # Callback handler examples
├── prompts/           # Prompt template examples
├── output_parsers/    # Output parsing examples
├── llms/              # Language model examples
├── chat_models/       # Chat model examples
├── embeddings/        # Embedding model examples
├── tools/             # Custom tool examples
├── messages/          # Message type examples
├── few_shot/          # Few-shot prompting examples
├── hybrid_search/     # Hybrid search examples
├── reranking/         # Reranking examples
├── langgraph/         # LangGraph examples
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 📖 Concepts Covered

### 1. Document Loaders (`loaders/`)
Learn how to load data from various sources into LangChain Document objects.
- **text_loader.py**: Loading plain text files
- **pdf_loader.py**: Loading PDF documents
- **web_loader.py**: Loading web pages
- **csv_loader.py**: Loading CSV files
- **json_loader.py**: Loading JSON files

### 2. Text Splitters (`splitters/`)
Break large documents into smaller, manageable chunks.
- **character_splitter.py**: Splitting by character count
- **recursive_splitter.py**: Recursive splitting with multiple separators
- **markdown_splitter.py**: Splitting markdown by headers
- **code_splitter.py**: Splitting code by programming language syntax

### 3. Indexing (`indexing/`)
Create searchable indexes from documents.
- **basic_indexing.py**: Creating a basic document index
- **record_manager.py**: Managing document updates with RecordManager

### 4. Nearest Neighbors (`nearest_neighbors/`)
Find similar documents using different search methods.
- **knn_retrieval.py**: Exact K-nearest neighbors search
- **ann_retrieval.py**: Approximate nearest neighbors search
- **knn_vs_ann.py**: Comparison between KNN and ANN

### 5. LCEL (`lcel/`)
LangChain Expression Language for declarative chain composition.
- **basic_lcel.py**: Introduction to LCEL
- **runnable_parallel.py**: Running chains in parallel
- **runnable_sequence.py**: Sequencing operations

### 6. Retrievers (`retrievers/`)
Fetch relevant documents from data sources.
- **vector_retriever.py**: Using vector stores as retrievers
- **multi_query_retriever.py**: Generating multiple queries
- **context_compression.py**: Compressing retrieved documents

### 7. Compressors (`compressors/`)
Compress and filter retrieved documents.
- **llm_chain_extractor.py**: Extracting relevant parts using LLM
- **embeddings_filter.py**: Filtering by embedding similarity
- **document_compressor_pipeline.py**: Combining multiple compressors

### 8. Chains (`chains/`)
Sequence calls to components to accomplish tasks.
- **simple_chain.py**: Basic LLM chain with prompt
- **sequential_chain.py**: Running multiple chains in sequence
- **retrieval_qa.py**: Question answering with document retrieval

### 9. Agents (`agents/`)
Use LLMs to decide actions and take them.
- **tool_agent.py**: Basic agent using custom tools
- **react_agent.py**: ReAct (Reasoning + Acting) agent pattern
- **openai_functions_agent.py**: Agent using OpenAI function calling

### 10. Vector Stores (`vector_stores/`)
Store and search vector embeddings.
- **chroma.py**: Using Chroma as a local vector database
- **faiss.py**: Using FAISS for efficient similarity search
- **pinecone.py**: Using Pinecone as a cloud vector database

### 11. Evaluation (`evaluation/`)
Measure and assess LLM application performance.
- **string_evaluator.py**: Evaluating string outputs
- **retrieval_evaluator.py**: Evaluating retrieval quality
- **rag_evaluator.py**: Evaluating RAG systems

### 12. Memory (`memory/`)
Store and retrieve conversation state.
- **conversation_buffer.py**: Storing full conversation history
- **summary_memory.py**: Summarizing conversation history
- **entity_memory.py**: Extracting and remembering entities

### 13. Callbacks (`callbacks/`)
Handle events during chain execution.
- **basic_callbacks.py**: Basic callback handlers
- **streaming_callbacks.py**: Streaming outputs in real-time

### 14. Prompts (`prompts/`)
Create reusable prompt templates.
- **prompt_templates.py**: Basic and chat prompt templates

### 15. Output Parsers (`output_parsers/`)
Parse LLM outputs into structured data.
- **basic_parsers.py**: Structured output, lists, datetime parsing

### 16. LLMs (`llms/`)
Using language models directly for text generation.
- **openai_llm.py**: Using OpenAI's language models
- **anthropic_llm.py**: Using Anthropic's Claude models

### 17. Chat Models (`chat_models/`)
Using chat-optimized language models with message objects.
- **openai_chat.py**: Using OpenAI's chat models with streaming

### 18. Embeddings (`embeddings/`)
Converting text to vector representations for semantic search.
- **openai_embeddings.py**: Using OpenAI's embedding models and similarity

### 19. Tools (`tools/`)
Creating custom tools for agents to use.
- **custom_tools.py**: Simple and structured tool creation

### 20. Messages (`messages/`)
Different message types for chat conversations.
- **message_types.py**: Human, AI, System, Function, and Tool messages

### 21. Few-shot Prompting (`few_shot/`)
Using examples to guide model behavior for better performance.
- **few_shot_prompting.py**: Basic and dynamic few-shot prompting

### 22. Hybrid Search (`hybrid_search/`)
Combining keyword and semantic search for better retrieval.
- **hybrid_search.py**: Hybrid search concepts and weighted search

### 23. Reranking (`reranking/`)
Re-ranking retrieved documents for better relevance.
- **reranking.py**: LLM-based reranking and score threshold filtering

### 24. LangGraph (`langgraph/`)
Building stateful applications with graphs.
- **basic_graph.py**: Basic graph with nodes and conditional edges

## 🚀 Usage

### Running Individual Examples

Each example file can be run independently:

```bash
python loaders/text_loader.py
python splitters/recursive_splitter.py
python chains/simple_chain.py
```

### Using the Common Utilities

All examples share common utilities from the `common` module:

```python
from common import print_section, print_result, OPENAI_API_KEY
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Configuration File

The `common/config.py` file contains:

- API key configurations
- Model settings
- Directory paths
- Environment variable loading

## 📝 Notes

- Most examples require an OpenAI API key
- Some examples require additional dependencies (e.g., `pinecone-client` for Pinecone examples)
- Install missing dependencies as needed: `pip install <package-name>`
- Examples are designed to be educational and may need modification for production use

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This repository is for educational purposes. Please refer to individual package licenses for LangChain and its dependencies.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Hub](https://smith.langchain.com/)

---

Happy Learning! 🎉
