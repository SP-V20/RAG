# RAG_LC - Retrieval-Augmented Generation with LangChain

A powerful Retrieval-Augmented Generation (RAG) system that combines document retrieval with language models to provide accurate, context-aware answers using LangChain and Hugging Face models.

## Overview

This project implements a complete RAG pipeline that:
1. Loads documents from text files
2. Splits them into manageable chunks
3. Converts text to vector embeddings
4. Stores embeddings in FAISS for fast similarity search
5. Retrieves relevant documents based on user queries
6. Generates answers using the Hugging Face Inference API with GPT-OSS-120B model

The system ensures responses are grounded in your provided documents, reducing hallucinations and improving reliability.

## Features

- 📄 **Document Loading**: Load documents from text files using TextLoader
- 🔀 **Smart Text Splitting**: RecursiveCharacterTextSplitter with configurable chunk size (500 chars) and overlap (50 chars)
- 🧠 **Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic understanding
- 🔍 **Fast Vector Search**: FAISS (Facebook AI Similarity Search) for efficient document retrieval
- 💬 **Context-Aware Q&A**: Retrieves top-3 most relevant documents and provides grounded answers
- 🤖 **LLM Integration**: Uses Hugging Face Inference API with GPT-OSS-120B model
- 🎯 **Interactive CLI**: Chat-like interface for querying documents

## Project Structure

```
RAG_LC/
├── main.py                 # Main RAG application
├── data/
│   └── sample.txt        # Sample documents for retrieval
├── requirements.txt       # Python dependencies
├── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.9+
- pip  
- Hugging Face API token (for inference)

### Setup

1. **Clone or download the project**
   ```bash
   cd RAG_LC
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   HuggingFaceToken=your_huggingface_api_token_here
   ```

## Usage

### Running the Application

```bash
python app.py
```

The application will:
1. Load documents from `data/sample.txt`
2. Build the vector index (first run may take a moment)
3. Start an interactive query loop

### Interactive Query Loop

```
Ask something (or 'exit'): What is the main topic?
Retrieved documents: [...]
Answer: [AI-generated response based on documents]
```

Type `exit` to quit the application.

## How It Works

### Step-by-Step Process

1. **Document Loading**: TextLoader reads the sample text file
2. **Text Splitting**: RecursiveCharacterTextSplitter breaks text into 500-character chunks with 50-character overlap for context preservation
3. **Embeddings**: Text chunks are converted to vector embeddings using a sentence transformer model
4. **Vector Store**: FAISS creates an indexed vector database for fast similarity search
5. **Retrieval**: For each query, the top-3 most semantically similar documents are retrieved
6. **Context Assembly**: Retrieved documents are joined with double newlines for clarity
7. **LLM Generation**: The context and query are sent to GPT-OSS-120B for answer generation
8. **Response**: The generated answer is displayed to the user

### Configuration Parameters

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters (prevents losing context at boundaries)
- **Retrieval K**: 3 (retrieves top-3 similar documents)
- **Max Tokens**: 200 (response length limit)
- **Temperature**: 0.7 (balances creativity and determinism)

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core RAG framework |
| `langchain-community` | Community integrations (loaders, vectorstores, embeddings) |
| `langchain-text-splitters` | Text chunking utilities |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Embedding models |
| `huggingface-hub` | Inference API client |
| `python-dotenv` | Environment variable management |


### Missing HuggingFaceToken warning
Ensure your `.env` file contains:
```
HuggingFaceToken=your_token
```


