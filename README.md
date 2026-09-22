# Document Intelligence Assistant (Local RAG PDF QA)

This project is a fully local, privacy-preserving Retrieval-Augmented Generation (RAG) application. It allows you to upload PDF documents and ask questions about them. The entire process (text extraction, vector embedding, and language model inference) runs **100% locally** on your machine using Ollama and ChromaDB. No data is sent to external APIs (like OpenAI or Google).

## Architecture
- **Backend:** FastAPI
- **RAG Orchestration:** LangChain
- **Vector Database:** ChromaDB (Local)
- **Local LLMs:** Ollama 
  - *Embeddings:* `nomic-embed-text`
  - *Generation:* `llama3.1`

## Prerequisites

Before running the project, you must install Ollama and pull the required models to your local machine.

### 1. Install Ollama
Download and install Ollama for your operating system from the official website:
[https://ollama.com/download](https://ollama.com/download)

### 2. Pull the Required Local Models
Once Ollama is installed and running in the background, open your terminal and run the following commands to download the models used by this project:

```bash
# Pull the LLM used for answering questions
ollama pull llama3.1

# Pull the embedding model used for vector search
ollama pull nomic-embed-text
```
*(Note: `llama3.1` is around 4.7GB, so this step might take a few minutes depending on your internet connection.)*

## Project Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd RAG_APP
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Python Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the FastAPI server:
```bash
python app.py
```

### 5. Access the Web Interface
Open your web browser and navigate to:
[http://localhost:8000](http://localhost:8000)

You can now upload a PDF, wait for it to process locally, and start asking questions!

## Notes for GitHub
This repository is configured with a `.gitignore` that prevents your local vector database (`chroma_db/`), uploaded PDFs (`uploads/`), and environment variables (`.env`) from being pushed to GitHub. This ensures your private data remains local.
