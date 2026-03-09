# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to
upload PDFs and ask questions about their documents. The system chunks
documents, stores embeddings in Pinecone, retrieves relevant context,
and generates answers using Groq LLM.

------------------------------------------------------------------------

## Features

-   Upload PDF documents
-   Automatic document chunking
-   Semantic search using vector embeddings
-   Context-aware answers using an LLM
-   Chat interface using Streamlit
-   Vector storage with Pinecone
-   Fast inference with Groq

------------------------------------------------------------------------

## Architecture

PDF Upload\
↓\
Document Loader\
↓\
Chunking\
↓\
Embedding Model (all-MiniLM-L6-v2)\
↓\
Pinecone Vector Database\
↓\
Retriever\
↓\
Groq LLM\
↓\
Answer Generation

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Streamlit
-   Pinecone
-   LangChain
-   Groq LLM
-   Sentence Transformers

------------------------------------------------------------------------

## Project Structure

    rag_pipeline
    │
    ├── app.py                     # Streamlit frontend
    │
    ├── src
    │   ├── chunking.py            # Document chunking
    │   ├── loader.py              # PDF loader
    │   ├── pinecone_database.py   # Pinecone ingestion
    │   └── search_on_pinecone.py  # Retrieval + LLM
    │
    ├── pdfs                       # Uploaded PDFs
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Installation

Clone the repository

    git clone https://github.com/yourusername/pdf-rag-assistant.git
    cd pdf-rag-assistant

Create a virtual environment

    python -m venv rag

Activate the environment

Windows:

    rag\Scripts\activate

Install dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## Environment Variables

Create a `.env` file in the root directory:

    PINECONE_API_KEY=your_pinecone_api_key
    GROQ_API_KEY=your_groq_api_key

------------------------------------------------------------------------

## Running the Application

Start the Streamlit app:

    streamlit run app.py

Then open your browser:

    http://localhost:8501

------------------------------------------------------------------------

## Usage

1.  Upload a PDF document
2.  The system will:
    -   Load the document
    -   Chunk the text
    -   Store embeddings in Pinecone
3.  Ask questions in the chat interface
4.  The system retrieves relevant chunks and generates an answer

------------------------------------------------------------------------

## Example Query

    What is transformer architecture?

Example response:

    Transformers are neural network architectures that use attention mechanisms
    to understand relationships between words in a sequence...

------------------------------------------------------------------------

## Future Improvements

-   Multi-document search
-   Hybrid retrieval (BM25 + embeddings)
-   Document source citations
-   Streaming responses
-   Multi-user support
-   Deployment on cloud

------------------------------------------------------------------------

## License

This project is licensed under the MIT License.
