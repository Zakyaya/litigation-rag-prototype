End-to-End Local Litigation RAG Prototype
This project is a complete, 100% local Retrieval-Augmented Generation (RAG) pipeline designed for querying and analyzing legal documents. It provides a secure, private, and cost-effective way to build a powerful legal AI assistant that runs entirely on your own machine.

The system ingests raw PDF documents, processes them into a searchable knowledge base, and uses a local Large Language Model (LLM) to generate synthesized, context-aware answers to user questions.

Features
100% Local & Private: No data ever leaves your machine. All processing, storage, and generation happens locally.

Multi-Document Support: Ingest and query across an entire library of legal documents from different cases.

Modular & Clean Codebase: The project is broken down into logical components for ingestion, indexing, and querying, making it easy to understand and extend.

State-of-the-Art Local Models: Uses powerful open-source models for high-quality embeddings (BAAI/bge-base-en-v1.5) and text generation (phi3:mini).

Built-in Debugging: A --debug flag in the query script allows you to inspect the retrieved context to evaluate and improve retrieval quality.

Setup & Installation
Follow these steps to set up the project environment.

1. Prerequisites
Python 3.10+

Ollama installed and running on your machine.

2. Clone the Repository
git clone <your-repo-url>
cd litigation-rag-prototype

3. Set Up the Environment
Create a Python virtual environment and install the required packages.

# Create the virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the root of the project by copying the example file.

cp .env.example .env

Now, open the .env file and add your secret API keys (e.g., for the Gemini API used in the ingestion script).

5. Download the LLM
Use Ollama to download the local language model needed for the query script.

ollama pull phi3:mini

How to Use the System
The pipeline is a three-step process: Ingest, Index, and Query.

Step 1: Ingest Documents
Place your PDF legal documents into the source_documents/ folder. Then, run the ingestion script:

python ingest.py

This will process the PDFs and create structured .json files in the processed_documents/ folder.

Step 2: Index the Data
Run the indexing script to create vector embeddings and build your local knowledge base:

python index.py

This will create a local_vector_db/ folder containing the searchable database.

Step 3: Query Your Documents
You can now ask questions from the command line. The script will search across all ingested documents to find an answer.

# Ask a question
python query.py "What is the legal standard for a motion to dismiss?"

# Ask a question and get the top 3 results
python query.py "Summarize the plaintiff's primary arguments." -n 3

# Run in debug mode to see the retrieved context
python query.py "What does this document say about Smith v. Jones?" --debug

Project Structure
Here is the complete file structure for the project. This modular layout keeps the code organized, secure, and easy to maintain.

litigation_rag/
├── .env                  # Stores all secret API keys and configuration variables.
├── .env.example          # An example file for environment variable setup.
├── .gitignore            # Specifies which files and folders Git should ignore.
├── README.md             # The main documentation for the project.
├── requirements.txt      # Lists all the Python packages the project depends on.
│
├── source_documents/     # You place all your raw PDF legal documents in this folder.
│   └── example_motion.pdf
│
├── processed_documents/  # The ingestion script saves its structured JSON output here.
│   └── example_motion.json
│
├── local_vector_db/      # ChromaDB stores its local database files in this directory.
│   └── ... (database files)
│
├── venv/                   # The isolated Python virtual environment for the project.
│
├── ingest.py             # Main script for the ingestion pipeline (uses helpers below).
├── config.py             # Helper module: All project configurations and constants.
├── cloud_utils.py        # Helper module: Functions for interacting with cloud services (R2).
├── chunker.py            # Helper module: Logic for chunking text using the Gemini API.
│
├── index.py              # The script to create local embeddings and build the vector database.
└── query.py              # The final script to ask questions and get answers from the local LLM.
