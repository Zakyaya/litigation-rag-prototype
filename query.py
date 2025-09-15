import chromadb
import argparse
import requests
import json
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
LOCAL_DB_PATH = "local_vector_db"
# Ensure this matches the model used in index.py for consistency
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "litigation_rag"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Using a faster, optimized model for local generation
OLLAMA_MODEL = "phi3:mini"

def generate_response(context, query):
    """
    Sends the context and query to the local LLM to generate a final answer.
    """
    prompt = f"""
    You are an expert legal assistant. Your task is to answer the user's question based *only* on the provided context.
    Do not use any outside knowledge. If the context does not contain the answer, state that clearly.
    Be concise and directly answer the question.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {query}

    ANSWER:
    """

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False # Set to False for a single, complete response
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Parse the JSON response from Ollama
        response_data = response.json()
        return response_data.get('response', 'Error: Could not parse response from LLM.')

    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to Ollama server at {OLLAMA_API_URL}. Is Ollama running?"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def main():
    """
    Main function to handle command-line arguments, perform retrieval, and generate a response.
    """
    parser = argparse.ArgumentParser(description="Query your local RAG system.")
    parser.add_argument("query", type=str, help="The question you want to ask your documents.")
    parser.add_argument("-n", "--num_results", type=int, default=5, help="Number of results to retrieve for context.")
    parser.add_argument("--debug", action="store_true", help="If set, prints the retrieved context before generation.")
    args = parser.parse_args()

    # --- 1. LOAD EMBEDDING MODEL ---
    print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading embedding model: {e}"); return

    # --- 2. CONNECT TO VECTOR DB ---
    print(f"Connecting to local vector DB at: {LOCAL_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Successfully connected to ChromaDB collection.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}\nDid you run the 'index.py' script first?")
        return

    # --- 3. RETRIEVAL STEP ---
    print(f"\nEmbedding your query and retrieving context...")
    query_embedding = model.encode(args.query).tolist()
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.num_results
        )
    except Exception as e:
        print(f"Error querying the database: {e}")
        return

    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        print("Could not retrieve any context from the document for your query."); return
        
    # Extract the original text from the metadata
    retrieved_docs = [meta['text'] for meta in results['metadatas'][0]]
    context_str = "\n\n".join(retrieved_docs)
    print("Successfully retrieved context.")
    
    # If debug flag is used, print context and exit
    if args.debug:
        print("\n--- DEBUG: Retrieved Context ---")
        print(context_str)
        
        # Also show the source for each chunk
        print("\n--- DEBUG: Sources ---")
        for i, meta in enumerate(results['metadatas'][0]):
            print(f"  Chunk {i+1}: {meta.get('source', 'N/A')}, Page: {meta.get('page', 'N/A')}")
        print("------------------------")
        return

    # --- 4. GENERATION STEP ---
    print("\nSynthesizing a final answer with local LLM...")
    print("-" * 50)
    
    final_answer = generate_response(context_str, args.query)
    
    print("\nFinal Answer:\n")
    print(final_answer)
    print("-" * 50)

if __name__ == "__main__":
    main()
