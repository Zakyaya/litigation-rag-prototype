import os
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
PROCESSED_DOCS_PATH = "processed_documents"
LOCAL_DB_PATH = "local_vector_db"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "litigation_rag"
BATCH_SIZE = 50 # Number of items to process at a time

def is_junk(text):
    """
    Checks if a given text string is likely junk or an instructional artifact.
    """
    if not text or not isinstance(text, str):
        return True
    # Filter out very short, non-descriptive text
    if len(text.strip()) < 10:
        return True
    # Filter out placeholder patterns
    if re.search(r'\[.*\]|\{.*\}', text):
        return True
    # Filter out text that is just a page number artifact
    if re.match(r'^\(Page: .*\)$', text.strip()):
        return True
    return False

def embed_and_index_documents(embedding_model, chroma_collection):
    """
    Finds processed JSON files, cleans the data, generates embeddings, 
    and indexes them in ChromaDB.
    """
    if not os.path.exists(PROCESSED_DOCS_PATH):
        print(f"Error: The directory '{PROCESSED_DOCS_PATH}' was not found.")
        return

    json_files = [f for f in os.listdir(PROCESSED_DOCS_PATH) if f.endswith('.json')]
    print(f"Found {len(json_files)} processed documents to index.")

    total_vectors_added = 0
    skipped_files_count = 0

    for file_name in json_files:
        print(f"\n--- Processing: {file_name} ---")
        file_path = os.path.join(PROCESSED_DOCS_PATH, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  - Error reading or parsing {file_name}: {e}. Skipping.")
            skipped_files_count += 1
            continue
            
        if not chunks:
            print(f"  - No content found in {file_name}. Skipping.")
            skipped_files_count += 1
            continue

        # --- START: DATA CLEANING ---
        cleaned_chunks = [chunk for chunk in chunks if not is_junk(chunk.get("text"))]
        num_filtered = len(chunks) - len(cleaned_chunks)
        if num_filtered > 0:
            print(f"  - Cleaned data: Removed {num_filtered} low-quality or junk entries.")
        # --- END: DATA CLEANING ---

        if not cleaned_chunks:
            print(f"  - No valid content found after cleaning {file_name}. Skipping.")
            skipped_files_count += 1
            continue

        for i in range(0, len(cleaned_chunks), BATCH_SIZE):
            batch = cleaned_chunks[i:i+BATCH_SIZE]
            print(f"  - Creating embeddings for batch {i//BATCH_SIZE + 1}...")

            texts_to_embed = [chunk.get("text", "") for chunk in batch]
            ids = [f"{os.path.splitext(file_name)[0]}_{i+j}" for j in range(len(batch))]
            
            # --- METADATA SANITIZATION ---
            metadatas_raw = [
                {
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number"),
                    "text": chunk.get("text")
                } for chunk in batch
            ]
            metadatas_clean = []
            for meta in metadatas_raw:
                clean_meta = {}
                for key, value in meta.items():
                    if value is None:
                        if key == 'page_number':
                            clean_meta[key] = 0
                        else:
                            clean_meta[key] = ""
                    else:
                        clean_meta[key] = value
                metadatas_clean.append(clean_meta)

            try:
                embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()
                chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas_clean
                )
                total_vectors_added += len(batch)
                print(f"    - Successfully added {len(batch)} vectors to ChromaDB.")
            except Exception as e:
                print(f"    - Error adding batch to ChromaDB: {e}")

    print("\n--- Indexing Complete ---")
    if skipped_files_count > 0:
        print(f"Warning: Skipped {skipped_files_count} files due to errors or no content.")
    print(f"Added {total_vectors_added} new vectors.")
    print(f"Vector database is ready in the '{LOCAL_DB_PATH}' directory.")


def main():
    """
    Main function to set up the database and run the indexing process.
    """
    print("Starting local vector indexing process...")
    print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print("Successfully loaded model.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return

    try:
        client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        initial_count = collection.count()
        print(f"Successfully connected to local ChromaDB collection '{COLLECTION_NAME}'.")
        print(f"Initial collection count: {initial_count}")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return

    embed_and_index_documents(model, collection)
    final_count = collection.count()
    print(f"Final collection count: {final_count}")


if __name__ == "__main__":
    main()
