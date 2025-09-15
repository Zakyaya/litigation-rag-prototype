import os
import json
import config
from unstructured.partition.pdf import partition_pdf
from chunker import chunk_batch_into_propositions
from cloud_utils import get_r2_client, upload_to_r2

# We define BATCH_SIZE here now as it's a processing parameter, not a rate-limit one.
BATCH_SIZE = 10 

def process_document(doc_path, r2_client):
    """
    Processes a single document: extracts elements, chunks them into
    propositions, saves them locally, and uploads them to R2.
    """
    file_name = os.path.basename(doc_path)
    print(f"\n--- Processing document: {file_name} ---")

    os.makedirs(config.PROCESSED_DOCS_PATH, exist_ok=True)

    try:
        elements = partition_pdf(
            filename=doc_path,
            strategy="hi_res",
            infer_table_structure=True,
            model_name="yolox"
        )
        print(f"  - Extracted {len(elements)} initial elements from PDF.")
    except Exception as e:
        print(f"  - Error partitioning PDF {file_name}: {e}")
        return

    all_propositions = []
    element_batch = []

    for i, element in enumerate(elements):
        element_batch.append(element)
        
        # Process the batch if it's full or if it's the last element
        if len(element_batch) >= BATCH_SIZE or i == len(elements) - 1:
            print(f"  - Chunking a batch of {len(element_batch)} elements...")
            # Corrected function call: only one argument is passed
            propositions = chunk_batch_into_propositions(element_batch)
            
            if propositions is None: # Check for a failure signal from the chunker
                print("  - A critical error occurred during chunking. Aborting processing for this document.")
                return # Stop processing this file
            
            all_propositions.extend(propositions)
            element_batch = []

    output_filename = f"{os.path.splitext(file_name)[0]}.json"
    output_path = os.path.join(config.PROCESSED_DOCS_PATH, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_propositions, f, ensure_ascii=False, indent=4)
        print(f"  - Successfully saved processed data locally to: {output_path}")
    except Exception as e:
        print(f"  - Error saving data locally: {e}")

    # Optionally upload to R2 if the client is configured and data exists
    if r2_client and all_propositions:
        upload_to_r2(r2_client, output_path, f"processed/{output_filename}")


def main():
    """
    Main function to find and process all documents in the source directory.
    """
    print("Starting document ingestion process...")
    r2_client = get_r2_client()

    if not os.path.exists(config.SOURCE_DOCS_PATH):
        print(f"Error: Source directory '{config.SOURCE_DOCS_PATH}' not found.")
        return

    pdf_files = [f for f in os.listdir(config.SOURCE_DOCS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for file_name in pdf_files:
        file_path = os.path.join(config.SOURCE_DOCS_PATH, file_name)
        process_document(file_path, r2_client)

if __name__ == "__main__":
    main()
