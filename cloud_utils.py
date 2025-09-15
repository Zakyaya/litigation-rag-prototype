import json
import requests
import config

def chunk_batch_into_propositions(elements):
    """
    Takes a batch of 'unstructured' elements, sends them to the local Ollama LLM,
    and returns a list of propositions. Returns None on critical failure.
    """
    prompt_context = "\n---\n".join([f"Element {i+1} (type: {type(el).__name__}):\n{str(el)}" for i, el in enumerate(elements)])
    
    prompt = f"""
    You are an expert legal text analyzer. Decompose the following text from a legal document into a series of clear, concise, self-contained factual statements (propositions).

    RULES:
    1. Output a valid JSON object with a single key "propositions", which is an array of strings.
    2. Extract statements as they are, without adding interpretation.
    3. For each proposition, include the original page number if available in the element's metadata. Format: "[Statement text] (Page: [number])"

    CONTEXT TO ANALYZE:
    ---
    {prompt_context}
    ---

    JSON OUTPUT:
    """

    try:
        payload = {
            "model": config.OLLAMA_CHUNKER_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        response = requests.post(config.OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        
        # --- START: ENHANCED DEBUGGING ---
        raw_response_text = response_json.get('response', '{}')
        print(f"  - DEBUG: Raw Ollama response: {raw_response_text}")
        # --- END: ENHANCED DEBUGGING ---
        
        response_data = json.loads(raw_response_text)
        
        propositions = response_data.get("propositions", [])
        
        # --- START: ENHANCED DEBUGGING ---
        if not propositions:
            print("  - WARNING: Ollama returned a valid JSON but found no propositions. The model may not be following instructions correctly.")
        # --- END: ENHANCED DEBUGGING ---
        
        enriched_propositions = []
        for prop_item in propositions: # Renamed for clarity
            actual_text_to_check = ""
            if isinstance(prop_item, str):
                actual_text_to_check = prop_item
            elif isinstance(prop_item, dict):
                for key in ['proposition', 'statement', 'text']:
                    if isinstance(prop_item.get(key), str):
                        actual_text_to_check = prop_item[key]
                        break
            
            if not actual_text_to_check:
                continue

            source_element = None
            for el in elements:
                if hasattr(el, 'text') and (el.text in actual_text_to_check or actual_text_to_check in el.text):
                    source_element = el
                    break
            
            page_number = None
            filename = 'unknown'
            if source_element and hasattr(source_element, 'metadata'):
                page_number = getattr(source_element.metadata, 'page_number', None)
                filename = getattr(source_element.metadata, 'filename', 'unknown')

            enriched_prop = {
                "text": actual_text_to_check,
                "document_name": filename,
                "page_number": page_number
            }
            enriched_propositions.append(enriched_prop)

        return enriched_propositions

    except requests.exceptions.RequestException:
        print(f"  - ERROR: Could not connect to Ollama server at {config.OLLAMA_API_URL}. Is Ollama running?")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        raw_response = response.json().get('response', 'No response text found.') if 'response' in locals() else 'Response object not created.'
        print(f"  - ERROR: Failed to parse JSON response from Ollama: {e}")
        print(f"  - RAW OLLAMA RESPONSE: {raw_response}")
        return None
    except Exception as e:
        print(f"  - An unexpected error occurred during chunking: {e}")
        return None
