import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PATHS ---
SOURCE_DOCS_PATH = "source_documents"
PROCESSED_DOCS_PATH = "processed_documents"
LOCAL_DB_PATH = "local_vector_db"

# --- R2 (Cloud Storage - Optional) ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# --- LOCAL LLM (Ollama) CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Model used for the one-time task of chunking documents
OLLAMA_CHUNKER_MODEL = "phi3:mini"

