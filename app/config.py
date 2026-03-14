import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory of the project (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# PDF document path
PDF_PATH = os.getenv("PDF_PATH", "")

# ChromaDB persistence directory
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "")

# LLM model
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Airawat Llama API settings
AIRAWAT_API_URL = os.getenv("AIRAWAT_API_URL", "")
AIRAWAT_TOKEN = os.getenv("AIRAWAT_TOKEN", "")
AIRAWAT_MODEL = os.getenv("AIRAWAT_MODEL", "meta/llama-3.2-11b-vision-instruct")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "0"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))

# Number of documents to retrieve from vector store
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "0"))

# Flask server settings
FLASK_HOST = os.getenv("FLASK_HOST", "")
FLASK_PORT = int(os.getenv("FLASK_PORT", "0"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "").lower() == "true"

# Streamlit server settings
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "0"))

# Max retries for agentic hallucination / answer grading loops
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "0"))
