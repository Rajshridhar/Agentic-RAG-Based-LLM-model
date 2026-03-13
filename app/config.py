"""Configuration module – reads settings from environment variables with sensible defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory of the project (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# PDF document path
PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "Indian_Cricket_Report.pdf"))

# ChromaDB persistence directory
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))

# LLM model (free, runs locally on CPU)
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")

# Embedding model (sentence-transformers, free)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))

# Number of documents to retrieve from vector store
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# Flask server settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# Streamlit server settings
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Max retries for agentic hallucination / answer grading loops
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
