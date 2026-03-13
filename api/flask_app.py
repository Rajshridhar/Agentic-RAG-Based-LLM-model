"""Flask REST API for the Indian Cricket Agentic RAG system.

Endpoints
---------
GET  /api/health        – System health check.
POST /api/query         – Ask a question (supports "agentic" and "simple" modes).
POST /api/ingest        – Upload and ingest a new PDF document.
GET  /api/history       – Retrieve conversation history.
"""

import os
import sys
from datetime import datetime, timezone

# Ensure project root is on sys.path so `app` package is importable when
# running flask_app.py directly.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app.config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    LLM_MODEL,
)

app = Flask(__name__)
CORS(app)

# In-memory conversation history
_history: list = []

# Lazy-loaded vector store (shared across requests)
_vector_store = None


def _get_vector_store():
    global _vector_store  # noqa: PLW0603
    if _vector_store is None:
        from app.vector_store import load_vector_store

        _vector_store = load_vector_store()
    return _vector_store


def _chunk_count() -> int:
    try:
        vs = _get_vector_store()
        return len(vs.get()["ids"])
    except Exception:
        return -1

# Endpoints
@app.get("/")
def home():
    return jsonify({
        "message": "Indian Cricket Agentic RAG API running",
        "endpoints": [
            "/api/health",
            "/api/query",
            "/api/ingest",
            "/api/history"
        ]
    })
    
@app.get("/api/health")
def health():
    """Return system health information."""
    return jsonify(
        {
            "status": "healthy",
            "model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": _chunk_count(),
        }
    )


@app.post("/api/query")
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "agentic").lower()

    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    try:
        vs = _get_vector_store()

        if mode == "simple":
            from app.rag_chain import query_simple

            result = query_simple(question, vector_store=vs)
            result["agent_trace"] = []
            result["mode"] = "simple"
        else:
            from app.agents.agentic_rag import query_agentic

            result = query_agentic(question, vector_store=vs)
            result["mode"] = "agentic"

        # Persist to history
        _history.append(
            {
                "question": question,
                "answer": result["answer"],
                "mode": result["mode"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/ingest")
def ingest():
    """Upload and ingest a new PDF document.

    Expects multipart/form-data with a field named ``file`` containing a PDF.

    Response::

        {"status": "success", "chunks_created": 10}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    pdf_file = request.files["file"]
    if not pdf_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(pdf_file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Save to data directory
    data_dir = os.path.join(_project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, filename)
    pdf_file.save(save_path)

    try:
        from app.chunker import chunk_documents
        from app.document_loader import load_and_validate_pdf
        from app.vector_store import create_vector_store

        documents = load_and_validate_pdf(save_path)
        chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        vs = create_vector_store(chunks, CHROMA_DB_PATH)

        # Refresh cached vector store
        global _vector_store  # noqa: PLW0603
        _vector_store = vs

        return jsonify({"status": "success", "chunks_created": len(chunks)})

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/api/history")
def history():
    """Return the in-memory conversation history."""
    return jsonify(_history)


# Entry-point
if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
