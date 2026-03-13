# 🏏 Indian Cricket History – Agentic RAG

A **Retrieval-Augmented Generation (RAG)** system about Indian Cricket history, upgraded from a simple Jupyter notebook to a full **Agentic RAG** application with a Flask REST API and a Streamlit chat UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│           Streamlit Chat UI  ──  Flask REST API                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Agentic RAG (LangGraph)                      │
│                                                                 │
│  question ──► Query Router ──► [not_relevant] ──► END           │
│                    │                                            │
│                    ▼ vectorstore                                │
│               Retrieve Docs                                     │
│                    │                                            │
│                    ▼                                            │
│              Grade Documents ──► filter irrelevant docs         │
│                    │                                            │
│                    ▼                                            │
│                Generate Answer                                  │
│                    │                                            │
│                    ▼                                            │
│          Check Hallucination ──► [no] ──► Regenerate (max 3x)  │
│                    │                                            │
│                    ▼ yes                                        │
│              Grade Answer ──► [no] ──► Fallback message        │
│                    │                                            │
│                    ▼ yes                                        │
│                   END                                           │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Core Modules                                │
│  document_loader ▸ chunker ▸ embeddings ▸ vector_store ▸ llm   │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Storage / Models                              │
│  ChromaDB (./chroma_db)   +   google/flan-t5-small (CPU)       │
│  sentence-transformers/all-MiniLM-L6-v2 (embeddings)           │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
project-root/
├── app/
│   ├── config.py              # Configuration (env vars, paths, model settings)
│   ├── document_loader.py     # PDF loading + validations (size, MIME, integrity, readability, OCR)
│   ├── chunker.py             # Text chunking with RecursiveCharacterTextSplitter
│   ├── embeddings.py          # Embedding model setup (HuggingFace)
│   ├── vector_store.py        # ChromaDB vector store (create/load)
│   ├── llm.py                 # LLM setup (HuggingFace pipeline)
│   ├── rag_chain.py           # Simple RAG chain (fallback mode)
│   ├── agents/
│   │   ├── query_router.py    # Routes query to vectorstore or rejects it
│   │   ├── retrieval_grader.py# Grades retrieved docs for relevance
│   │   ├── hallucination_grader.py # Checks for hallucinations
│   │   ├── answer_grader.py   # Grades if answer addresses question
│   │   └── agentic_rag.py     # LangGraph StateGraph orchestrator
│   └── tools/
│       ├── vector_search.py   # ChromaDB search tool
│       └── web_search.py      # Web search placeholder
├── api/
│   └── flask_app.py           # Flask REST API
├── ui/
│   └── streamlit_app.py       # Streamlit chat interface
├── data/
│   └── Indian_Cricket_Report.pdf
├── scripts/
│   └── ingest.py              # PDF ingestion script
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Ingest the PDF

The `chroma_db/` directory with pre-built embeddings is included.  To rebuild it:

```bash
python scripts/ingest.py
# Or to force a recreate:
python scripts/ingest.py --recreate
```

---

## Running the Flask API

```bash
python api/flask_app.py
# API will start on http://localhost:5000
```

---

## Running the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
# UI will open at http://localhost:8501
```

---

## API Endpoints

### `GET /api/health`

Returns system health information.

```json
{
  "status": "healthy",
  "model": "google/flan-t5-small",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunks": 10
}
```

### `POST /api/query`

Ask a question about Indian Cricket history.

**Request:**
```json
{
  "question": "Tell me about the IPL",
  "mode": "agentic"
}
```

`mode` can be `"agentic"` (default) or `"simple"`.

**Response:**
```json
{
  "answer": "The Indian Premier League (IPL) is ...",
  "sources": [
    {
      "page": 2,
      "chunk_id": 5,
      "content_preview": "..."
    }
  ],
  "agent_trace": [
    "route_query → vectorstore",
    "retrieve → 5 document(s) retrieved",
    "grade_docs → 3/5 document(s) relevant",
    "generate → answer produced",
    "check_hallucination → grounded=yes (retry 0)",
    "grade_answer → useful=yes"
  ],
  "mode": "agentic"
}
```

### `POST /api/ingest`

Upload and index a new PDF document.

```bash
curl -X POST http://localhost:5000/api/ingest \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "chunks_created": 10
}
```

### `GET /api/history`

Retrieve in-memory conversation history.

```json
[
  {
    "question": "Who is Sachin Tendulkar?",
    "answer": "Sachin Tendulkar is ...",
    "mode": "agentic",
    "timestamp": "2024-01-01T12:00:00+00:00"
  }
]
```

---

## Example Queries

- *"Who is considered the greatest Indian batsman of all time?"*
- *"Tell me about India's first cricket World Cup victory."*
- *"What is the IPL and when was it founded?"*
- *"Who holds the record for most Test centuries for India?"*
- *"Describe India's cricket performance in the 1983 World Cup."*

---

## Notes

- All models run **locally on CPU** – no API keys or cloud costs required.
- `google/flan-t5-small` is intentionally lightweight; upgrade to a larger model for better accuracy.
- The Agentic RAG system uses the **same LLM** for routing, grading, and generation.
- Web search fallback is a placeholder – integrate Tavily/SerpAPI for production use.
