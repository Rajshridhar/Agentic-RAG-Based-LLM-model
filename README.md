# 🏏 Indian Cricket History — Agentic RAG

A production-grade **Retrieval-Augmented Generation (RAG)** system built around Indian Cricket history, featuring both **Simple RAG** and **Agentic RAG** pipelines. The project exposes a **Flask REST API** and an interactive **Streamlit Chat UI**, letting users ask natural-language questions and receive grounded, source-cited answers from uploaded PDF documents.

---

## Key Features

### 1. Document-Grounded Question Answering (PDF Only)

The system answers questions **exclusively from user-provided PDF documents**. When a query is submitted, the pipeline retrieves the most relevant chunks from the indexed knowledge base and generates a response grounded strictly in the document content — no external knowledge is fabricated. If the information is not present in the documents, the model explicitly states so.

### 2. Airawat API — LLM Integration

All language understanding and generation is powered by **Meta's Llama 3.2 11B Vision Instruct** (`meta/llama-3.2-11b-vision-instruct`) served through the **Airawat API**.

| Attribute             | Detail                                                                                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model**             | `meta/llama-3.2-11b-vision-instruct`                                                                                                                             |
| **Parameters**        | **11 billion**                                                                                                                                                   |
| **Architecture**      | Llama 3.2 — an optimised decoder-only transformer with Grouped-Query Attention (GQA) for faster inference                                                        |
| **Modality**          | Multimodal (text + vision); this project uses the text instruction-following capability                                                                          |
| **Instruction-Tuned** | Fine-tuned with RLHF and instruction tuning for precise, safe, and context-aware responses                                                                       |
| **Context Window**    | 128 K tokens                                                                                                                                                     |
| **Strengths**         | Excellent at structured Q&A, summarisation, reasoning over long context, and following multi-step instructions — ideal for a RAG grading and generation pipeline |

A custom LangChain `LLM` wrapper (`AirawatLlamaLLM`) handles API communication with Bearer-token authentication, configurable temperature (default 0.7), max tokens (512), and a 120-second timeout.

### 3. Pinecone — Vector Store for the RAG Knowledge Base

Document embeddings (the knowledge base) are stored and queried through **Pinecone**, a purpose-built managed vector database.

| Why Pinecone?                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Serverless & Fully Managed** — No infrastructure to maintain; Pinecone handles scaling, replication, and indexing automatically (deployed on AWS `us-east-1`)                  |
| **Blazing-Fast Similarity Search** — Optimised ANN (Approximate Nearest Neighbour) indexing delivers sub-second retrieval even over millions of vectors                          |
| **Metadata Filtering** — Supports rich metadata filters (`source_file`, `page`, `chunk_id`, etc.) enabling precise duplicate detection and scoped queries                        |
| **Hosted Inference Embeddings** — The project uses Pinecone's own hosted embedding model (`llama-text-embed-v2`) so embeddings are computed server-side with zero local overhead |
| **Cosine Similarity Metric** — Uses cosine distance for semantic matching, the gold standard for text-embedding comparison                                                       |
| **Seamless LangChain Integration** — First-class `langchain-pinecone` SDK makes it plug-and-play with the rest of the pipeline                                                   |

### 4. Dual RAG Modes — Simple RAG & Agentic RAG

Users can switch between two RAG modes from the sidebar:

#### Simple RAG

A straightforward **Retrieve → Generate** pipeline. The user's question is sent to the Pinecone retriever, the top-K relevant chunks are fetched, combined into a context prompt, and passed to the LLM for answer generation. Simple RAG is fast and lightweight — ideal when quick answers are acceptable without multi-step quality checks.

#### Agentic RAG

A multi-stage, **LLM-as-judge** pipeline orchestrated by **LangGraph StateGraph**. After retrieval, a series of intelligent agents evaluate and refine the response:

1. **Query Router** — Classifies whether the question is cricket-related. Irrelevant queries are rejected immediately.
2. **Document Retriever** — Fetches the top-K chunks from Pinecone.
3. **Retrieval Grader** — Each retrieved document is individually evaluated by the LLM for relevance; irrelevant chunks are filtered out.
4. **Answer Generator** — The LLM produces a response from the filtered, high-quality context.
5. **Hallucination Grader** — Verifies the generated answer is fully grounded in the source documents. If not, the answer is regenerated (up to a configurable `MAX_RETRIES`).
6. **Answer Grader** — Confirms the answer actually addresses the user's question. If it does not, a fallback message is returned.

Agentic RAG ensures higher factual accuracy and relevance at the cost of additional LLM calls.

### 5. Full Agent Trace & Source Citations

Every Agentic RAG response includes a **step-by-step reasoning trace** that shows exactly how the agent processed the query:

```
→ route_query → vectorstore
→ retrieve → 5 document(s) retrieved
→ grade_docs → 3/5 document(s) relevant
→ generate → answer produced
→ check_hallucination → grounded=yes (retry 0)
→ grade_answer → useful=yes
```

Each response also contains **source citations** with page numbers, chunk IDs, and content previews, so users can verify the origin of every claim. The trace can be toggled on or off from the Streamlit sidebar.

### 6. Real-Time PDF Upload & Knowledge Base Expansion

Users can upload new PDF documents directly through the **Streamlit sidebar** or the `/api/ingest` endpoint to expand the knowledge base on-the-fly for real-time Q&A.

The ingestion pipeline enforces a **robust 5-tier validation layer** before any document enters the knowledge base:

| Validation Step      | What It Checks                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **File Size**        | Rejects files exceeding 50 MB                                                                                            |
| **MIME Type**        | Ensures the file is a genuine `application/pdf` (via `python-magic`)                                                     |
| **PDF Integrity**    | Opens the file with PyMuPDF (`fitz`) to verify it is not corrupted and has at least one page                             |
| **Text Readability** | Computes a Flesch Reading Ease score (minimum threshold: 18.0) on the first page to ensure meaningful text content       |
| **OCR Quality**      | Calculates the ratio of alphabetic characters + spaces across all pages (minimum ratio: 0.7) to catch garbled OCR output |

**Duplicate prevention**: Before ingestion, the system queries Pinecone metadata to check if a document with the same filename has already been indexed. Duplicate uploads are rejected with a `409 Conflict` response, preventing redundant vectors from polluting the knowledge base.

### 7. Web Search Layer (Placeholder)

The architecture includes a **web search tool** (`app/tools/web_search.py`) designed as a fallback for queries that cannot be answered from the indexed documents. In a production deployment, this layer would integrate with services such as **Tavily**, **SerpAPI**, or **DuckDuckGo** to fetch and store real-time web content for response generation. Currently, it returns a clear message indicating that web search is not yet configured.

### 8. Additional Highlights

- **Rich Chunk Metadata** — Each chunk is enriched with `chunk_id`, `page`, `document_title`, `section` (auto-detected from first line), `chunk_size`, table detection (`has_tables`), and a per-chunk Flesch readability score.
- **Configurable Chunking** — Uses LangChain's `RecursiveCharacterTextSplitter` with environment-variable-driven `CHUNK_SIZE` and `CHUNK_OVERLAP`.
- **Response Formatting** — A dedicated formatter cleans up LLM output: collapses excessive newlines, wraps long paragraphs, and capitalises the first character for polished presentation.
- **Centralised Logging** — Every module uses a shared logger (`app/logger.py`) with timestamped, levelled output to stdout for easy debugging.
- **CORS-Enabled REST API** — The Flask backend supports cross-origin requests, enabling integration with any frontend.
- **Conversation History** — In-memory history stores all Q&A exchanges with timestamps, accessible via `/api/history`.
- **Singleton Resource Management** — LLM, embeddings, Pinecone client, and vector store instances are cached globally to avoid redundant initialisation across requests.
- **Environment-Driven Configuration** — All secrets, model names, chunking parameters, and server settings are loaded from a `.env` file via `python-dotenv`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACES                             │
│        Streamlit Chat UI (port 8501)  ◄──►  Flask REST API (5000)    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                     FLASK API LAYER  (api/flask_app.py)               │
│  GET  /api/health   — model info, chunk count, index status          │
│  POST /api/query    — question + mode (agentic | simple)             │
│  POST /api/ingest   — PDF upload, validate, chunk, index             │
│  GET  /api/history  — conversation history                           │
└──────────┬──────────────────────────────────┬────────────────────────┘
           │  mode: agentic                   │  mode: simple
           ▼                                  ▼
┌────────────────────────────────┐  ┌──────────────────────────────────┐
│   AGENTIC RAG (LangGraph)      │  │   SIMPLE RAG (LCEL Chain)        │
│                                │  │                                  │
│  1. Query Router (LLM)         │  │  1. Retrieve top-K docs          │
│     └─► not_relevant → END     │  │  2. Format context               │
│     └─► vectorstore ↓         │  │  3. Generate answer (LLM)        │
│  2. Retrieve top-K docs        │  │  4. Format & return              │
│  3. Grade Documents (LLM)      │  └──────────────────────────────────┘
│     └─► filter irrelevant      │
│  4. Generate Answer (LLM)      │
│  5. Hallucination Check (LLM)  │
│     └─► no → Regenerate (×3)   │
│     └─► yes ↓                 │
│  6. Grade Answer (LLM)         │
│     └─► no  → Fallback msg    │
│     └─► yes → END             │
└────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────────────┐
│                         CORE MODULES  (app/)                         │
│  document_loader ▸ chunker ▸ embeddings ▸ vector_store ▸ llm         │
│  formatter ▸ logger ▸ rag_chain                                      │
└──────────┬───────────────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                              │
│  🤖 Airawat API  →  meta/llama-3.2-11b-vision-instruct  (LLM)        │
│  📊 Pinecone     →  Serverless vector DB  (AWS us-east-1, cosine)    │
│                     Embeddings: llama-text-embed-v2 (hosted)         │
└──────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
project-root/
├── app/
│   ├── config.py              # Environment-driven configuration (API keys, model settings, chunking params)
│   ├── document_loader.py     # PDF loading + 5-tier validation (size, MIME, integrity, readability, OCR)
│   ├── chunker.py             # Text chunking with RecursiveCharacterTextSplitter + metadata enrichment
│   ├── embeddings.py          # Pinecone hosted embedding model (llama-text-embed-v2)
│   ├── vector_store.py        # Pinecone vector store — create, load, duplicate check, retriever
│   ├── llm.py                 # Airawat Llama API wrapper (custom LangChain LLM class)
│   ├── rag_chain.py           # Simple RAG chain (direct retrieve → generate)
│   ├── formatter.py           # Response cleanup — paragraph wrapping, newline normalisation
│   ├── logger.py              # Centralised timestamped logging to stdout
│   ├── agents/
│   │   ├── query_router.py    # LLM-based query classifier (cricket-related or not)
│   │   ├── retrieval_grader.py# Per-document relevance grading via LLM
│   │   ├── hallucination_grader.py # Verifies answer is grounded in source documents
│   │   ├── answer_grader.py   # Checks if the answer addresses the user's question
│   │   └── agentic_rag.py     # LangGraph StateGraph orchestrator (9 nodes, conditional edges)
│   └── tools/
│       ├── vector_search.py   # Pinecone retrieval wrapper for tool-calling agents
│       └── web_search.py      # Web search placeholder (Tavily / SerpAPI / DuckDuckGo)
├── api/
│   └── flask_app.py           # Flask REST API (health, query, ingest, history)
├── ui/
│   └── streamlit_app.py       # Streamlit chat interface with sidebar controls
├── data/
│   └── *.pdf                  # Uploaded PDF documents
├── scripts/
│   └── ingest.py              # CLI ingestion script (load → validate → chunk → index)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set the following required variables:

| Variable              | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `AIRAWAT_API_KEY`     | Airawat API endpoint URL                                       |
| `AIRAWAT_API_TOKEN`   | Bearer token for Airawat authentication                        |
| `AIRAWAT_MODEL`       | LLM model name (default: `meta/llama-3.2-11b-vision-instruct`) |
| `PINECONE_API_KEY`    | Pinecone API key                                               |
| `PINECONE_INDEX_NAME` | Pinecone index name                                            |
| `EMBEDDING_MODEL`     | Embedding model name (e.g. `llama-text-embed-v2`)              |
| `EMBEDDING_DIMENSION` | Embedding vector dimension                                     |
| `CHUNK_SIZE`          | Number of characters per chunk                                 |
| `CHUNK_OVERLAP`       | Overlap between consecutive chunks                             |
| `RETRIEVER_K`         | Number of top documents to retrieve                            |
| `MAX_RETRIES`         | Maximum hallucination regeneration attempts                    |

### 3. Ingest a PDF

```bash
python scripts/ingest.py --pdf data/Indian_Cricket_Report.pdf

# To delete all existing vectors and re-index from scratch:
python scripts/ingest.py --pdf data/Indian_Cricket_Report.pdf --recreate
```

---

## Running the Application

### Start the Flask API

```bash
python api/flask_app.py
# API starts on http://localhost:5000
```

### Start the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
# UI opens at http://localhost:8501
```

> **Note:** The Streamlit UI communicates with the Flask backend. Make sure the Flask API is running before launching the UI.

---

## API Endpoints

### `GET /api/health`

Returns system health information.

```json
{
  "status": "healthy",
  "model": "meta/llama-3.2-11b-vision-instruct",
  "embedding_model": "llama-text-embed-v2",
  "pinecone_index": "your-index-name",
  "chunks": 42
}
```

### `POST /api/query`

Ask a question about the indexed documents.

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

Upload and index a new PDF document. Duplicate filenames are rejected.

```bash
curl -X POST http://localhost:5000/api/ingest \
  -F "file=@/path/to/document.pdf"
```

**Response (success):**

```json
{
  "status": "success",
  "chunks_created": 10
}
```

**Response (duplicate):**

```json
{
  "error": "Duplicate document: 'document.pdf' has already been ingested."
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

- _"Who is considered the greatest Indian batsman of all time?"_
- _"Tell me about India's first Cricket World Cup victory."_
- _"What is the IPL and when was it founded?"_
- _"Who holds the record for most Test centuries for India?"_
- _"Describe India's cricket performance in the 1983 World Cup."_
- _"Tell me about Rohit Sharma and his contribution to Indian Cricket."_

---

## Tech Stack

| Layer               | Technology                                           |
| ------------------- | ---------------------------------------------------- |
| **LLM**             | `meta/llama-3.2-11b-vision-instruct` via Airawat API |
| **Embeddings**      | `llama-text-embed-v2` (Pinecone hosted inference)    |
| **Vector Database** | Pinecone (serverless, AWS us-east-1, cosine metric)  |
| **Orchestration**   | LangGraph StateGraph, LangChain LCEL                 |
| **PDF Processing**  | PDFPlumber, PyMuPDF, python-magic, textstat          |
| **Backend API**     | Flask + Flask-CORS                                   |
| **Frontend UI**     | Streamlit                                            |
| **Configuration**   | python-dotenv                                        |

---

## Notes

- The Airawat API and Pinecone are cloud-hosted services — valid API keys are required.
- The **same LLM** (`meta/llama-3.2-11b-vision-instruct`) is used for query routing, document grading, hallucination checking, answer grading, and final answer generation.
- The web search tool is a **placeholder** — integrate Tavily, SerpAPI, or DuckDuckGo for production-level web fallback.
- This is an **exploratory/educational project** demonstrating Agentic RAG concepts; it is not based on a production use case.
