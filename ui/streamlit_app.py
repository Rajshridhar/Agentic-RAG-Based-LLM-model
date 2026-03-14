"""Streamlit chat interface for the Indian Cricket Agentic RAG system.

Run with::

    streamlit run ui/streamlit_app.py

Requires the Flask backend to be running::

    python api/flask_app.py
"""

import os
import sys

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import requests
import streamlit as st

from app.config import FLASK_HOST, FLASK_PORT

# Flask API base URL
_API_HOST = "127.0.0.1" if FLASK_HOST == "0.0.0.0" else FLASK_HOST
_API_BASE = f"http://{_API_HOST}:{FLASK_PORT}"

# Page configuration (must be the first Streamlit call)
st.set_page_config(
    page_title="🏏 Indian Cricket RAG",
    page_icon="🏏",
    layout="wide",
)


def _check_backend():
    """Return True if Flask backend is reachable."""
    try:
        r = requests.get(f"{_API_BASE}/api/health", timeout=5)
        return r.ok
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Agentic RAG"

if "show_trace" not in st.session_state:
    st.session_state.show_trace = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏏 Indian Cricket RAG")
    st.markdown("---")

    # Mode toggle
    st.session_state.mode = st.radio(
        "RAG Mode",
        ["Agentic RAG", "Simple RAG"],
        index=0,
        help=(
            "**Agentic RAG** uses LangGraph agents for query routing, document "
            "grading, hallucination checking, and answer grading.\n\n"
            "**Simple RAG** is a direct retrieval → generate chain."
        ),
    )

    # Show/hide agent trace
    st.session_state.show_trace = st.checkbox(
        "Show agent reasoning trace",
        value=st.session_state.show_trace,
    )

    st.markdown("---")

    # PDF upload
    st.subheader("📄 Upload New Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Ingesting document…"):
            try:
                resp = requests.post(
                    f"{_API_BASE}/api/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )
                data = resp.json()
                if resp.ok:
                    st.success(f"✅ Ingested {data.get('chunks_created', '?')} chunk(s) from '{uploaded_file.name}'")
                else:
                    st.error(f"❌ Ingestion failed: {data.get('error', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("❌ Backend not reachable. Start it with: `python api/flask_app.py`")
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")

    st.markdown("---")

    # System info
    st.subheader("ℹ️ System Info")
    try:
        resp = requests.get(f"{_API_BASE}/api/health", timeout=5)
        if resp.ok:
            info = resp.json()
            st.write(f"**LLM:** `{info.get('model', 'N/A')}`")
            st.write(f"**Embeddings:** `{info.get('embedding_model', 'N/A')}`")
            st.write(f"**Chunks indexed:** {info.get('chunks', 'N/A')}")
            st.write("**Backend:** 🟢 Connected")
        else:
            st.write("**Backend:** 🔴 Error")
    except requests.ConnectionError:
        st.write("**Backend:** 🔴 Not running")
        st.warning("Start backend: `python api/flask_app.py`")
    except Exception:
        st.write("System info unavailable.")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🏏 Ask about Indian Cricket History")
st.caption(
    "Powered by `meta/llama-3.2-11b-vision-instruct` (Airawat API) + Pinecone | "
    f"Mode: **{st.session_state.mode}**"
)

# Replay existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **Page {s.get('page', 'N/A')}** (chunk {s.get('chunk_id', 'N/A')}): "
                        f"{s.get('content_preview', '')[:200]}…"
                    )
        if st.session_state.show_trace and msg.get("agent_trace"):
            with st.expander("🔍 Agent reasoning trace"):
                for step in msg["agent_trace"]:
                    st.write(f"→ {step}")

# Chat input
if prompt := st.chat_input("Ask a question about Indian Cricket…"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response via Flask API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                mode_key = st.session_state.mode
                api_mode = "simple" if mode_key == "Simple RAG" else "agentic"

                resp = requests.post(
                    f"{_API_BASE}/api/query",
                    json={"question": prompt, "mode": api_mode},
                    timeout=300,
                )
                result = resp.json()

                if resp.ok:
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources", [])
                    agent_trace = result.get("agent_trace", [])
                else:
                    answer = f"❌ Error: {result.get('error', 'Unknown error')}"
                    sources = []
                    agent_trace = []

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.markdown(
                                f"- **Page {s.get('page', 'N/A')}** (chunk {s.get('chunk_id', 'N/A')}): "
                                f"{s.get('content_preview', '')[:200]}…"
                            )

                if st.session_state.show_trace and agent_trace:
                    with st.expander("🔍 Agent reasoning trace"):
                        for step in agent_trace:
                            st.write(f"→ {step}")

            except requests.ConnectionError:
                answer = "❌ Backend not reachable. Start it with: `python api/flask_app.py`"
                sources = []
                agent_trace = []
                st.error(answer)
            except Exception as exc:
                answer = f"❌ Error: {exc}"
                sources = []
                agent_trace = []
                st.error(answer)

    # Persist assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "agent_trace": agent_trace,
        }
    )
