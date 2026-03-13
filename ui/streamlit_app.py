"""Streamlit chat interface for the Indian Cricket Agentic RAG system.

Run with::

    streamlit run ui/streamlit_app.py
"""

import os
import sys

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🏏 Indian Cricket RAG",
    page_icon="🏏",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy imports (heavy ML libraries)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading vector store…")
def _load_vector_store():
    from app.vector_store import load_vector_store

    return load_vector_store()


@st.cache_resource(show_spinner="Loading LLM…")
def _load_llm():
    from app.llm import get_llm

    return get_llm()


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
        data_dir = os.path.join(_project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Ingesting document…"):
            try:
                from app.chunker import chunk_documents
                from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CHROMA_DB_PATH
                from app.document_loader import load_and_validate_pdf
                from app.vector_store import create_vector_store

                docs = load_and_validate_pdf(save_path)
                chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
                create_vector_store(chunks, CHROMA_DB_PATH)
                # Clear cached vector store so it reloads
                _load_vector_store.clear()
                st.success(f"✅ Ingested {len(chunks)} chunk(s) from '{uploaded_file.name}'")
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")

    st.markdown("---")

    # System info
    st.subheader("ℹ️ System Info")
    try:
        from app.config import EMBEDDING_MODEL, LLM_MODEL

        st.write(f"**LLM:** `{LLM_MODEL}`")
        st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
        try:
            vs = _load_vector_store()
            chunk_count = len(vs.get()["ids"])
            st.write(f"**Chunks indexed:** {chunk_count}")
        except Exception:
            st.write("**Chunks indexed:** N/A")
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
    "Powered by `google/flan-t5-small` + ChromaDB | "
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

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                vs = _load_vector_store()
                mode_key = st.session_state.mode

                if mode_key == "Simple RAG":
                    from app.rag_chain import query_simple

                    result = query_simple(prompt, vector_store=vs)
                    result["agent_trace"] = []
                else:
                    from app.agents.agentic_rag import query_agentic

                    result = query_agentic(prompt, vector_store=vs)

                answer = result["answer"]
                sources = result.get("sources", [])
                agent_trace = result.get("agent_trace", [])

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
