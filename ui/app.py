"""
ui/app.py
----------
Streamlit chat interface for the Multi-Agent Knowledge Assistant.

Run with:  streamlit run ui/app.py

Features:
  - Chat-style message history
  - File upload (PDF, DOCX, TXT) -> ingests into RAG pipeline
  - Agent badge on each response shows which agent answered
  - Sidebar shows system status (vector store size, DB connectivity)
  - Memory clear button
"""

import sys
import os
import tempfile

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from main import Orchestrator
from database.db_setup import seed_database
from rag.document_processor import load_document, chunk_documents
from rag.vector_store import vector_store
from agents.gemini_doc_agent import (
    extract_text_from_file,
    store_document,
    list_documents,
    has_documents,
    clear_documents,
)
from utils.logger import get_logger

log = get_logger("StreamlitUI")


# -- Page config (must be first Streamlit call) ---------------------------------
st.set_page_config(
    page_title="Multi-Agent Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Custom CSS (minimal, clean) ------------------------------------------------
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .badge-rag     { background: #e0f2f1; color: #00695c; }
    .badge-sql     { background: #fff3e0; color: #e65100; }
    .badge-general { background: #e8eaf6; color: #283593; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# -- Session state initialisation -----------------------------------------------
def init_session():
    if "orchestrator" not in st.session_state:
        with st.spinner("🚀 Initialising agents..."):
            seed_database()            # Create DB if it doesn't exist
            st.session_state.orchestrator = Orchestrator()
            st.session_state.messages     = []
            st.session_state.doc_count    = vector_store.doc_count()

            # Add welcome message
            welcome = st.session_state.orchestrator.get_welcome()
            st.session_state.messages.append({
                "role":    "assistant",
                "content": welcome,
                "agent":   "system",
            })


init_session()
orch: Orchestrator = st.session_state.orchestrator


# -- Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Panel")
    st.divider()

    # -- Status indicators --------------------------------------------------
    st.subheader("📊 Status")
    doc_count = vector_store.doc_count()
    col1, col2 = st.columns(2)
    col1.metric("Vectors", f"{doc_count:,}")
    col2.metric("Memory turns", len(orch.memory))

    # DB status
    try:
        from database.db_setup import get_connection
        conn = get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM students")
        student_count = cursor.fetchone()[0]
        conn.close()
        st.success(f"✅ Database: {student_count} students")
    except Exception as e:
        st.error(f"❌ DB Error: {e}")

    st.divider()

    # -- Document upload ----------------------------------------------------
    st.subheader("📂 Upload Documents")
    st.caption("PDF, DOCX, or TXT — Gemini will read and analyse the full content.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", type="primary", use_container_width=True):
            total_chunks = 0
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        suffix = os.path.splitext(uploaded_file.name)[1]

                        # Write to temp file (Windows-safe)
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tfile.write(uploaded_file.read())
                        tmp_path = tfile.name
                        tfile.close()

                        try:
                            # ── 1. Extract raw text → GeminiDocAgent (native analysis) ──
                            raw_text = extract_text_from_file(tmp_path, uploaded_file.name)
                            store_document(uploaded_file.name, raw_text)

                            # ── 2. Also chunk → FAISS for RAG fallback ───────────────
                            docs   = load_document(tmp_path)
                            chunks = chunk_documents(docs)
                            for c in chunks:
                                c.metadata["source"] = uploaded_file.name
                            vector_store.add_documents(chunks)
                            total_chunks += len(chunks)

                            st.success(
                                f"✅ **{uploaded_file.name}** — "
                                f"{len(raw_text):,} chars extracted, "
                                f"{len(chunks)} chunks indexed"
                            )
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"✗ {uploaded_file.name}: {e}")

            st.session_state.doc_count = vector_store.doc_count()
            st.info(f"Vectors in FAISS store: {vector_store.doc_count():,}")
            st.rerun()

    # Show currently loaded docs
    loaded = list_documents()
    if loaded:
        st.divider()
        st.caption("📄 **Loaded documents (Gemini will read these):**")
        for fname in loaded:
            st.markdown(f"&nbsp;&nbsp;📄 `{fname}`", unsafe_allow_html=True)
        if st.button("🗑️ Clear Documents", use_container_width=True, key="clear_docs"):
            clear_documents()
            st.rerun()
    else:
        st.caption("_No documents loaded yet._")

    st.divider()

    # -- Example queries ----------------------------------------------------
    st.subheader("💡 Example Queries")

    doc_examples = [
        "Summarise the uploaded document",
        "What are the key points in the PDF?",
        "What does the document say about [topic]?",
        "List the main conclusions from the document",
    ]

    sql_examples = [
        "Show students with marks above 80",
        "Average score per subject",
        "Top 3 students by total marks",
        "Students who failed (marks < 40)",
    ]

    general_examples = [
        "What is RAG in AI?",
        "Explain vector databases",
        "Hello! What can you do?",
    ]

    st.caption("📄 Document Q&A (upload a PDF first)")
    for q in doc_examples:
        if st.button(q, use_container_width=True, key=f"doc_{q}"):
            st.session_state["pending_query"] = q

    st.caption("🗄️ Database queries")
    for q in sql_examples:
        if st.button(q, use_container_width=True, key=f"sql_{q}"):
            st.session_state["pending_query"] = q

    st.caption("🤖 General questions")
    for q in general_examples:
        if st.button(q, use_container_width=True, key=f"gen_{q}"):
            st.session_state["pending_query"] = q

    st.divider()

    # -- Clear memory -------------------------------------------------------
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        orch.clear_memory()
        # Keep only the welcome message
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()


# -- Main chat area -------------------------------------------------------------
st.title("🎓 Multi-Agent Knowledge Assistant")
st.caption("Powered by Claude · LangChain · FAISS · SQLite")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle example query button clicks
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")
else:
    user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, agent_name = orch.process(user_input)
            except Exception as e:
                log.error(f"Orchestrator error: {e}")
                answer     = f"⚠️ System error: {e}"
                agent_name = "error"

        st.markdown(answer)

    # Store in history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "agent":   agent_name,
    })
    st.rerun()
