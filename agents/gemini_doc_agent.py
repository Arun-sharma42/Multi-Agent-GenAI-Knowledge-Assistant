"""
agents/gemini_doc_agent.py
---------------------------
A dedicated document Q&A agent that uses the Gemini API natively
to read and analyse uploaded PDF / DOCX / TXT files.

Unlike the RAGAgent (which chunks text → FAISS → retrieve → LLM),
this agent sends the FULL extracted text of the uploaded documents
directly to Gemini so it can reason over the entire content.

Flow:
  1. Receive user query + list of uploaded document texts
  2. Build a prompt: [system] + [all doc content] + [question]
  3. Send to Gemini via LangChain → get grounded answer
  4. Return AgentResponse with source filenames

Interview talking point:
  "Native document understanding lets Gemini reason across the whole
   document without losing context from chunking. For large files we
   still chunk because Gemini has a token limit, but for typical
   PDFs (<100 pages) the full-text approach is more accurate."
"""

import os
from typing import List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent, AgentResponse
from utils.llm_client import get_llm
from utils.logger import get_logger

log = get_logger("GeminiDocAgent")

# Session-level document store: { filename: extracted_text }
_uploaded_docs: Dict[str, str] = {}


# ── Public helpers used by the Streamlit UI ──────────────────────────────────

def store_document(filename: str, text: str) -> None:
    """Save extracted document text into the in-memory store."""
    _uploaded_docs[filename] = text
    log.info(f"Stored doc '{filename}' ({len(text):,} chars)")


def clear_documents() -> None:
    """Remove all stored documents (e.g. on session reset)."""
    _uploaded_docs.clear()
    log.info("Document store cleared")


def list_documents() -> List[str]:
    """Return names of currently loaded documents."""
    return list(_uploaded_docs.keys())


def has_documents() -> bool:
    return len(_uploaded_docs) > 0


# ── Document text extraction helpers ─────────────────────────────────────────

def extract_text_from_file(file_path: str, original_name: str) -> str:
    """
    Extract plain text from a PDF, DOCX, or TXT file.
    Returns the extracted text string.
    """
    ext = os.path.splitext(original_name)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image — try uploading a text-based PDF."
        )
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("No text found in this DOCX file.")
    return "\n".join(paragraphs)


def _extract_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ── The Agent ─────────────────────────────────────────────────────────────────

DOC_SYSTEM_PROMPT = """You are an expert document analyst assistant.

The user has uploaded one or more documents. Your job is to:
1. Answer questions accurately based ONLY on the provided document content.
2. Quote or paraphrase relevant sections to support your answer.
3. If the answer is not found in the documents, clearly say so.
4. At the end of your answer, mention which document(s) you used.

Be thorough, accurate, and cite specific page numbers or sections when possible.
"""


class GeminiDocAgent(BaseAgent):
    """
    Answers questions about uploaded documents by sending the full
    document text to Gemini for native comprehension.
    """

    def __init__(self):
        super().__init__("GeminiDocAgent")
        self.llm = get_llm(temperature=0.1)

    def run(self, query: str, context: str = "") -> AgentResponse:
        """
        Build a prompt with ALL uploaded document text and ask Gemini.
        """
        if not has_documents():
            return AgentResponse(
                answer=(
                    "📂 **No documents have been uploaded yet.**\n\n"
                    "Please upload a PDF, DOCX, or TXT file using the "
                    "**📂 Upload Documents** section in the sidebar, then ask your question."
                ),
                agent_name=self.name,
                success=False,
            )

        # Build the document context block
        doc_blocks = []
        for fname, text in _uploaded_docs.items():
            # Truncate very large documents to stay within token limits
            # Gemini Flash has ~1M token context but we keep it reasonable
            max_chars = 400_000  # ~100k tokens
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... document truncated for length ...]"
            doc_blocks.append(f"=== DOCUMENT: {fname} ===\n{text}\n=== END OF {fname} ===")

        all_docs_text = "\n\n".join(doc_blocks)
        doc_names = list(_uploaded_docs.keys())

        user_message = f"""{all_docs_text}

---
CONVERSATION HISTORY:
{context if context else "None"}

USER QUESTION:
{query}
"""

        messages = [
            SystemMessage(content=DOC_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        self.log.info(
            f"Sending {len(doc_blocks)} doc(s) to Gemini "
            f"({sum(len(t) for t in _uploaded_docs.values()):,} chars total)"
        )

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in content
            )

        return AgentResponse(
            answer=str(content).strip(),
            agent_name=self.name,
            metadata={"sources": doc_names, "doc_count": len(doc_names)},
        )
