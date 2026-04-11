"""
agents/rag_agent.py
────────────────────
Retrieval-Augmented Generation agent.
Flow:
  1. Receive user query
  2. Embed query → search vector store → retrieve top-k chunks
  3. Build a prompt with retrieved context + conversation history
  4. Send to LLM → return grounded answer with source citations

Interview talking point:
  "RAG solves the hallucination problem by giving the LLM real evidence
   to work from. The model is instructed to say 'I don't know' if the
   context doesn't contain the answer — preventing confabulation."
"""

from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent, AgentResponse
from rag.vector_store import vector_store
from utils.llm_client import get_llm


RAG_SYSTEM_PROMPT = """
You are a knowledgeable assistant that answers questions ONLY using the provided context.

Rules:
1. Base your answer strictly on the CONTEXT below.
2. If the context does not contain enough information, say: "I don't have enough information in the uploaded documents to answer this."
3. Be concise and clear.
4. At the end, list which document(s) your answer came from under "Sources:".
5. Never fabricate information that isn't in the context.
"""


class RAGAgent(BaseAgent):
    """
    Answers questions grounded in uploaded documents.
    """

    def __init__(self):
        super().__init__("RAGAgent")
        self.llm = get_llm(temperature=0.2)

    def run(self, query: str, context: str = "") -> AgentResponse:
        """
        Retrieve relevant chunks, then generate a grounded answer.
        """
        # ── Step 1: Retrieve relevant document chunks ──────────────────────
        try:
            chunks = vector_store.similarity_search(query)
        except RuntimeError as e:
            return AgentResponse(
                answer=(
                    "📂 No documents have been uploaded yet.\n\n"
                    "Please upload a PDF, DOCX, or TXT file using the sidebar, "
                    "then ask your question again."
                ),
                agent_name=self.name,
                success=False,
                metadata={"error": str(e)},
            )

        # ── Step 2: Format retrieved context ──────────────────────────────
        context_text  = self._format_chunks(chunks)
        source_labels = self._extract_sources(chunks)

        self.log.debug(f"Using {len(chunks)} chunks from: {source_labels}")

        # ── Step 3: Build prompt and call LLM ─────────────────────────────
        user_message = f"""
CONTEXT (from uploaded documents):
{context_text}

CONVERSATION HISTORY:
{context if context else "No prior conversation."}

QUESTION:
{query}
"""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        answer   = response.content.strip()

        return AgentResponse(
            answer=answer,
            agent_name=self.name,
            metadata={
                "sources":     source_labels,
                "chunks_used": len(chunks),
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_chunks(self, chunks) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, doc in enumerate(chunks, 1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            parts.append(
                f"[{i}] Source: {source} (page {page})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks) -> list[str]:
        """Pull unique source filenames from chunk metadata."""
        import os
        seen    = set()
        sources = []
        for doc in chunks:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources
