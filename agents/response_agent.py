"""
agents/response_agent.py
─────────────────────────
The final stage in the pipeline.
Takes a raw AgentResponse and polishes it for display:
  - Adds agent attribution badge
  - Attaches source citations for RAG responses
  - Adds a footer with the SQL used (for SQL responses)
  - Handles error formatting

Interview talking point:
  "Separating presentation from logic means I can change the UI
   formatting without touching any agent business logic."
"""

from agents.base_agent import AgentResponse


class ResponseAgent:
    """
    Formats any AgentResponse into a clean, user-facing Markdown string.
    This is NOT a BaseAgent subclass because it doesn't call the LLM —
    it's purely a formatting/presentation layer.
    """

    AGENT_LABELS = {
        "RAGAgent":     ("📄", "Knowledge Base"),
        "SQLAgent":     ("🗄️",  "Database Query"),
        "GeneralAgent": ("🤖", "AI Assistant"),
        "RouterAgent":  ("🔀", "Router"),
    }

    def format(self, response: AgentResponse) -> str:
        """
        Return a clean Markdown string ready for display in the chat UI.
        """
        emoji, label = self.AGENT_LABELS.get(
            response.agent_name, ("💬", response.agent_name)
        )

        # ── Agent attribution badge ────────────────────────────────────────
        header = f"{emoji} **{label}**\n\n"

        # ── Main answer ───────────────────────────────────────────────────
        body = response.answer

        # ── RAG: append source citations ──────────────────────────────────
        sources = response.metadata.get("sources", [])
        if sources and response.agent_name == "RAGAgent":
            source_list = "\n".join(f"  - `{s}`" for s in sources)
            body += f"\n\n---\n📎 **Sources used:**\n{source_list}"

        # ── Error state ───────────────────────────────────────────────────
        if not response.success:
            body = f"⚠️ **Something went wrong:**\n\n{body}"

        return header + body

    def format_welcome(self) -> str:
        """Return the initial greeting message shown when the app loads."""
        return """
🎓 **Multi-Agent Knowledge Assistant**

I'm your AI-powered assistant with three capabilities:

| Capability | What it does | Example query |
|---|---|---|
| 📄 **RAG** | Answers from uploaded docs | *"Summarise the uploaded PDF"* |
| 🗄️ **SQL** | Queries the student database | *"Show students with marks > 80"* |
| 🤖 **General** | General AI assistant | *"What is machine learning?"* |

**To get started:**
1. Upload a document using the sidebar (PDF, DOCX, or TXT)
2. Ask any question — I'll automatically route it to the right agent

---
*Built with Claude · LangChain · FAISS · SQLite · Streamlit*
""".strip()
