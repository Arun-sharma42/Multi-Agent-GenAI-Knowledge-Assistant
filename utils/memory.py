"""
utils/memory.py
---------------
Simple in-session conversation memory shared across all agents.
Each turn is stored so any agent can reference prior context.

Interview talking point:
  "Memory is centralised -- the router, RAG agent, and SQL agent all
   write to and read from the same ConversationMemory instance, so
   follow-up questions like 'tell me more about that' work correctly."
"""

from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime


@dataclass
class Turn:
    """A single conversation turn."""
    role:      Literal["user", "assistant"]
    content:   str
    agent_used: str = "unknown"
    timestamp: str  = field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory:
    """
    Lightweight conversation history store.
    Thread-safe enough for Streamlit's single-user sessions.
    """

    def __init__(self, max_turns: int = 20):
        self._history: list[Turn] = []
        self.max_turns = max_turns   # Prevent unbounded context growth

    # -- Write -----------------------------------------------------------------

    def add_user(self, message: str) -> None:
        self._history.append(Turn(role="user", content=message))
        self._trim()

    def add_assistant(self, message: str, agent_used: str = "unknown") -> None:
        self._history.append(
            Turn(role="assistant", content=message, agent_used=agent_used)
        )
        self._trim()

    # -- Read ------------------------------------------------------------------

    def get_history(self) -> list[Turn]:
        return list(self._history)

    def get_recent_context(self, n: int = 4) -> str:
        """Return the last n turns as a plain string for prompt injection."""
        recent = self._history[-n:]
        lines = [f"{t.role.upper()}: {t.content}" for t in recent]
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()

    # -- Internal --------------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the last max_turns entries."""
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def __len__(self) -> int:
        return len(self._history)


# Singleton for the Streamlit session (imported by main.py)
shared_memory = ConversationMemory()
