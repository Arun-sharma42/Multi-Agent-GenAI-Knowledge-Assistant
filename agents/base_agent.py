"""
agents/base_agent.py
---------------------
Abstract base class that every agent inherits.
Enforces a consistent interface: every agent must implement .run().

Interview talking point:
  "Using an ABC means I can plug in new agents without changing any
   orchestration code -- open/closed principle in practice."
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from utils.logger import get_logger


class AgentResponse(BaseModel):
    """
    Typed return value for every agent.
    Pydantic ensures the orchestrator always gets a well-formed object.
    """
    answer:     str           # The main text response shown to the user
    agent_name: str           # Which agent produced this
    metadata:   dict = {}     # Extra data (SQL query used, doc sources, etc.)
    success:    bool = True   # False if the agent hit an error


class BaseAgent(ABC):
    """
    All agents inherit from this.
    Provides shared logging and enforces the .run() contract.
    """

    def __init__(self, name: str):
        self.name = name
        self.log  = get_logger(name)

    @abstractmethod
    def run(self, query: str, context: str = "") -> AgentResponse:
        """
        Process a user query and return a structured AgentResponse.

        Args:
            query:   The user's question.
            context: Optional conversation history or extra context.
        """
        ...

    def _safe_run(self, query: str, context: str = "") -> AgentResponse:
        """
        Wraps run() with error handling.
        Call this from the orchestrator instead of run() directly.
        """
        try:
            self.log.info(f"Processing: '{query[:80]}...'")
            result = self.run(query, context)
            self.log.info(f"Completed -- success={result.success}")
            return result
        except Exception as e:
            self.log.error(f"Agent error: {e}")
            return AgentResponse(
                answer=f"⚠️ Sorry, the {self.name} encountered an error: {e}",
                agent_name=self.name,
                success=False,
                metadata={"error": str(e)},
            )
