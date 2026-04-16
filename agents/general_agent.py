"""
agents/general_agent.py
────────────────────────
Fallback agent for queries that don't need RAG or SQL.
Uses conversation history for continuity (follow-up questions work).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent, AgentResponse
from utils.llm_client import get_llm


GENERAL_SYSTEM_PROMPT = """
You are a helpful, concise AI assistant for a university knowledge management system.

You help students and faculty with:
- General knowledge questions
- Explaining AI/ML concepts
- Helping understand data and analytics

Keep responses clear and structured. Use bullet points or numbered lists when helpful.
If the user seems to be asking about documents or database records, gently remind them
they can upload files or ask about student data.
"""


class GeneralAgent(BaseAgent):

    def __init__(self):
        super().__init__("GeneralAgent")
        self.llm = get_llm(temperature=0.5)

    def run(self, query: str, context: str = "") -> AgentResponse:
        user_message = query
        if context:
            user_message = f"Conversation so far:\n{context}\n\nCurrent question: {query}"

        messages = [
            SystemMessage(content=GENERAL_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content])

        return AgentResponse(
            answer=str(content).strip(),
            agent_name=self.name,
        )
