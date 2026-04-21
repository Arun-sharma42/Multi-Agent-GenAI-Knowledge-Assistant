"""
agents/router_agent.py
-----------------------
The entry point for every user message.
Uses the LLM to classify intent, then returns which downstream
agent should handle the query.

Intent categories
-----------------
  "rag"     -> User is asking about uploaded documents
  "sql"     -> User is asking about students / marks / database records
  "general" -> General conversation, greetings, anything else

Interview talking point:
  "The router uses a zero-shot classification prompt rather than
   keyword matching, so it handles paraphrases and typos gracefully."
"""

from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent, AgentResponse
from utils.llm_client import get_llm
from utils.config import config


ROUTER_SYSTEM_PROMPT = """
You are an intent classification assistant for a multi-agent AI system.
Classify the user's query into EXACTLY ONE of these categories:

  rag     - The user is asking about ANY factual topic, broad knowledge, or specific details 
            that might be in uploaded documents (e.g., science, AI, project notes, company info).
  sql     - The user is asking about quantitative student data: marks, scores, list of names, 
            ages, grades, or specific results from the 'students' database.
  general - Purely conversational items: greetings (hello/hi), thanks, or questions 
            about your own capabilities.

Respond with ONLY the category label (lowercase). No explanation.

Examples:
  "What does the document say about neural networks?" -> rag
  "What is machine learning?"                         -> rag
  "Explain the transformer architecture"               -> rag
  "Show students who scored above 80 in Maths"        -> sql
  "What is Bob's average mark?"                       -> sql
  "List all students in the database"                 -> sql
  "Hello, how are you?"                               -> general
  "Who are you?"                                      -> general
"""


class RouterAgent(BaseAgent):
    """
    Classifies user intent and decides which agent handles the query.
    """

    def __init__(self):
        super().__init__("RouterAgent")
        # Use temperature=0 for deterministic routing
        self.llm = get_llm(temperature=0)

    def run(self, query: str, context: str = "") -> AgentResponse:
        """
        Classify the query and return the chosen route in metadata.
        """
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]

        raw = self.llm.invoke(messages)
        content = raw.content
        if isinstance(content, list):
            content = "".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content])
        
        # Robust parsing: Look for the intent labels inside the LLM answer
        res = str(content).lower()
        if "sql" in res:
            route = "sql"
        elif "rag" in res:
            route = "rag"
        else:
            route = "general"

        self.log.info(f"Routed '{query[:60]}' -> {route.upper()}")

        return AgentResponse(
            answer=f"Routing to **{route.upper()} agent**...",
            agent_name=self.name,
            metadata={"route": route},
        )
