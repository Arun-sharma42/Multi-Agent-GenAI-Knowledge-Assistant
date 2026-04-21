"""
tests/test_agents.py
---------------------
Unit tests for the multi-agent system.
Uses unittest.mock to avoid real API calls during testing --
critical for CI/CD pipelines and development without burning API credits.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Make sure imports resolve from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -- Test: Router Agent ---------------------------------------------------------

class TestRouterAgent(unittest.TestCase):

    def test_routes_to_sql(self):
        """Router should classify student/marks queries as 'sql'."""
        import agents.router_agent as ra
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="sql")

        with patch.object(ra, "get_llm", return_value=mock_llm):
            from agents.router_agent import RouterAgent
            router = RouterAgent()
            result = router.run("Show students with marks above 80")

        self.assertEqual(result.metadata["route"], "sql")
        self.assertTrue(result.success)

    def test_routes_to_rag(self):
        """Router should classify document queries as 'rag'."""
        import agents.router_agent as ra
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="rag")

        with patch.object(ra, "get_llm", return_value=mock_llm):
            from agents.router_agent import RouterAgent
            router = RouterAgent()
            result = router.run("What does the PDF say about neural networks?")

        self.assertEqual(result.metadata["route"], "rag")

    def test_fallback_on_unknown_route(self):
        """Router should default to 'general' for unexpected LLM output."""
        import agents.router_agent as ra
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="BANANA")  # Invalid

        with patch.object(ra, "get_llm", return_value=mock_llm):
            from agents.router_agent import RouterAgent
            router = RouterAgent()
            result = router.run("Hello!")

        self.assertEqual(result.metadata["route"], "general")


# -- Test: SQL Agent ------------------------------------------------------------

class TestSQLAgent(unittest.TestCase):

    def test_sql_validation_blocks_delete(self):
        """SQL agent must reject DELETE statements."""
        from agents.sql_agent import SQLAgent
        agent = SQLAgent.__new__(SQLAgent)  # Skip __init__ (avoids LLM init)
        agent.name = "SQLAgent"
        agent.log  = MagicMock()

        error = agent._validate_sql("DELETE FROM students WHERE id=1")
        self.assertIsNotNone(error)
        self.assertIn("DELETE", error)

    def test_sql_validation_blocks_drop(self):
        """SQL agent must reject DROP statements."""
        from agents.sql_agent import SQLAgent
        agent = SQLAgent.__new__(SQLAgent)
        agent.name = "SQLAgent"
        agent.log  = MagicMock()

        error = agent._validate_sql("DROP TABLE students")
        self.assertIsNotNone(error)

    def test_sql_validation_allows_select(self):
        """SQL agent must allow valid SELECT statements."""
        from agents.sql_agent import SQLAgent
        agent = SQLAgent.__new__(SQLAgent)
        agent.name = "SQLAgent"
        agent.log  = MagicMock()

        error = agent._validate_sql("SELECT * FROM students WHERE age > 20")
        self.assertIsNone(error)

    def test_format_results_returns_markdown_table(self):
        """Result formatter should produce valid Markdown tables."""
        from agents.sql_agent import SQLAgent
        agent = SQLAgent.__new__(SQLAgent)
        agent.name = "SQLAgent"
        agent.log  = MagicMock()

        rows    = [{"name": "Alice", "marks": 92.0}]
        columns = ["name", "marks"]
        result  = agent._format_results(rows, columns)

        self.assertIn("| name | marks |", result)
        self.assertIn("| Alice | 92.0 |", result)

    def test_format_results_empty(self):
        """Formatter should handle empty result sets gracefully."""
        from agents.sql_agent import SQLAgent
        agent = SQLAgent.__new__(SQLAgent)
        agent.name = "SQLAgent"
        agent.log  = MagicMock()

        result = agent._format_results([], [])
        self.assertIn("no rows", result.lower())


# -- Test: Memory ---------------------------------------------------------------

class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        from utils.memory import ConversationMemory
        self.memory = ConversationMemory(max_turns=5)

    def test_stores_user_and_assistant_turns(self):
        self.memory.add_user("Hello")
        self.memory.add_assistant("Hi there!", agent_used="GeneralAgent")
        self.assertEqual(len(self.memory), 2)

    def test_trims_to_max_turns(self):
        for i in range(10):
            self.memory.add_user(f"Message {i}")
        self.assertEqual(len(self.memory), 5)

    def test_get_recent_context_returns_string(self):
        self.memory.add_user("What is RAG?")
        self.memory.add_assistant("RAG stands for...")
        context = self.memory.get_recent_context(n=2)
        self.assertIsInstance(context, str)
        self.assertIn("What is RAG?", context)

    def test_clear_resets_memory(self):
        self.memory.add_user("test")
        self.memory.clear()
        self.assertEqual(len(self.memory), 0)


# -- Test: Document Processor ---------------------------------------------------

class TestDocumentProcessor(unittest.TestCase):

    def test_chunk_documents_respects_chunk_size(self):
        """Chunks should not exceed CHUNK_SIZE significantly."""
        from langchain_core.documents import Document
        from rag.document_processor import chunk_documents
        from utils.config import config

        # Create a long document
        long_text = "This is a sentence about artificial intelligence. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test"})]

        chunks = chunk_documents(docs)

        # Each chunk should be approximately within chunk size
        for chunk in chunks:
            self.assertLessEqual(
                len(chunk.page_content),
                config.CHUNK_SIZE * 1.5,  # Allow some tolerance
                "Chunk exceeded expected max size",
            )

    def test_unsupported_file_type_raises(self):
        """Unsupported file types should raise ValueError."""
        from rag.document_processor import load_document
        with self.assertRaises(ValueError):
            load_document("document.xyz")


# -- Test: Response Agent -------------------------------------------------------

class TestResponseAgent(unittest.TestCase):

    def setUp(self):
        from agents.response_agent import ResponseAgent
        from agents.base_agent import AgentResponse
        self.formatter = ResponseAgent()
        self.AgentResponse = AgentResponse

    def test_rag_response_includes_sources(self):
        response = self.AgentResponse(
            answer="RAG stands for Retrieval Augmented Generation.",
            agent_name="RAGAgent",
            metadata={"sources": ["ai_knowledge_base.txt"]},
        )
        formatted = self.formatter.format(response)
        self.assertIn("Sources", formatted)
        self.assertIn("ai_knowledge_base.txt", formatted)

    def test_error_response_shows_warning(self):
        response = self.AgentResponse(
            answer="Something broke.",
            agent_name="SQLAgent",
            success=False,
        )
        formatted = self.formatter.format(response)
        self.assertIn("went wrong", formatted.lower())

    def test_welcome_message_is_non_empty(self):
        welcome = self.formatter.format_welcome()
        self.assertGreater(len(welcome), 100)


# -- Run all tests --------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
