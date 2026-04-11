"""
utils/config.py
───────────────
Single source of truth for all configuration.
Reads from .env via python-dotenv so no secrets are ever hard-coded.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above utils/)
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """
    Centralised configuration object.
    Access from anywhere:  from utils.config import config
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str  = os.getenv("LLM_PROVIDER", "anthropic")
    LLM_MODEL: str     = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")

    # ── Embeddings (local, no key needed) ────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Paths ─────────────────────────────────────────────────────────────────
    VECTOR_DB_PATH:   str = os.getenv("VECTOR_DB_PATH",   "./rag/vector_store")
    DATABASE_PATH:    str = os.getenv("DATABASE_PATH",    "./database/students.db")
    DOCS_UPLOAD_PATH: str = os.getenv("DOCS_UPLOAD_PATH", "./sample_docs")
    LOG_PATH:         str = os.getenv("LOG_PATH",         "./logs/app.log")

    # ── RAG ───────────────────────────────────────────────────────────────────
    CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE",    500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP",  50))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS",   4))

    # ── Router prompt labels (used by RouterAgent) ────────────────────────────
    INTENT_LABELS = {
        "rag":     "Questions about uploaded documents or knowledge base",
        "sql":     "Questions about students, marks, scores, database records",
        "general": "General questions, greetings, or anything else",
    }


# Singleton instance — import this everywhere
config = Config()
