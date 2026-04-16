'''

"""
utils/config.py
───────────────
Single source of truth for all configuration.
Reads from .env via python-dotenv so no secrets are ever hard-coded.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Centralised configuration object.
    """

    # ── LLM ───────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")

    # THIS IS THE MAIN FIX
   
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.0-pro")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # ── Embeddings ────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Paths ─────────────────────────────────────────────
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./rag/vector_store")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./database/students.db")
    DOCS_UPLOAD_PATH: str = os.getenv("DOCS_UPLOAD_PATH", "./sample_docs")
    LOG_PATH: str = os.getenv("LOG_PATH", "./logs/app.log")

    # ── RAG ───────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 4))

    # ── Router labels ─────────────────────────────────────
    INTENT_LABELS = {
        "rag": "Questions about uploaded documents",
        "sql": "Questions about students database",
        "general": "General questions",
    }


config = Config()

# DEBUG
print(f"DEBUG KEY: {config.GOOGLE_API_KEY[:10]}...") # Only print first 10 chars for safety
print(f"DEBUG MODEL: {config.LLM_MODEL}")  '''


import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """
    Centralised configuration object.
    """

    # ── LLM ───────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")

    # ✅ SAFE & STABLE MODEL (FINAL FIX)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # ── Embeddings ────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )

    # ── Paths ─────────────────────────────────────────────
    VECTOR_DB_PATH: str = os.getenv(
        "VECTOR_DB_PATH",
        "./rag/vector_store"
    )

    DATABASE_PATH: str = os.getenv(
        "DATABASE_PATH",
        "./database/students.db"
    )

    DOCS_UPLOAD_PATH: str = os.getenv(
        "DOCS_UPLOAD_PATH",
        "./sample_docs"
    )

    LOG_PATH: str = os.getenv(
        "LOG_PATH",
        "./logs/app.log"
    )

    # ── RAG ───────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 4))

    # ── Router labels ─────────────────────────────────────
    INTENT_LABELS = {
        "rag": "Questions about uploaded documents",
        "sql": "Questions about students database",
        "general": "General questions",
    }


# Singleton instance
config = Config()


# 🔍 SAFE DEBUG (key masked)
if config.GOOGLE_API_KEY:
    print("DEBUG KEY:", config.GOOGLE_API_KEY[:10] + "...")
else:
    print("DEBUG KEY: NOT FOUND ❌")

print("DEBUG MODEL:", config.LLM_MODEL)