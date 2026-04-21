"""
rag/vector_store.py
--------------------
Manages the FAISS vector database:
  - Builds embeddings using sentence-transformers (runs locally, free)
  - Saves / loads the index to/from disk
  - Exposes a similarity_search() method used by RAGAgent

Interview talking point:
  "I chose sentence-transformers for embeddings because they run
   locally with no API cost -- good for a student project. In production
   I'd swap to text-embedding-3-small for higher accuracy."
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import config
from utils.logger import get_logger

log = get_logger("VectorStore")


class VectorStoreManager:
    """
    Wraps FAISS to provide a clean add / search interface.
    Auto-loads an existing index from disk if one exists.
    """

    def __init__(self):
        self._store: Optional[FAISS] = None
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # Change to "cuda" if you have GPU
        )
        self._index_path = Path(config.VECTOR_DB_PATH)

        # Auto-load existing index on startup
        if self._index_exists():
            self._load()

    # -- Public API -------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and add documents to the vector store.
        Returns the number of documents added.
        """
        if not documents:
            log.warning("add_documents called with empty list -- nothing to do")
            return 0

        log.info(f"Embedding {len(documents)} chunks...")

        if self._store is None:
            # First time -- build from scratch
            self._store = FAISS.from_documents(documents, self._embeddings)
        else:
            # Append to existing index
            self._store.add_documents(documents)

        self._save()
        log.info(f"Vector store now contains {self.doc_count()} vectors")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = None,
    ) -> List[Document]:
        """
        Return the k most relevant document chunks for a query.
        Raises RuntimeError if the store is empty.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is empty. Upload documents first."
            )

        k = k or config.TOP_K_RESULTS
        results = self._store.similarity_search(query, k=k)
        log.info(f"Retrieved {len(results)} chunks for: '{query[:60]}'")
        return results

    def doc_count(self) -> int:
        """Return approximate number of vectors stored."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def clear(self) -> None:
        """Wipe the vector store from memory and disk."""
        self._store = None
        if self._index_exists():
            import shutil
            shutil.rmtree(self._index_path)
        log.info("Vector store cleared")

    # -- Private helpers --------------------------------------------------------

    def _save(self) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_path))
        log.debug(f"Index saved -> {self._index_path}")

    def _load(self) -> None:
        log.info(f"Loading existing FAISS index from {self._index_path}")
        self._store = FAISS.load_local(
            str(self._index_path),
            self._embeddings,
            allow_dangerous_deserialization=True,  # Safe -- our own files
        )
        log.info(f"Loaded {self.doc_count()} vectors")

    def _index_exists(self) -> bool:
        return (self._index_path / "index.faiss").exists()


# Singleton -- shared across all agents in the session
vector_store = VectorStoreManager()
