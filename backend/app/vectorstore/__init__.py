"""Vector store integration for SolVX Knowledge Core."""

from backend.app.vectorstore.chroma_store import ChromaVectorStore, get_vector_store

__all__ = ["ChromaVectorStore", "get_vector_store"]
