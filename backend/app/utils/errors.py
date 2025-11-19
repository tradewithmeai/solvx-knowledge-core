"""Custom exception classes and error handling."""

from typing import Any


class SolVXError(Exception):
    """Base exception for SolVX errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ParsingError(SolVXError):
    """Raised when document parsing fails."""

    pass


class EmbeddingError(SolVXError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(SolVXError):
    """Raised when vector store operations fail."""

    pass


class SearchError(SolVXError):
    """Raised when search operations fail."""

    pass


class RAGError(SolVXError):
    """Raised when RAG operations fail."""

    pass


class ConfigurationError(SolVXError):
    """Raised when configuration is invalid."""

    pass


class RateLimitError(SolVXError):
    """Raised when rate limits are exceeded."""

    pass
