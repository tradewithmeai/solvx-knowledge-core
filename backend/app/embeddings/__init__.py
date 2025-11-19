"""Embeddings cache and utilities module."""

from .cache import EmbeddingsCache, get_cache_instance
from .providers import (
    CohereProvider,
    EmbeddingAPIError,
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingModelError,
    EmbeddingProvider,
    EmbeddingRateLimitError,
    LocalProvider,
    OpenAIProvider,
    VoyageProvider,
    embed_texts,
    get_embedding_metadata,
    get_embedding_provider,
)

__all__ = [
    # Cache
    "EmbeddingsCache",
    "get_cache_instance",
    # Providers
    "EmbeddingProvider",
    "OpenAIProvider",
    "CohereProvider",
    "VoyageProvider",
    "LocalProvider",
    # Exceptions
    "EmbeddingError",
    "EmbeddingAPIError",
    "EmbeddingRateLimitError",
    "EmbeddingAuthenticationError",
    "EmbeddingModelError",
    # Factory and convenience functions
    "get_embedding_provider",
    "embed_texts",
    "get_embedding_metadata",
]
