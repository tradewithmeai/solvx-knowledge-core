"""Production-ready embeddings provider system with multiple backends.

This module provides a unified interface for generating embeddings using different
providers (OpenAI, Cohere, Voyage, Local). Features include:
- Retry logic with exponential backoff
- Rate limiting
- Batch processing
- Comprehensive error handling
- Detailed logging
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

    def __init__(self, message: str, provider: str, cause: Optional[Exception] = None):
        """Initialize embedding error.

        Args:
            message: Error message
            provider: Provider name that raised the error
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.provider = provider
        self.cause = cause


class EmbeddingAPIError(EmbeddingError):
    """Exception for API-related errors."""

    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Exception for rate limit errors."""

    pass


class EmbeddingAuthenticationError(EmbeddingError):
    """Exception for authentication errors."""

    pass


class EmbeddingModelError(EmbeddingError):
    """Exception for model loading or inference errors."""

    pass


# ============================================================================
# Base Provider Interface
# ============================================================================


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(
        self,
        model: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        batch_size: int = 64,
    ):
        """Initialize embedding provider.

        Args:
            model: Model name/identifier (provider-specific)
            rate_limit_delay: Delay between API requests in seconds
            batch_size: Maximum batch size for processing
        """
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.batch_size = batch_size
        self.last_request_time = 0.0

    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (provider-specific implementation).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (one per text)

        Raises:
            EmbeddingError: On embedding failure
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions for this provider/model.

        Returns:
            Number of dimensions in embeddings
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name string
        """
        pass

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting by sleeping if needed."""
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    def embed_texts(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Embed multiple texts with batching and rate limiting.

        Args:
            texts: List of texts to embed
            model: Optional model override (provider-specific)

        Returns:
            List of embeddings (one per text)

        Raises:
            EmbeddingError: On embedding failure
        """
        if not texts:
            logger.warning("Empty text list provided to embed_texts")
            return []

        # Override model if provided
        original_model = self.model
        if model:
            self.model = model

        try:
            logger.info(
                f"Embedding {len(texts)} texts with {self.get_provider_name()} "
                f"(model={self.model}, batch_size={self.batch_size})"
            )

            all_embeddings: List[List[float]] = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

                logger.debug(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} texts)"
                )

                # Apply rate limiting
                self._apply_rate_limit()

                # Embed batch with retries
                batch_embeddings = self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Batch {batch_num}/{total_batches} completed")

            logger.info(
                f"Successfully embedded {len(texts)} texts "
                f"(dimensions={self.get_dimensions()})"
            )

            return all_embeddings

        finally:
            # Restore original model
            if model:
                self.model = original_model

    def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata.

        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.model,
            "dimensions": self.get_dimensions(),
            "batch_size": self.batch_size,
            "rate_limit_delay": self.rate_limit_delay,
        }


# ============================================================================
# OpenAI Provider
# ============================================================================


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings provider using text-embedding-3-small by default."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        rate_limit_delay: float = 0.05,
        batch_size: int = 100,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to text-embedding-3-small)
            rate_limit_delay: Delay between requests
            batch_size: Batch size for processing
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            rate_limit_delay=rate_limit_delay,
            batch_size=batch_size,
        )
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise EmbeddingAuthenticationError(
                "OpenAI API key not provided",
                provider="openai",
            )

        # Import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise EmbeddingError(
                "OpenAI package not installed. Run: pip install openai",
                provider="openai",
                cause=e,
            )

        logger.info(f"Initialized OpenAI provider with model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EmbeddingAPIError),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI API with retries.

        Args:
            texts: Texts to embed

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: On API failure
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            error_msg = str(e).lower()

            # Classify error type
            if "rate_limit" in error_msg or "429" in error_msg:
                logger.warning(f"OpenAI rate limit hit: {e}")
                raise EmbeddingRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="openai",
                    cause=e,
                )
            elif "auth" in error_msg or "401" in error_msg:
                logger.error(f"OpenAI authentication failed: {e}")
                raise EmbeddingAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="openai",
                    cause=e,
                )
            else:
                logger.error(f"OpenAI API error: {e}")
                raise EmbeddingAPIError(
                    f"API request failed: {e}",
                    provider="openai",
                    cause=e,
                )

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        # text-embedding-3-small and text-embedding-3-large use 1536
        # text-embedding-ada-002 uses 1536
        return self.DIMENSIONS

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"


# ============================================================================
# Cohere Provider
# ============================================================================


class CohereProvider(EmbeddingProvider):
    """Cohere embeddings provider using embed-english-v3.0 by default."""

    DEFAULT_MODEL = "embed-english-v3.0"
    DIMENSIONS = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        batch_size: int = 96,
        input_type: Literal[
            "search_document", "search_query", "classification", "clustering"
        ] = "search_document",
    ):
        """Initialize Cohere provider.

        Args:
            api_key: Cohere API key (defaults to settings)
            model: Model name (defaults to embed-english-v3.0)
            rate_limit_delay: Delay between requests
            batch_size: Batch size (Cohere allows up to 96)
            input_type: Type of input for embeddings
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            rate_limit_delay=rate_limit_delay,
            batch_size=batch_size,
        )
        self.api_key = api_key or settings.cohere_api_key
        self.input_type = input_type

        if not self.api_key:
            raise EmbeddingAuthenticationError(
                "Cohere API key not provided",
                provider="cohere",
            )

        # Import Cohere client
        try:
            import cohere
            self.client = cohere.Client(api_key=self.api_key)
        except ImportError as e:
            raise EmbeddingError(
                "Cohere package not installed. Run: pip install cohere",
                provider="cohere",
                cause=e,
            )

        logger.info(
            f"Initialized Cohere provider with model={self.model}, "
            f"input_type={self.input_type}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EmbeddingAPIError),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Cohere API with retries.

        Args:
            texts: Texts to embed

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: On API failure
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type,
            )

            # Convert to list of lists
            embeddings = [list(emb) for emb in response.embeddings]
            return embeddings

        except Exception as e:
            error_msg = str(e).lower()

            # Classify error type
            if "rate" in error_msg or "429" in error_msg:
                logger.warning(f"Cohere rate limit hit: {e}")
                raise EmbeddingRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="cohere",
                    cause=e,
                )
            elif "auth" in error_msg or "401" in error_msg:
                logger.error(f"Cohere authentication failed: {e}")
                raise EmbeddingAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="cohere",
                    cause=e,
                )
            else:
                logger.error(f"Cohere API error: {e}")
                raise EmbeddingAPIError(
                    f"API request failed: {e}",
                    provider="cohere",
                    cause=e,
                )

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.DIMENSIONS

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "cohere"


# ============================================================================
# Voyage Provider
# ============================================================================


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embeddings provider using voyage-2 by default."""

    DEFAULT_MODEL = "voyage-2"
    DIMENSIONS = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        batch_size: int = 128,
    ):
        """Initialize Voyage provider.

        Args:
            api_key: Voyage API key (defaults to settings)
            model: Model name (defaults to voyage-2)
            rate_limit_delay: Delay between requests
            batch_size: Batch size (Voyage allows up to 128)
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            rate_limit_delay=rate_limit_delay,
            batch_size=batch_size,
        )
        self.api_key = api_key or settings.voyage_api_key

        if not self.api_key:
            raise EmbeddingAuthenticationError(
                "Voyage API key not provided",
                provider="voyage",
            )

        # Import Voyage client
        try:
            import voyageai
            self.client = voyageai.Client(api_key=self.api_key)
        except ImportError as e:
            raise EmbeddingError(
                "VoyageAI package not installed. Run: pip install voyageai",
                provider="voyage",
                cause=e,
            )

        logger.info(f"Initialized Voyage provider with model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EmbeddingAPIError),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Voyage API with retries.

        Args:
            texts: Texts to embed

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: On API failure
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
            )

            embeddings = response.embeddings
            return embeddings

        except Exception as e:
            error_msg = str(e).lower()

            # Classify error type
            if "rate" in error_msg or "429" in error_msg:
                logger.warning(f"Voyage rate limit hit: {e}")
                raise EmbeddingRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="voyage",
                    cause=e,
                )
            elif "auth" in error_msg or "401" in error_msg:
                logger.error(f"Voyage authentication failed: {e}")
                raise EmbeddingAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="voyage",
                    cause=e,
                )
            else:
                logger.error(f"Voyage API error: {e}")
                raise EmbeddingAPIError(
                    f"API request failed: {e}",
                    provider="voyage",
                    cause=e,
                )

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.DIMENSIONS

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "voyage"


# ============================================================================
# Local Provider (Sentence Transformers)
# ============================================================================


class LocalProvider(EmbeddingProvider):
    """Local embeddings using sentence-transformers (all-mpnet-base-v2)."""

    DEFAULT_MODEL = "all-mpnet-base-v2"
    DIMENSIONS = 768

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """Initialize local provider.

        Args:
            model: Model name (defaults to all-mpnet-base-v2)
            batch_size: Batch size for processing
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            rate_limit_delay=0.0,  # No rate limiting for local
            batch_size=batch_size,
        )
        self.device = device

        # Import and load model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local model: {self.model}")
            self.encoder = SentenceTransformer(self.model, device=self.device)
            logger.info(
                f"Loaded local model on device: {self.encoder.device} "
                f"(dimensions={self.get_dimensions()})"
            )

        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers",
                provider="local",
                cause=e,
            )
        except Exception as e:
            raise EmbeddingModelError(
                f"Failed to load model '{self.model}': {e}",
                provider="local",
                cause=e,
            )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using local model.

        Args:
            texts: Texts to embed

        Returns:
            List of embeddings

        Raises:
            EmbeddingError: On inference failure
        """
        try:
            # Encode with sentence-transformers
            embeddings = self.encoder.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise EmbeddingModelError(
                f"Model inference failed: {e}",
                provider="local",
                cause=e,
            )

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.DIMENSIONS

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "local"


# ============================================================================
# Provider Factory
# ============================================================================


def get_embedding_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to get an embedding provider.

    Args:
        provider_name: Provider name ('openai', 'cohere', 'voyage', 'local')
                      Defaults to settings.embed_provider
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider name is invalid
        EmbeddingError: If provider initialization fails
    """
    provider_name = provider_name or settings.embed_provider

    logger.info(f"Initializing embedding provider: {provider_name}")

    try:
        if provider_name == "openai":
            return OpenAIProvider(model=model, **kwargs)
        elif provider_name == "cohere":
            return CohereProvider(model=model, **kwargs)
        elif provider_name == "voyage":
            return VoyageProvider(model=model, **kwargs)
        elif provider_name == "local":
            return LocalProvider(model=model, **kwargs)
        else:
            raise ValueError(
                f"Invalid provider: {provider_name}. "
                f"Must be one of: openai, cohere, voyage, local"
            )

    except EmbeddingError:
        # Re-raise embedding errors
        raise
    except Exception as e:
        # Wrap other errors
        raise EmbeddingError(
            f"Failed to initialize provider '{provider_name}': {e}",
            provider=provider_name,
            cause=e,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def embed_texts(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Convenience function to embed texts with the configured provider.

    Args:
        texts: List of texts to embed
        provider: Provider name (defaults to settings)
        model: Model name (defaults to provider default)
        **kwargs: Additional provider-specific arguments

    Returns:
        List of embeddings

    Raises:
        EmbeddingError: On embedding failure
    """
    provider_instance = get_embedding_provider(
        provider_name=provider,
        model=model,
        **kwargs,
    )

    return provider_instance.embed_texts(texts)


def get_embedding_metadata(provider: Optional[str] = None) -> Dict[str, Any]:
    """Get metadata for the configured embedding provider.

    Args:
        provider: Provider name (defaults to settings)

    Returns:
        Provider metadata dictionary

    Raises:
        EmbeddingError: If provider initialization fails
    """
    provider_instance = get_embedding_provider(provider_name=provider)
    return provider_instance.get_metadata()
