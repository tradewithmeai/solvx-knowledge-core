"""
Embeddings coordinator for batch processing with caching, rate limiting, and parallel execution.

This module coordinates embedding generation across multiple chunks, managing:
- Cache hits/misses via EmbeddingsCache
- Provider selection and API calls
- Batch processing with configurable batch sizes
- Daily token usage tracking and enforcement
- Parallel execution with ThreadPoolExecutor
- Progress tracking and comprehensive logging
- Graceful failure handling
"""

import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from tqdm import tqdm

from backend.app.config import settings
from backend.app.embeddings.cache import get_cache_instance
from backend.app.logging import get_logger
from backend.app.utils.errors import EmbeddingError, RateLimitError

logger = get_logger(__name__)


class TokenUsageTracker:
    """
    Thread-safe daily token usage tracker using SQLite.

    Tracks embedding token usage per day to enforce MAX_EMBED_TOKENS_PER_DAY limits.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize token usage tracker.

        Args:
            db_path: Path to SQLite database for tracking. Defaults to data dir.
        """
        if db_path is None:
            db_path = settings.solvx_data_dir / "token_usage.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        logger.info(f"Initializing TokenUsageTracker at {self.db_path}")
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create database schema if it doesn't exist."""
        try:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                cursor = conn.cursor()

                # Create token usage table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embed_token_usage (
                        date TEXT PRIMARY KEY,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        last_updated DATETIME NOT NULL
                    )
                """)

                # Create index on date
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embed_usage_date
                    ON embed_token_usage (date)
                """)

                conn.commit()
                logger.debug("Token usage database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize token usage database: {e}", exc_info=True)
            raise

    def get_today_usage(self) -> int:
        """
        Get total tokens used today.

        Returns:
            Number of tokens used today
        """
        today = datetime.utcnow().date().isoformat()

        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT token_count FROM embed_token_usage WHERE date = ?",
                        (today,)
                    )
                    result = cursor.fetchone()
                    return result[0] if result else 0
            except Exception as e:
                logger.error(f"Error getting today's token usage: {e}", exc_info=True)
                return 0

    def add_tokens(self, count: int) -> int:
        """
        Add tokens to today's usage count.

        Args:
            count: Number of tokens to add

        Returns:
            New total token count for today

        Raises:
            RateLimitError: If adding tokens would exceed daily limit
        """
        if count < 0:
            raise ValueError("Token count cannot be negative")

        today = datetime.utcnow().date().isoformat()
        now = datetime.utcnow().isoformat()

        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.cursor()

                    # Get current usage
                    cursor.execute(
                        "SELECT token_count FROM embed_token_usage WHERE date = ?",
                        (today,)
                    )
                    result = cursor.fetchone()
                    current_usage = result[0] if result else 0

                    # Check if adding would exceed limit
                    new_total = current_usage + count
                    if new_total > settings.max_embed_tokens_per_day:
                        logger.warning(
                            f"Daily embedding token limit exceeded: "
                            f"{new_total} > {settings.max_embed_tokens_per_day}"
                        )
                        raise RateLimitError(
                            f"Daily embedding token limit of {settings.max_embed_tokens_per_day:,} "
                            f"would be exceeded. Current usage: {current_usage:,}, "
                            f"requested: {count:,}. Please try again tomorrow or increase "
                            f"MAX_EMBED_TOKENS_PER_DAY in your configuration.",
                            details={
                                "current_usage": current_usage,
                                "requested": count,
                                "limit": settings.max_embed_tokens_per_day,
                                "date": today,
                            }
                        )

                    # Update or insert usage
                    cursor.execute("""
                        INSERT INTO embed_token_usage (date, token_count, last_updated)
                        VALUES (?, ?, ?)
                        ON CONFLICT(date) DO UPDATE SET
                            token_count = token_count + ?,
                            last_updated = ?
                    """, (today, count, now, count, now))

                    conn.commit()

                    logger.debug(
                        f"Added {count:,} tokens to daily usage. "
                        f"New total: {new_total:,}/{settings.max_embed_tokens_per_day:,}"
                    )

                    return new_total
            except RateLimitError:
                raise
            except Exception as e:
                logger.error(f"Error adding tokens to usage tracker: {e}", exc_info=True)
                raise

    def get_remaining_tokens(self) -> int:
        """
        Get remaining tokens available today.

        Returns:
            Number of tokens remaining in daily quota
        """
        today_usage = self.get_today_usage()
        remaining = max(0, settings.max_embed_tokens_per_day - today_usage)
        return remaining

    def cleanup_old_records(self, days: int = 90) -> int:
        """
        Delete usage records older than specified days.

        Args:
            days: Keep records from last N days

        Returns:
            Number of deleted records
        """
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM embed_token_usage WHERE date < ?",
                        (cutoff_date,)
                    )
                    deleted = cursor.rowcount
                    conn.commit()

                    logger.info(f"Cleaned up {deleted} token usage records older than {days} days")
                    return deleted
            except Exception as e:
                logger.error(f"Error cleaning up old token usage records: {e}", exc_info=True)
                return 0


# Global token tracker instance
_token_tracker: Optional[TokenUsageTracker] = None
_tracker_lock = threading.RLock()


def get_token_tracker() -> TokenUsageTracker:
    """Get or create singleton TokenUsageTracker instance."""
    global _token_tracker

    if _token_tracker is None:
        with _tracker_lock:
            if _token_tracker is None:
                _token_tracker = TokenUsageTracker()

    return _token_tracker


class EmbeddingProvider:
    """
    Base interface for embedding providers.

    Subclasses should implement embed_batch() for their specific API.
    """

    def __init__(self, model: str):
        """
        Initialize provider.

        Args:
            model: Model identifier for this provider
        """
        self.model = model
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of numpy arrays (one per text)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        raise NotImplementedError("Subclasses must implement embed_batch()")

    def count_tokens(self, texts: List[str]) -> int:
        """
        Estimate token count for a list of texts.

        Default implementation uses a rough approximation.
        Providers should override with more accurate counting.

        Args:
            texts: List of text strings

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        total_chars = sum(len(text) for text in texts)
        return max(1, total_chars // 4)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    This provider runs models locally and doesn't consume API tokens.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding provider.

        Args:
            model: Model name from sentence-transformers hub
        """
        super().__init__(model)
        self.provider_name = "local"

        # Lazy load sentence transformers to avoid import overhead
        self._model_instance = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {self.model}")
                self._model_instance = SentenceTransformer(self.model)
                logger.info(f"Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {e}", exc_info=True)
                raise EmbeddingError(
                    f"Failed to load local embedding model '{self.model}': {e}",
                    details={"model": self.model, "error": str(e)}
                )
        return self._model_instance

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local sentence transformer model."""
        if not texts:
            return []

        try:
            model = self._get_model()

            # Generate embeddings
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Ensure float32 and convert to list of arrays
            embeddings = embeddings.astype(np.float32)
            return [embeddings[i] for i in range(len(embeddings))]
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}", exc_info=True)
            raise EmbeddingError(
                f"Local embedding generation failed: {e}",
                details={"model": self.model, "batch_size": len(texts), "error": str(e)}
            )

    def count_tokens(self, texts: List[str]) -> int:
        """
        Local models don't consume API tokens.

        Returns 0 to avoid counting against daily limits.
        """
        return 0


def get_provider() -> EmbeddingProvider:
    """
    Get the configured embedding provider.

    Returns:
        EmbeddingProvider instance based on settings.embed_provider

    Raises:
        ValueError: If provider is not supported or not configured
    """
    provider_name = settings.embed_provider.lower()

    if provider_name == "local":
        return LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    elif provider_name == "openai":
        # Placeholder for OpenAI provider implementation
        # TODO: Implement OpenAIEmbeddingProvider
        raise NotImplementedError(
            "OpenAI provider not yet implemented. "
            "Use EMBED_PROVIDER=local for now or implement OpenAIEmbeddingProvider."
        )
    elif provider_name == "cohere":
        # Placeholder for Cohere provider implementation
        # TODO: Implement CohereEmbeddingProvider
        raise NotImplementedError(
            "Cohere provider not yet implemented. "
            "Use EMBED_PROVIDER=local for now or implement CohereEmbeddingProvider."
        )
    elif provider_name == "voyage":
        # Placeholder for Voyage provider implementation
        # TODO: Implement VoyageEmbeddingProvider
        raise NotImplementedError(
            "Voyage provider not yet implemented. "
            "Use EMBED_PROVIDER=local for now or implement VoyageEmbeddingProvider."
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider_name}. "
            f"Supported providers: local, openai, cohere, voyage"
        )


def _embed_single_chunk(
    chunk: dict,
    provider: EmbeddingProvider,
    cache: Any,
    tracker: TokenUsageTracker,
    check_rate_limit: bool = True,
) -> dict:
    """
    Embed a single chunk with caching and error handling.

    Args:
        chunk: Chunk dictionary with 'text' field
        provider: EmbeddingProvider instance
        cache: EmbeddingsCache instance
        tracker: TokenUsageTracker instance
        check_rate_limit: Whether to check/enforce rate limits

    Returns:
        Updated chunk dictionary with 'embedding' field added

    Raises:
        RateLimitError: If rate limit would be exceeded
        EmbeddingError: If embedding generation fails
    """
    text = chunk.get("text", "").strip()

    if not text:
        logger.warning(f"Skipping chunk with empty text: {chunk.get('id', 'unknown')}")
        chunk["embedding"] = None
        chunk["embedding_error"] = "Empty text"
        return chunk

    try:
        # Check cache first
        cached_embedding = cache.get_cached_embedding(
            text=text,
            provider=provider.provider_name,
            model=provider.model,
        )

        if cached_embedding is not None:
            logger.debug(f"Cache hit for chunk {chunk.get('id', 'unknown')}")
            chunk["embedding"] = cached_embedding.tolist()
            chunk["from_cache"] = True
            return chunk

        # Cache miss - need to generate embedding
        logger.debug(f"Cache miss for chunk {chunk.get('id', 'unknown')}")

        # Check token limit before generating (if not local provider)
        if check_rate_limit:
            token_count = provider.count_tokens([text])
            if token_count > 0:
                # This will raise RateLimitError if limit exceeded
                tracker.add_tokens(token_count)

        # Generate embedding
        embeddings = provider.embed_batch([text])

        if not embeddings or embeddings[0] is None:
            raise EmbeddingError("Provider returned None/empty embedding")

        embedding = embeddings[0]

        # Cache the result
        cache.cache_embedding(
            text=text,
            provider=provider.provider_name,
            model=provider.model,
            vector=embedding,
        )

        chunk["embedding"] = embedding.tolist()
        chunk["from_cache"] = False
        return chunk

    except RateLimitError:
        # Re-raise rate limit errors
        raise
    except Exception as e:
        logger.error(
            f"Failed to embed chunk {chunk.get('id', 'unknown')}: {e}",
            exc_info=True
        )
        chunk["embedding"] = None
        chunk["embedding_error"] = str(e)
        return chunk


def embed_chunks(
    chunks: List[dict],
    show_progress: bool = True,
    fail_on_error: bool = False,
) -> List[dict]:
    """
    Generate embeddings for a list of chunks with caching, batching, and parallel processing.

    This is the main entry point for embedding generation. It coordinates:
    - Cache lookups to avoid redundant API calls
    - Rate limiting based on daily token quotas
    - Parallel processing with ThreadPoolExecutor
    - Progress tracking and comprehensive logging
    - Graceful error handling

    Args:
        chunks: List of chunk dictionaries. Each must have a 'text' field.
               The 'embedding' field will be added to each chunk.
        show_progress: Whether to show progress bar (default: True)
        fail_on_error: If True, raise exception on first error. If False,
                      skip failed chunks and continue (default: False)

    Returns:
        List of chunk dictionaries with 'embedding' field added.
        Successful embeddings will have 'embedding' as a list of floats.
        Failed embeddings will have 'embedding' = None and 'embedding_error' field.

    Raises:
        RateLimitError: If daily token limit would be exceeded (HTTP 429)
        EmbeddingError: If fail_on_error=True and any embedding fails
        ValueError: If chunks is empty or invalid

    Example:
        >>> chunks = [
        ...     {"id": 1, "text": "Hello world", "seq": 0},
        ...     {"id": 2, "text": "Another chunk", "seq": 1},
        ... ]
        >>> embedded_chunks = embed_chunks(chunks)
        >>> assert "embedding" in embedded_chunks[0]
        >>> assert isinstance(embedded_chunks[0]["embedding"], list)
    """
    if not chunks:
        logger.warning("embed_chunks called with empty chunk list")
        return []

    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list of dictionaries")

    logger.info(f"Starting embedding for {len(chunks)} chunks")

    # Initialize components
    provider = get_provider()
    cache = get_cache_instance(
        db_path=str(settings.solvx_data_dir / "embeddings_cache.db")
    )
    tracker = get_token_tracker()

    # Log initial state
    remaining_tokens = tracker.get_remaining_tokens()
    logger.info(
        f"Embedding with provider={provider.provider_name}, "
        f"model={provider.model}, "
        f"batch_size={settings.embed_batch_size}, "
        f"max_workers={settings.max_embed_workers}, "
        f"remaining_tokens={remaining_tokens:,}/{settings.max_embed_tokens_per_day:,}"
    )

    # Process chunks in parallel
    embedded_chunks = []
    failed_chunks = []
    cache_hits = 0
    cache_misses = 0

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(settings.max_embed_workers, len(chunks))

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(
                    _embed_single_chunk,
                    chunk,
                    provider,
                    cache,
                    tracker,
                    check_rate_limit=True,
                ): chunk
                for chunk in chunks
            }

            # Process completed futures with progress bar
            if show_progress:
                futures_iter = tqdm(
                    as_completed(future_to_chunk),
                    total=len(chunks),
                    desc="Embedding chunks",
                    unit="chunk",
                )
            else:
                futures_iter = as_completed(future_to_chunk)

            for future in futures_iter:
                original_chunk = future_to_chunk[future]

                try:
                    result_chunk = future.result()
                    embedded_chunks.append(result_chunk)

                    # Track cache statistics
                    if result_chunk.get("from_cache"):
                        cache_hits += 1
                    elif result_chunk.get("embedding") is not None:
                        cache_misses += 1

                    # Track failures
                    if result_chunk.get("embedding") is None:
                        failed_chunks.append(result_chunk)
                        if fail_on_error:
                            error_msg = result_chunk.get("embedding_error", "Unknown error")
                            raise EmbeddingError(
                                f"Embedding failed for chunk {result_chunk.get('id')}: {error_msg}",
                                details={"chunk": result_chunk}
                            )

                except RateLimitError as e:
                    logger.error(f"Rate limit exceeded: {e}")
                    # Rate limit errors should stop all processing
                    raise

                except Exception as e:
                    logger.error(f"Error processing chunk: {e}", exc_info=True)
                    if fail_on_error:
                        raise
                    # Add error chunk to results
                    error_chunk = original_chunk.copy()
                    error_chunk["embedding"] = None
                    error_chunk["embedding_error"] = str(e)
                    embedded_chunks.append(error_chunk)
                    failed_chunks.append(error_chunk)

    except RateLimitError:
        # Re-raise rate limit errors
        raise
    except Exception as e:
        logger.error(f"Fatal error in embed_chunks: {e}", exc_info=True)
        if fail_on_error:
            raise

    # Log final statistics
    success_count = len(embedded_chunks) - len(failed_chunks)
    total_count = len(chunks)
    cache_hit_rate = (cache_hits / total_count * 100) if total_count > 0 else 0

    logger.info(
        f"Embedding complete: {success_count}/{total_count} successful, "
        f"{len(failed_chunks)} failed, "
        f"cache_hit_rate={cache_hit_rate:.1f}% ({cache_hits}/{total_count})"
    )

    if failed_chunks:
        logger.warning(
            f"Failed to embed {len(failed_chunks)} chunks. "
            f"Check logs for details. Failed chunk IDs: "
            f"{[c.get('id', 'unknown') for c in failed_chunks[:10]]}"
        )

    # Update final token usage
    final_remaining = tracker.get_remaining_tokens()
    logger.info(f"Remaining tokens after embedding: {final_remaining:,}/{settings.max_embed_tokens_per_day:,}")

    return embedded_chunks
