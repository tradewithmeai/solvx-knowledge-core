"""
Embeddings cache system using SQLite with thread-safe operations.

This module provides a persistent cache for embeddings vectors using SQLite,
with support for numpy array serialization, compression, and statistics tracking.

Features:
- BLAKE3 hashing of normalized text + provider + model
- Compressed numpy array storage as BLOB
- Thread-safe operations
- Comprehensive logging
- Optional automatic cleanup of old entries
- Resilient error handling
"""

import sqlite3
import logging
import zlib
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager

import numpy as np

try:
    from blake3 import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """
    Thread-safe SQLite-based cache for embedding vectors.

    Stores embeddings with their metadata, using BLAKE3 hashing for key
    generation and zlib compression for vector storage.
    """

    # Schema version for future migrations
    SCHEMA_VERSION = 1

    # Compression level for zlib (0-9, higher = more compression but slower)
    COMPRESSION_LEVEL = 6

    def __init__(
        self,
        db_path: str = "/tmp/embeddings_cache.db",
        cleanup_days: Optional[int] = None,
        auto_cleanup: bool = False,
    ):
        """
        Initialize the embeddings cache.

        Args:
            db_path: Path to SQLite database file
            cleanup_days: Delete cache entries older than this many days (None = no cleanup)
            auto_cleanup: If True, run cleanup on initialization

        Raises:
            ValueError: If cleanup_days is invalid
        """
        if cleanup_days is not None and cleanup_days < 1:
            raise ValueError("cleanup_days must be >= 1 or None")

        self.db_path = Path(db_path)
        self.cleanup_days = cleanup_days

        # Thread-safe lock for database operations
        self._lock = threading.RLock()

        logger.info(f"Initializing EmbeddingsCache at {self.db_path}")

        try:
            self._db_path_parent = self.db_path.parent
            self._db_path_parent.mkdir(parents=True, exist_ok=True)
            self._initialize_database()

            if auto_cleanup and cleanup_days:
                self.cleanup_old_entries()

            logger.info("EmbeddingsCache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingsCache: {e}", exc_info=True)
            raise

    def _initialize_database(self) -> None:
        """Create database schema if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create main cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS emb_cache (
                        hash TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        dims INTEGER NOT NULL,
                        created_ts DATETIME NOT NULL,
                        accessed_ts DATETIME NOT NULL
                    )
                """)

                # Create index for frequent queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_provider_model
                    ON emb_cache (provider, model)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_ts
                    ON emb_cache (created_ts)
                """)

                # Create metadata table for cache statistics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)

                # Initialize schema version
                cursor.execute(
                    "INSERT OR IGNORE INTO cache_metadata (key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION))
                )

                conn.commit()
                logger.debug("Database schema initialized")
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
            raise

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _compute_hash(self, text: str, provider: str, model: str) -> str:
        """
        Compute BLAKE3 hash of normalized text + provider + model.

        Falls back to SHA256 if blake3 is not available.

        Args:
            text: The text to hash
            provider: Embedding provider name
            model: Model identifier

        Returns:
            Hex-encoded hash string
        """
        try:
            # Normalize text: strip, lowercase, unicode normalize
            normalized = text.strip().lower()

            # Create composite key
            composite = f"{normalized}|{provider}|{model}".encode("utf-8")

            if HAS_BLAKE3:
                h = blake3(composite).hexdigest()
            else:
                h = hashlib.sha256(composite).hexdigest()
                if not HAS_BLAKE3:
                    logger.debug(
                        "blake3 not available, using SHA256 for hashing. "
                        "Install blake3 package for better performance."
                    )

            return h
        except Exception as e:
            logger.error(f"Hash computation error: {e}", exc_info=True)
            raise

    def _compress_vector(self, vector: np.ndarray) -> bytes:
        """
        Compress numpy array using zlib.

        Args:
            vector: Numpy array to compress

        Returns:
            Compressed bytes
        """
        try:
            # Ensure contiguous array for serialization
            if not vector.flags["C_CONTIGUOUS"]:
                vector = np.ascontiguousarray(vector)

            # Serialize array to bytes
            serialized = vector.tobytes()

            # Compress
            compressed = zlib.compress(serialized, level=self.COMPRESSION_LEVEL)
            logger.debug(
                f"Compressed vector: {len(serialized)} -> {len(compressed)} bytes"
            )

            return compressed
        except Exception as e:
            logger.error(f"Vector compression error: {e}", exc_info=True)
            raise

    def _decompress_vector(self, compressed: bytes, dims: int) -> Optional[np.ndarray]:
        """
        Decompress and reconstruct numpy array.

        Args:
            compressed: Compressed bytes from database
            dims: Number of dimensions in original vector

        Returns:
            Reconstructed numpy array or None if decompression fails
        """
        try:
            # Decompress
            serialized = zlib.decompress(compressed)

            # Reconstruct array
            vector = np.frombuffer(serialized, dtype=np.float32)

            # Validate dimensions
            if len(vector) != dims:
                logger.warning(
                    f"Dimension mismatch: expected {dims}, got {len(vector)}"
                )
                return None

            return vector
        except Exception as e:
            logger.error(f"Vector decompression error: {e}", exc_info=True)
            return None

    def get_cached_embedding(
        self,
        text: str,
        provider: str,
        model: str,
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding vector.

        Updates the accessed_ts timestamp on retrieval.

        Args:
            text: The text that was embedded
            provider: Embedding provider name
            model: Model identifier

        Returns:
            Numpy array if found, None if not found or on error
        """
        if not text or not provider or not model:
            logger.warning("Invalid arguments to get_cached_embedding")
            return None

        with self._lock:
            try:
                hash_key = self._compute_hash(text, provider, model)

                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        SELECT vector, dims FROM emb_cache
                        WHERE hash = ? AND provider = ? AND model = ?
                        """,
                        (hash_key, provider, model),
                    )

                    result = cursor.fetchone()

                    if result is None:
                        logger.debug(
                            f"Cache miss for {provider}/{model} "
                            f"(hash={hash_key[:8]}...)"
                        )
                        return None

                    compressed_vector, dims = result

                    # Update accessed timestamp
                    cursor.execute(
                        """
                        UPDATE emb_cache
                        SET accessed_ts = ?
                        WHERE hash = ?
                        """,
                        (datetime.utcnow().isoformat(), hash_key),
                    )
                    conn.commit()

                    # Decompress and return
                    vector = self._decompress_vector(compressed_vector, dims)

                    if vector is not None:
                        logger.debug(
                            f"Cache hit for {provider}/{model} "
                            f"(dims={dims}, hash={hash_key[:8]}...)"
                        )

                    return vector

            except Exception as e:
                logger.error(
                    f"Error retrieving cached embedding: {e}",
                    exc_info=True
                )
                return None

    def cache_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        vector: np.ndarray,
    ) -> bool:
        """
        Store an embedding vector in the cache.

        Args:
            text: The text that was embedded
            provider: Embedding provider name
            model: Model identifier
            vector: Numpy array of the embedding

        Returns:
            True if cached successfully, False on error
        """
        if not text or not provider or not model:
            logger.warning("Invalid arguments to cache_embedding")
            return False

        if not isinstance(vector, np.ndarray):
            logger.warning(
                f"Expected numpy.ndarray, got {type(vector).__name__}"
            )
            return False

        if vector.dtype != np.float32:
            logger.debug(f"Converting vector from {vector.dtype} to float32")
            try:
                vector = vector.astype(np.float32)
            except Exception as e:
                logger.error(f"Vector dtype conversion error: {e}", exc_info=True)
                return False

        with self._lock:
            try:
                hash_key = self._compute_hash(text, provider, model)
                compressed = self._compress_vector(vector)
                dims = len(vector)
                now = datetime.utcnow().isoformat()

                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO emb_cache
                        (hash, provider, model, vector, dims, created_ts, accessed_ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (hash_key, provider, model, compressed, dims, now, now),
                    )
                    conn.commit()

                    logger.debug(
                        f"Cached embedding for {provider}/{model} "
                        f"(dims={dims}, hash={hash_key[:8]}...)"
                    )

                    return True

            except Exception as e:
                logger.error(
                    f"Error caching embedding: {e}",
                    exc_info=True
                )
                return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Total entries
                    cursor.execute("SELECT COUNT(*) FROM emb_cache")
                    total_entries = cursor.fetchone()[0]

                    # Unique providers and models
                    cursor.execute(
                        "SELECT COUNT(DISTINCT provider) FROM emb_cache"
                    )
                    unique_providers = cursor.fetchone()[0]

                    cursor.execute(
                        "SELECT COUNT(DISTINCT model) FROM emb_cache"
                    )
                    unique_models = cursor.fetchone()[0]

                    # Total storage size
                    cursor.execute("SELECT SUM(LENGTH(vector)) FROM emb_cache")
                    total_size_result = cursor.fetchone()[0]
                    total_size = total_size_result or 0

                    # Provider/model breakdown
                    cursor.execute(
                        """
                        SELECT provider, model, COUNT(*) as count,
                               SUM(LENGTH(vector)) as size
                        FROM emb_cache
                        GROUP BY provider, model
                        ORDER BY count DESC
                        """
                    )
                    provider_stats = [
                        {
                            "provider": row[0],
                            "model": row[1],
                            "count": row[2],
                            "size_bytes": row[3] or 0,
                        }
                        for row in cursor.fetchall()
                    ]

                    # Oldest and newest entries
                    cursor.execute(
                        """
                        SELECT
                            MIN(created_ts) as oldest,
                            MAX(created_ts) as newest,
                            MAX(accessed_ts) as last_accessed
                        FROM emb_cache
                        """
                    )
                    timestamps = cursor.fetchone()

                    # Database file size
                    db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                    stats = {
                        "total_entries": total_entries,
                        "unique_providers": unique_providers,
                        "unique_models": unique_models,
                        "total_vector_size_bytes": total_size,
                        "database_file_size_bytes": db_size,
                        "oldest_entry": timestamps[0],
                        "newest_entry": timestamps[1],
                        "last_accessed": timestamps[2],
                        "provider_breakdown": provider_stats,
                        "cleanup_days": self.cleanup_days,
                    }

                    logger.info(
                        f"Cache stats: {total_entries} entries, "
                        f"{unique_providers} providers, "
                        f"{unique_models} models, "
                        f"{total_size / (1024*1024):.2f} MB"
                    )

                    return stats

            except Exception as e:
                logger.error(f"Error getting cache stats: {e}", exc_info=True)
                return {
                    "total_entries": 0,
                    "unique_providers": 0,
                    "unique_models": 0,
                    "total_vector_size_bytes": 0,
                    "database_file_size_bytes": 0,
                    "error": str(e),
                }

    def cleanup_old_entries(self, days: Optional[int] = None) -> int:
        """
        Delete cache entries older than the specified number of days.

        Args:
            days: Delete entries older than this many days.
                  If None, uses self.cleanup_days.

        Returns:
            Number of deleted entries
        """
        if days is None:
            days = self.cleanup_days

        if days is None:
            logger.warning(
                "No cleanup_days configured. Use days parameter or "
                "set cleanup_days on initialization."
            )
            return 0

        if days < 1:
            logger.warning("cleanup_days must be >= 1")
            return 0

        with self._lock:
            try:
                cutoff_date = (
                    datetime.utcnow() - timedelta(days=days)
                ).isoformat()

                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        DELETE FROM emb_cache
                        WHERE created_ts < ?
                        """,
                        (cutoff_date,),
                    )

                    deleted_count = cursor.rowcount
                    conn.commit()

                    logger.info(
                        f"Cleanup removed {deleted_count} entries older than "
                        f"{days} days (before {cutoff_date})"
                    )

                    return deleted_count

            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)
                return 0

    def clear_all(self) -> bool:
        """
        Clear all entries from the cache.

        Returns:
            True if successful, False on error
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM emb_cache")
                    count = cursor.rowcount
                    conn.commit()

                    logger.info(f"Cleared all {count} cache entries")
                    return True

            except Exception as e:
                logger.error(f"Error clearing cache: {e}", exc_info=True)
                return False

    def vacuum(self) -> bool:
        """
        Optimize database by running VACUUM.

        Returns:
            True if successful, False on error
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.commit()

                    db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                    logger.info(f"Database vacuumed. File size: {db_size / 1024:.2f} KB")

                    return True

            except Exception as e:
                logger.error(f"Error vacuuming database: {e}", exc_info=True)
                return False


# Singleton instance for module-level access
_cache_instance: Optional[EmbeddingsCache] = None
_singleton_lock = threading.RLock()


def get_cache_instance(
    db_path: str = "/tmp/embeddings_cache.db",
    cleanup_days: Optional[int] = None,
) -> EmbeddingsCache:
    """
    Get or create a singleton EmbeddingsCache instance.

    Args:
        db_path: Path to SQLite database file
        cleanup_days: Delete cache entries older than this many days

    Returns:
        EmbeddingsCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        with _singleton_lock:
            if _cache_instance is None:
                _cache_instance = EmbeddingsCache(
                    db_path=db_path,
                    cleanup_days=cleanup_days,
                )

    return _cache_instance
