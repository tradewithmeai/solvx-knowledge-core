"""ChromaDB vector store integration for SolVX Knowledge Core.

Provides persistent vector storage with:
- Cosine similarity search
- Metadata filtering (path, doc_id, date range)
- Batch upsert and delete operations
- Collection statistics and health checks
- Comprehensive error handling and logging
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from backend.app.config import settings
from backend.app.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store for document chunks with persistent storage."""

    COLLECTION_NAME = "solvx_chunks"
    DISTANCE_METRIC = "cosine"  # ChromaDB default and recommended for embeddings

    def __init__(self, persist_directory: Path | None = None):
        """Initialize ChromaDB client and collection.

        Args:
            persist_directory: Path to ChromaDB storage directory.
                             Defaults to settings.chroma_path.

        Raises:
            RuntimeError: If ChromaDB initialization fails critically.
        """
        self.persist_directory = persist_directory or settings.chroma_path
        self._client: chromadb.Client | None = None
        self._collection: Collection | None = None

        try:
            self._initialize_client()
            self._initialize_collection()
            logger.info(
                f"ChromaDB initialized successfully",
                extra={
                    "collection": self.COLLECTION_NAME,
                    "path": str(self.persist_directory),
                    "metric": self.DISTANCE_METRIC,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize ChromaDB",
                extra={"error": str(e), "path": str(self.persist_directory)},
            )
            # Don't raise - allow graceful degradation
            self._client = None
            self._collection = None

    def _initialize_client(self) -> None:
        """Initialize persistent ChromaDB client."""
        try:
            # Ensure persistence directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Create persistent client with default settings
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.debug(
                "ChromaDB client created",
                extra={"path": str(self.persist_directory)},
            )
        except Exception as e:
            logger.error(
                "Failed to create ChromaDB client",
                extra={"error": str(e), "path": str(self.persist_directory)},
                exc_info=True,
            )
            raise RuntimeError(f"ChromaDB client initialization failed: {e}") from e

    def _initialize_collection(self) -> None:
        """Initialize or retrieve the solvx_chunks collection."""
        if not self._client:
            raise RuntimeError("ChromaDB client not initialized")

        try:
            # Get or create collection with cosine similarity (default)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": self.DISTANCE_METRIC},
            )
            logger.debug(
                "Collection initialized",
                extra={"collection": self.COLLECTION_NAME},
            )
        except Exception as e:
            logger.error(
                "Failed to initialize collection",
                extra={"error": str(e), "collection": self.COLLECTION_NAME},
                exc_info=True,
            )
            raise RuntimeError(f"Collection initialization failed: {e}") from e

    @property
    def is_healthy(self) -> bool:
        """Check if ChromaDB is healthy and operational."""
        return self._client is not None and self._collection is not None

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            dict: Health status with details:
                - healthy: bool
                - client_initialized: bool
                - collection_initialized: bool
                - collection_name: str
                - persist_directory: str
                - total_chunks: int (if healthy)
                - error: str (if unhealthy)
        """
        status = {
            "healthy": False,
            "client_initialized": self._client is not None,
            "collection_initialized": self._collection is not None,
            "collection_name": self.COLLECTION_NAME,
            "persist_directory": str(self.persist_directory),
        }

        try:
            if self.is_healthy:
                # Try to get collection count
                count = self._collection.count()
                status["healthy"] = True
                status["total_chunks"] = count
                logger.debug("Health check passed", extra={"count": count})
            else:
                status["error"] = "ChromaDB client or collection not initialized"
                logger.warning("Health check failed: not initialized")
        except Exception as e:
            status["error"] = str(e)
            logger.error(
                "Health check failed",
                extra={"error": str(e)},
                exc_info=True,
            )

        return status

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Upsert document chunks into ChromaDB.

        Each chunk dict should contain:
        - id: int - Unique chunk ID
        - embedding: list[float] - Vector embedding
        - text: str - Chunk text content
        - doc_id: int - Document ID
        - path: str - Document path
        - page: int | None - Page number (optional)
        - section: str | None - Section name (optional)
        - seq: int - Sequence number in document
        - modified_ts: datetime | str | None - Document modification timestamp

        Args:
            chunks: List of chunk dictionaries with embeddings and metadata.

        Returns:
            dict: Operation result with:
                - success: bool
                - chunks_upserted: int
                - error: str (if failed)
        """
        if not self.is_healthy:
            error = "ChromaDB not initialized"
            logger.error(f"Upsert failed: {error}")
            return {"success": False, "chunks_upserted": 0, "error": error}

        if not chunks:
            logger.warning("Upsert called with empty chunks list")
            return {"success": True, "chunks_upserted": 0}

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunks:
                # Validate required fields
                if not all(k in chunk for k in ["id", "embedding", "text", "doc_id"]):
                    logger.warning(
                        "Skipping chunk missing required fields",
                        extra={"chunk_keys": list(chunk.keys())},
                    )
                    continue

                # ChromaDB requires string IDs
                ids.append(str(chunk["id"]))
                embeddings.append(chunk["embedding"])
                documents.append(chunk["text"])

                # Build metadata (ChromaDB supports: str, int, float, bool)
                metadata = {
                    "doc_id": int(chunk["doc_id"]),
                    "path": str(chunk.get("path", "")),
                    "seq": int(chunk.get("seq", 0)),
                }

                # Add optional fields
                if chunk.get("page") is not None:
                    metadata["page"] = int(chunk["page"])

                if chunk.get("section"):
                    metadata["section"] = str(chunk["section"])

                # Handle datetime - convert to ISO string
                if chunk.get("modified_ts"):
                    ts = chunk["modified_ts"]
                    if isinstance(ts, datetime):
                        metadata["modified_ts"] = ts.isoformat()
                    elif isinstance(ts, str):
                        metadata["modified_ts"] = ts
                    # ChromaDB doesn't support datetime objects directly

                metadatas.append(metadata)

            if not ids:
                logger.warning("No valid chunks to upsert after validation")
                return {"success": True, "chunks_upserted": 0}

            # Upsert to ChromaDB (upsert = add or update)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "Chunks upserted successfully",
                extra={"count": len(ids)},
            )

            return {
                "success": True,
                "chunks_upserted": len(ids),
            }

        except Exception as e:
            logger.error(
                "Failed to upsert chunks",
                extra={"error": str(e), "chunk_count": len(chunks)},
                exc_info=True,
            )
            return {
                "success": False,
                "chunks_upserted": 0,
                "error": str(e),
            }

    def delete_by_document_id(self, doc_id: int) -> dict[str, Any]:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: Document ID to delete chunks for.

        Returns:
            dict: Operation result with:
                - success: bool
                - chunks_deleted: int (approximate)
                - error: str (if failed)
        """
        if not self.is_healthy:
            error = "ChromaDB not initialized"
            logger.error(f"Delete failed: {error}")
            return {"success": False, "chunks_deleted": 0, "error": error}

        try:
            # ChromaDB delete by metadata filter
            # First, count chunks to be deleted (for logging)
            count_before = self._collection.count()

            self._collection.delete(
                where={"doc_id": int(doc_id)},
            )

            count_after = self._collection.count()
            deleted = count_before - count_after

            logger.info(
                "Chunks deleted by document ID",
                extra={"doc_id": doc_id, "deleted_count": deleted},
            )

            return {
                "success": True,
                "chunks_deleted": deleted,
            }

        except Exception as e:
            logger.error(
                "Failed to delete chunks",
                extra={"error": str(e), "doc_id": doc_id},
                exc_info=True,
            )
            return {
                "success": False,
                "chunks_deleted": 0,
                "error": str(e),
            }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for similar chunks using vector similarity.

        Supports metadata filters:
        - path: str - Filter by document path (exact match)
        - doc_id: int - Filter by document ID
        - date_range: dict - Filter by modified_ts with 'start' and/or 'end' ISO strings

        Args:
            query_embedding: Query vector embedding.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters dict.

        Returns:
            dict: Search results with:
                - success: bool
                - results: list of dicts with:
                    - id: str - Chunk ID
                    - text: str - Chunk text
                    - metadata: dict - Chunk metadata
                    - distance: float - Cosine distance (lower = more similar)
                - count: int - Number of results returned
                - error: str (if failed)
        """
        if not self.is_healthy:
            error = "ChromaDB not initialized"
            logger.error(f"Search failed: {error}")
            return {"success": False, "results": [], "count": 0, "error": error}

        try:
            # Build ChromaDB where filter
            where_filter = None
            if filters:
                where_filter = self._build_where_filter(filters)

            # Query ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 1000),  # ChromaDB limit
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "id": chunk_id,
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                        }
                    )

            logger.debug(
                "Search completed",
                extra={
                    "results_count": len(formatted_results),
                    "filters": filters,
                },
            )

            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
            }

        except Exception as e:
            logger.error(
                "Search failed",
                extra={"error": str(e), "top_k": top_k, "filters": filters},
                exc_info=True,
            )
            return {
                "success": False,
                "results": [],
                "count": 0,
                "error": str(e),
            }

    def _build_where_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Build ChromaDB where filter from user filters.

        Supports:
        - path: str - Exact match
        - doc_id: int - Exact match
        - date_range: dict with 'start' and/or 'end' ISO datetime strings

        Args:
            filters: User-provided filters dict.

        Returns:
            ChromaDB where filter dict, or None if no valid filters.
        """
        where_conditions = []

        # Path filter
        if "path" in filters and filters["path"]:
            where_conditions.append({"path": {"$eq": str(filters["path"])}})

        # Document ID filter
        if "doc_id" in filters and filters["doc_id"] is not None:
            where_conditions.append({"doc_id": {"$eq": int(filters["doc_id"])}})

        # Date range filter
        if "date_range" in filters and filters["date_range"]:
            date_range = filters["date_range"]
            if "start" in date_range:
                where_conditions.append(
                    {"modified_ts": {"$gte": str(date_range["start"])}}
                )
            if "end" in date_range:
                where_conditions.append(
                    {"modified_ts": {"$lte": str(date_range["end"])}}
                )

        # Combine conditions with AND logic
        if not where_conditions:
            return None
        elif len(where_conditions) == 1:
            return where_conditions[0]
        else:
            return {"$and": where_conditions}

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics and metadata.

        Returns:
            dict: Collection statistics with:
                - success: bool
                - collection_name: str
                - total_chunks: int
                - persist_directory: str
                - distance_metric: str
                - error: str (if failed)
        """
        if not self.is_healthy:
            error = "ChromaDB not initialized"
            logger.error(f"Get stats failed: {error}")
            return {
                "success": False,
                "collection_name": self.COLLECTION_NAME,
                "total_chunks": 0,
                "error": error,
            }

        try:
            count = self._collection.count()

            stats = {
                "success": True,
                "collection_name": self.COLLECTION_NAME,
                "total_chunks": count,
                "persist_directory": str(self.persist_directory),
                "distance_metric": self.DISTANCE_METRIC,
            }

            logger.debug("Collection stats retrieved", extra={"count": count})

            return stats

        except Exception as e:
            logger.error(
                "Failed to get collection stats",
                extra={"error": str(e)},
                exc_info=True,
            )
            return {
                "success": False,
                "collection_name": self.COLLECTION_NAME,
                "total_chunks": 0,
                "error": str(e),
            }

    def reset_collection(self) -> dict[str, Any]:
        """Reset (delete and recreate) the collection.

        WARNING: This deletes all stored vectors and metadata!

        Returns:
            dict: Operation result with:
                - success: bool
                - message: str
                - error: str (if failed)
        """
        if not self._client:
            error = "ChromaDB client not initialized"
            logger.error(f"Reset failed: {error}")
            return {"success": False, "error": error}

        try:
            # Delete existing collection
            try:
                self._client.delete_collection(name=self.COLLECTION_NAME)
                logger.info("Collection deleted", extra={"collection": self.COLLECTION_NAME})
            except Exception:
                # Collection might not exist, that's fine
                pass

            # Reinitialize collection
            self._initialize_collection()

            logger.warning(
                "Collection reset completed",
                extra={"collection": self.COLLECTION_NAME},
            )

            return {
                "success": True,
                "message": f"Collection '{self.COLLECTION_NAME}' reset successfully",
            }

        except Exception as e:
            logger.error(
                "Failed to reset collection",
                extra={"error": str(e)},
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
            }


# Global vector store instance
_vector_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    """Get or create the global ChromaDB vector store instance.

    Returns:
        ChromaVectorStore: Global vector store instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store
