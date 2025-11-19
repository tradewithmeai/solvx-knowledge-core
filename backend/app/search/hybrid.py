"""Hybrid search system combining FTS5 keyword search and semantic vector search.

This module provides a production-ready hybrid search implementation that:
- Combines keyword (FTS5) and semantic (vector) search results
- Uses weighted score combination based on settings
- Supports metadata filters (path, doc_id, date_range)
- Returns enriched results with chunk text, metadata, and combined scores
- Never crashes - comprehensive error handling throughout
- Provides detailed logging for debugging and monitoring
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.embeddings.providers import get_embedding_provider
from backend.app.logging import get_logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.vectorstore import get_vector_store

logger = get_logger(__name__)


# ============================================================================
# FTS5 Keyword Search
# ============================================================================


def _ensure_fts5_table(db: Session) -> bool:
    """Ensure FTS5 virtual table exists for keyword search.

    Creates the FTS5 table if it doesn't exist. This table mirrors the chunks
    table but adds full-text search capabilities.

    Args:
        db: SQLAlchemy database session

    Returns:
        bool: True if table exists or was created successfully, False otherwise
    """
    try:
        # Get raw SQLite connection from SQLAlchemy session
        connection = db.connection().connection
        cursor = connection.cursor()

        # Check if FTS5 table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='chunks_fts'
        """)

        if cursor.fetchone():
            logger.debug("FTS5 table 'chunks_fts' already exists")
            return True

        # Create FTS5 virtual table
        logger.info("Creating FTS5 virtual table 'chunks_fts'")
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize='porter unicode61 remove_diacritics 1'
            )
        """)

        connection.commit()
        logger.info("FTS5 table created successfully")
        return True

    except Exception as e:
        logger.error(
            f"Failed to create FTS5 table: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        return False


def _sync_fts5_table(db: Session) -> Dict[str, Any]:
    """Synchronize FTS5 table with chunks table.

    This function populates the FTS5 table with data from the chunks table.
    Should be called periodically or after bulk ingestion.

    Args:
        db: SQLAlchemy database session

    Returns:
        dict: Sync result with success status and count
    """
    try:
        connection = db.connection().connection
        cursor = connection.cursor()

        # Get count of chunks not in FTS
        cursor.execute("""
            SELECT COUNT(*) FROM chunks
            WHERE id NOT IN (SELECT chunk_id FROM chunks_fts)
        """)
        missing_count = cursor.fetchone()[0]

        if missing_count == 0:
            logger.debug("FTS5 table is already in sync")
            return {"success": True, "synced_count": 0, "message": "Already in sync"}

        logger.info(f"Syncing {missing_count} chunks to FTS5 table")

        # Insert missing chunks
        cursor.execute("""
            INSERT INTO chunks_fts (chunk_id, text)
            SELECT id, text FROM chunks
            WHERE id NOT IN (SELECT chunk_id FROM chunks_fts)
        """)

        synced_count = cursor.rowcount
        connection.commit()

        logger.info(f"Successfully synced {synced_count} chunks to FTS5")
        return {
            "success": True,
            "synced_count": synced_count,
            "message": f"Synced {synced_count} chunks"
        }

    except Exception as e:
        logger.error(
            f"Failed to sync FTS5 table: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        return {
            "success": False,
            "synced_count": 0,
            "error": str(e)
        }


def keyword_search(
    query: str,
    db: Session,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None
) -> List[Tuple[int, float]]:
    """Perform FTS5 keyword search on chunks.

    Uses SQLite's FTS5 full-text search to find relevant chunks based on
    keyword matching with BM25 ranking.

    Args:
        query: Search query string
        db: SQLAlchemy database session
        limit: Maximum number of results to return (default: 100)
        filters: Optional metadata filters (path, doc_id, date_range)

    Returns:
        List of tuples: [(chunk_id, bm25_score), ...]
        Returns empty list on error to prevent crashes
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to keyword_search")
        return []

    try:
        # Ensure FTS5 table exists
        if not _ensure_fts5_table(db):
            logger.error("FTS5 table not available, skipping keyword search")
            return []

        # Sync FTS5 table if needed (lightweight check)
        _sync_fts5_table(db)

        # Get raw SQLite connection
        connection = db.connection().connection
        cursor = connection.cursor()

        # Build base FTS5 query with BM25 ranking
        base_query = """
            SELECT
                chunks_fts.chunk_id,
                bm25(chunks_fts) as score
            FROM chunks_fts
        """

        # Add filters if provided
        where_clauses = ["chunks_fts MATCH ?"]
        params = [query]

        if filters:
            # Join with chunks and documents for filtering
            base_query = """
                SELECT
                    chunks_fts.chunk_id,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                INNER JOIN chunks ON chunks.id = chunks_fts.chunk_id
                INNER JOIN documents ON documents.id = chunks.document_id
            """

            # Path filter
            if filters.get("path"):
                where_clauses.append("documents.path = ?")
                params.append(filters["path"])

            # Document ID filter
            if filters.get("doc_id") is not None:
                where_clauses.append("documents.id = ?")
                params.append(int(filters["doc_id"]))

            # Date range filter
            if filters.get("date_range"):
                date_range = filters["date_range"]
                if date_range.get("start"):
                    where_clauses.append("documents.modified_ts >= ?")
                    params.append(date_range["start"])
                if date_range.get("end"):
                    where_clauses.append("documents.modified_ts <= ?")
                    params.append(date_range["end"])

        # Combine query
        where_clause = " AND ".join(where_clauses)
        full_query = f"""
            {base_query}
            WHERE {where_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        params.append(limit)

        # Execute query
        cursor.execute(full_query, params)
        results = cursor.fetchall()

        # Convert BM25 scores (negative) to positive and normalize
        # BM25 scores are negative in SQLite FTS5 (lower is better)
        if results:
            # Get absolute values and invert so higher is better
            scored_results = [(chunk_id, abs(score)) for chunk_id, score in results]

            # Normalize scores to 0-1 range
            max_score = max(score for _, score in scored_results) if scored_results else 1.0
            if max_score > 0:
                scored_results = [
                    (chunk_id, score / max_score)
                    for chunk_id, score in scored_results
                ]
        else:
            scored_results = []

        logger.debug(
            f"Keyword search found {len(scored_results)} results",
            extra={
                "query": query,
                "result_count": len(scored_results),
                "filters": filters
            }
        )

        return scored_results

    except sqlite3.OperationalError as e:
        # FTS5 query syntax errors
        logger.warning(
            f"FTS5 query syntax error: {e}",
            extra={"query": query, "error": str(e)}
        )
        return []

    except Exception as e:
        # Catch all other errors - never crash
        logger.error(
            f"Keyword search failed: {e}",
            exc_info=True,
            extra={
                "query": query,
                "error": str(e),
                "filters": filters
            }
        )
        return []


# ============================================================================
# Semantic Vector Search
# ============================================================================


def semantic_search(
    query: str,
    top_k: int = 100,
    filters: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """Perform semantic vector search on chunks.

    Uses the configured embedding provider to generate a query embedding,
    then searches the vector store for similar chunks.

    Args:
        query: Search query string
        top_k: Maximum number of results to return (default: 100)
        filters: Optional metadata filters (path, doc_id, date_range)

    Returns:
        List of tuples: [(chunk_id, similarity_score), ...]
        Returns empty list on error to prevent crashes
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to semantic_search")
        return []

    try:
        # Get embedding provider
        provider = get_embedding_provider()

        # Generate query embedding
        logger.debug(f"Generating embedding for query: {query[:50]}...")
        embeddings = provider.embed_texts([query])

        if not embeddings or not embeddings[0]:
            logger.error("Failed to generate query embedding")
            return []

        query_embedding = embeddings[0]

        # Get vector store
        vector_store = get_vector_store()

        if not vector_store.is_healthy:
            logger.error("Vector store is not healthy, skipping semantic search")
            return []

        # Search vector store
        search_result = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        if not search_result.get("success"):
            logger.error(
                f"Vector store search failed: {search_result.get('error')}",
                extra={"error": search_result.get("error")}
            )
            return []

        # Convert results to (chunk_id, score) tuples
        # ChromaDB returns distance (lower is better), convert to similarity
        results = []
        for result in search_result.get("results", []):
            chunk_id = result["id"]
            distance = result["distance"]

            # Convert cosine distance to similarity score (0-1 range)
            # Cosine distance is typically 0-2, where 0 = identical
            similarity = max(0.0, 1.0 - (distance / 2.0))

            results.append((chunk_id, similarity))

        logger.debug(
            f"Semantic search found {len(results)} results",
            extra={
                "query": query[:50],
                "result_count": len(results),
                "filters": filters
            }
        )

        return results

    except Exception as e:
        # Catch all errors - never crash
        logger.error(
            f"Semantic search failed: {e}",
            exc_info=True,
            extra={
                "query": query,
                "error": str(e),
                "filters": filters
            }
        )
        return []


# ============================================================================
# Score Combination and Result Merging
# ============================================================================


def combine_scores(
    keyword_results: List[Tuple[int, float]],
    semantic_results: List[Tuple[str, float]],
) -> List[Tuple[int, float, str]]:
    """Combine keyword and semantic search results with weighted scoring.

    Merges results from both search methods, deduplicates, and calculates
    combined scores using configured weights.

    Args:
        keyword_results: List of (chunk_id, keyword_score) tuples
        semantic_results: List of (chunk_id_str, semantic_score) tuples

    Returns:
        List of tuples: [(chunk_id, combined_score, source), ...]
        where source is 'keyword', 'semantic', or 'both'
        Sorted by combined score descending
    """
    try:
        # Get weights from settings
        keyword_weight = settings.keyword_weight
        semantic_weight = settings.semantic_weight

        # Build dictionaries for fast lookup
        keyword_scores = {int(cid): score for cid, score in keyword_results}
        semantic_scores = {int(cid): score for cid, score in semantic_results}

        # Get all unique chunk IDs
        all_chunk_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())

        # Combine scores
        combined_results = []
        for chunk_id in all_chunk_ids:
            kw_score = keyword_scores.get(chunk_id, 0.0)
            sem_score = semantic_scores.get(chunk_id, 0.0)

            # Calculate weighted combined score
            combined_score = (kw_score * keyword_weight) + (sem_score * semantic_weight)

            # Determine source
            if kw_score > 0 and sem_score > 0:
                source = "both"
            elif kw_score > 0:
                source = "keyword"
            else:
                source = "semantic"

            combined_results.append((chunk_id, combined_score, source))

        # Sort by combined score descending
        combined_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Combined {len(combined_results)} unique results",
            extra={
                "keyword_count": len(keyword_results),
                "semantic_count": len(semantic_results),
                "combined_count": len(combined_results),
                "both_sources": sum(1 for _, _, s in combined_results if s == "both")
            }
        )

        return combined_results

    except Exception as e:
        logger.error(
            f"Failed to combine scores: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        # Return empty list on error - never crash
        return []


def enrich_results(
    scored_chunks: List[Tuple[int, float, str]],
    db: Session,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Enrich chunk results with text and metadata from database.

    Fetches full chunk and document data for the top-scored results.

    Args:
        scored_chunks: List of (chunk_id, score, source) tuples
        db: SQLAlchemy database session
        limit: Maximum number of results to return

    Returns:
        List of enriched result dictionaries with:
        - chunk_id: int
        - text: str
        - combined_score: float
        - source: str ('keyword', 'semantic', or 'both')
        - metadata: dict with chunk and document info
    """
    try:
        # Take top N results
        top_chunks = scored_chunks[:limit]

        if not top_chunks:
            logger.debug("No chunks to enrich")
            return []

        # Get chunk IDs
        chunk_ids = [chunk_id for chunk_id, _, _ in top_chunks]

        # Fetch chunks and documents from database
        chunks = (
            db.query(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Chunk.id.in_(chunk_ids))
            .all()
        )

        # Build lookup dictionary
        chunk_lookup = {chunk.id: (chunk, doc) for chunk, doc in chunks}

        # Enrich results
        enriched_results = []
        for chunk_id, score, source in top_chunks:
            if chunk_id not in chunk_lookup:
                logger.warning(
                    f"Chunk {chunk_id} not found in database",
                    extra={"chunk_id": chunk_id}
                )
                continue

            chunk, doc = chunk_lookup[chunk_id]

            # Build enriched result
            result = {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "combined_score": round(score, 4),
                "source": source,
                "metadata": {
                    # Chunk metadata
                    "seq": chunk.seq,
                    "token_count": chunk.token_count,
                    "page": chunk.page,
                    "section": chunk.section,
                    "lang": chunk.lang,
                    # Document metadata
                    "doc_id": doc.id,
                    "path": doc.path,
                    "source_type": doc.source_type,
                    "title": doc.title,
                    "modified_ts": doc.modified_ts.isoformat() if doc.modified_ts else None,
                    "indexed_ts": doc.indexed_ts.isoformat() if doc.indexed_ts else None,
                    "size_bytes": doc.size_bytes,
                }
            }

            enriched_results.append(result)

        logger.debug(
            f"Enriched {len(enriched_results)} results",
            extra={"requested": len(top_chunks), "enriched": len(enriched_results)}
        )

        return enriched_results

    except Exception as e:
        logger.error(
            f"Failed to enrich results: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        # Return empty list on error - never crash
        return []


# ============================================================================
# Main Hybrid Search Function
# ============================================================================


def hybrid_search(
    query: str,
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Perform hybrid search combining keyword and semantic search.

    This is the main entry point for hybrid search. It:
    1. Performs keyword search using FTS5
    2. Performs semantic search using vector embeddings
    3. Combines results with weighted scoring
    4. Enriches results with chunk text and metadata
    5. Returns comprehensive search results

    Args:
        query: Search query string
        top_k: Number of results to return (default: 20)
        filters: Optional metadata filters:
            - path: str - Filter by document path
            - doc_id: int - Filter by document ID
            - date_range: dict with 'start' and/or 'end' ISO datetime strings
        db: SQLAlchemy database session (required)

    Returns:
        dict: Search results with:
            - success: bool
            - query: str
            - results: list of enriched result dicts
            - count: int
            - metadata: dict with search stats
            - error: str (if failed)

    Raises:
        Never raises - returns error dict on failure

    Example:
        >>> from backend.app.db.session import get_db
        >>> db = next(get_db())
        >>> results = hybrid_search("machine learning", top_k=10, db=db)
        >>> print(f"Found {results['count']} results")
        >>> for r in results['results']:
        ...     print(f"{r['chunk_id']}: {r['text'][:100]}... (score: {r['combined_score']})")
    """
    # Validate inputs
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return {
            "success": False,
            "query": query,
            "results": [],
            "count": 0,
            "error": "Empty query provided"
        }

    if db is None:
        logger.error("Database session is required for hybrid_search")
        return {
            "success": False,
            "query": query,
            "results": [],
            "count": 0,
            "error": "Database session is required"
        }

    try:
        start_time = datetime.utcnow()

        logger.info(
            f"Starting hybrid search",
            extra={
                "query": query[:100],
                "top_k": top_k,
                "filters": filters
            }
        )

        # Perform keyword search
        keyword_results = keyword_search(
            query=query,
            db=db,
            limit=top_k * 3,  # Get more candidates for merging
            filters=filters
        )

        # Perform semantic search
        semantic_results = semantic_search(
            query=query,
            top_k=top_k * 3,  # Get more candidates for merging
            filters=filters
        )

        # Check if we got any results
        if not keyword_results and not semantic_results:
            logger.info(
                f"No results found for query",
                extra={"query": query[:100]}
            )
            return {
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "metadata": {
                    "keyword_count": 0,
                    "semantic_count": 0,
                    "combined_count": 0,
                    "duration_ms": 0
                }
            }

        # Combine scores
        combined_results = combine_scores(
            keyword_results=keyword_results,
            semantic_results=semantic_results
        )

        # Enrich with full chunk data
        enriched_results = enrich_results(
            scored_chunks=combined_results,
            db=db,
            limit=top_k
        )

        # Calculate duration
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Build response
        response = {
            "success": True,
            "query": query,
            "results": enriched_results,
            "count": len(enriched_results),
            "metadata": {
                "keyword_count": len(keyword_results),
                "semantic_count": len(semantic_results),
                "combined_count": len(combined_results),
                "duration_ms": duration_ms,
                "keyword_weight": settings.keyword_weight,
                "semantic_weight": settings.semantic_weight,
                "filters": filters
            }
        }

        logger.info(
            f"Hybrid search completed successfully",
            extra={
                "query": query[:100],
                "result_count": len(enriched_results),
                "duration_ms": duration_ms,
                "keyword_count": len(keyword_results),
                "semantic_count": len(semantic_results)
            }
        )

        return response

    except Exception as e:
        # Catch all errors - never crash
        logger.error(
            f"Hybrid search failed: {e}",
            exc_info=True,
            extra={
                "query": query,
                "error": str(e),
                "filters": filters
            }
        )

        return {
            "success": False,
            "query": query,
            "results": [],
            "count": 0,
            "error": f"Search failed: {str(e)}"
        }


# ============================================================================
# Utility Functions
# ============================================================================


def rebuild_fts5_index(db: Session) -> Dict[str, Any]:
    """Rebuild the FTS5 index from scratch.

    This function:
    1. Drops the existing FTS5 table
    2. Creates a new FTS5 table
    3. Populates it with all chunks from the database

    Use this when the FTS5 index gets corrupted or needs a full refresh.

    Args:
        db: SQLAlchemy database session

    Returns:
        dict: Rebuild result with success status and count
    """
    try:
        logger.info("Starting FTS5 index rebuild")

        connection = db.connection().connection
        cursor = connection.cursor()

        # Drop existing FTS5 table
        cursor.execute("DROP TABLE IF EXISTS chunks_fts")
        connection.commit()

        logger.info("Dropped existing FTS5 table")

        # Create new FTS5 table
        if not _ensure_fts5_table(db):
            return {
                "success": False,
                "error": "Failed to create FTS5 table"
            }

        # Populate with all chunks
        cursor.execute("""
            INSERT INTO chunks_fts (chunk_id, text)
            SELECT id, text FROM chunks
        """)

        indexed_count = cursor.rowcount
        connection.commit()

        logger.info(f"FTS5 index rebuilt with {indexed_count} chunks")

        return {
            "success": True,
            "indexed_count": indexed_count,
            "message": f"Rebuilt FTS5 index with {indexed_count} chunks"
        }

    except Exception as e:
        logger.error(
            f"Failed to rebuild FTS5 index: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        return {
            "success": False,
            "indexed_count": 0,
            "error": str(e)
        }


def get_search_stats(db: Session) -> Dict[str, Any]:
    """Get statistics about the search indices.

    Args:
        db: SQLAlchemy database session

    Returns:
        dict: Statistics including chunk counts and index health
    """
    try:
        connection = db.connection().connection
        cursor = connection.cursor()

        # Count total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        # Count FTS5 indexed chunks
        cursor.execute("SELECT COUNT(*) FROM chunks_fts")
        fts_chunks = cursor.fetchone()[0]

        # Get vector store stats
        vector_store = get_vector_store()
        vector_stats = vector_store.get_collection_stats()

        stats = {
            "total_chunks": total_chunks,
            "fts_indexed": fts_chunks,
            "fts_coverage": round(fts_chunks / total_chunks * 100, 2) if total_chunks > 0 else 0,
            "vector_indexed": vector_stats.get("total_chunks", 0),
            "vector_coverage": round(
                vector_stats.get("total_chunks", 0) / total_chunks * 100, 2
            ) if total_chunks > 0 else 0,
            "keyword_weight": settings.keyword_weight,
            "semantic_weight": settings.semantic_weight,
        }

        logger.debug("Retrieved search statistics", extra=stats)
        return stats

    except Exception as e:
        logger.error(
            f"Failed to get search stats: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        return {
            "error": str(e)
        }
