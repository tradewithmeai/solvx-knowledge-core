"""Search API routes for hybrid semantic and keyword search."""

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.embeddings import embed_texts, get_embedding_provider
from backend.app.logging import get_logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.vectorstore.chroma_store import get_vector_store

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


# Response models (would use Pydantic models in production)
class SearchResult:
    """Individual search result with metadata and citation."""

    def __init__(
        self,
        chunk_id: int,
        text: str,
        metadata: dict[str, Any],
        score: float,
        document_path: str,
        document_id: int,
        seq: int,
        page: int | None = None,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score
        self.document_path = document_path
        self.document_id = document_id
        self.seq = seq
        self.page = page

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary with citation."""
        citation = f"solvx://doc/{self.document_id}#chunk={self.seq}"
        if self.page is not None:
            citation += f"&page={self.page}"

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "document_path": self.document_path,
            "citation": citation,
        }


def _parse_filters(filters_json: str | None) -> dict[str, Any]:
    """
    Parse filter JSON string.

    Args:
        filters_json: JSON string with filters

    Returns:
        Parsed filters dict

    Raises:
        ValueError: If JSON is invalid
    """
    if not filters_json:
        return {}

    try:
        filters = json.loads(filters_json)
        if not isinstance(filters, dict):
            raise ValueError("Filters must be a JSON object")
        return filters
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid filters JSON: {str(e)}")


def _get_query_embedding(query: str) -> list[float]:
    """
    Get embedding for a query string.

    Args:
        query: Query text

    Returns:
        Embedding vector

    Raises:
        RuntimeError: If embedding fails
    """
    try:
        logger.debug(f"Generating embedding for query", extra={"query_length": len(query)})
        embeddings = embed_texts([query])
        if not embeddings:
            raise RuntimeError("Failed to generate embedding")
        return embeddings[0]
    except Exception as e:
        logger.error(f"Failed to generate query embedding", extra={"error": str(e)})
        raise


def _semantic_search(
    query_embedding: list[float],
    top_k: int,
    filters: dict[str, Any],
    db: Session,
) -> list[SearchResult]:
    """
    Perform semantic search using vector similarity.

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        filters: Metadata filters
        db: Database session

    Returns:
        List of SearchResult objects
    """
    try:
        vector_store = get_vector_store()

        # Build vector store filters
        vs_filters = {}
        if "doc_id" in filters and filters["doc_id"] is not None:
            vs_filters["doc_id"] = filters["doc_id"]
        if "path" in filters and filters["path"]:
            vs_filters["path"] = filters["path"]
        if "date_range" in filters and filters["date_range"]:
            vs_filters["date_range"] = filters["date_range"]

        # Perform semantic search
        logger.debug(
            "Performing semantic search",
            extra={"top_k": top_k, "filters": bool(vs_filters)},
        )
        vs_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=vs_filters if vs_filters else None,
        )

        if not vs_results["success"]:
            logger.error(
                "Semantic search failed",
                extra={"error": vs_results.get("error", "Unknown error")},
            )
            return []

        results = []
        for result in vs_results["results"]:
            chunk_id = int(result["id"])
            metadata = result["metadata"]

            # Convert distance (lower is better) to similarity score (0-1)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # Convert to similarity: 1 - (distance / 2)
            similarity_score = 1 - (result["distance"] / 2)

            # Get document path
            doc_id = metadata.get("doc_id", 0)
            doc = db.query(Document).filter(Document.id == doc_id).first()
            doc_path = doc.path if doc else metadata.get("path", "")

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    metadata=metadata,
                    score=similarity_score,
                    document_path=doc_path,
                    document_id=doc_id,
                    seq=metadata.get("seq", 0),
                    page=metadata.get("page"),
                )
            )

        logger.debug(
            "Semantic search completed",
            extra={"results_count": len(results)},
        )
        return results

    except Exception as e:
        logger.error(
            "Semantic search error",
            extra={"error": str(e)},
            exc_info=True,
        )
        return []


def _keyword_search(
    query: str,
    top_k: int,
    filters: dict[str, Any],
    db: Session,
) -> list[SearchResult]:
    """
    Perform keyword search using database full-text search.

    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Metadata filters
        db: Database session

    Returns:
        List of SearchResult objects
    """
    try:
        logger.debug(
            "Performing keyword search",
            extra={"query": query, "top_k": top_k},
        )

        # Build base query
        q = db.query(Chunk, Document).join(Document, Chunk.document_id == Document.id)

        # Apply filters
        if "doc_id" in filters and filters["doc_id"] is not None:
            q = q.filter(Chunk.document_id == filters["doc_id"])

        if "path" in filters and filters["path"]:
            q = q.filter(Document.path == filters["path"])

        if "date_range" in filters and filters["date_range"]:
            date_range = filters["date_range"]
            if "start" in date_range:
                try:
                    start_dt = datetime.fromisoformat(date_range["start"])
                    q = q.filter(Document.modified_ts >= start_dt)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid date_range.start format",
                        extra={"value": date_range.get("start")},
                    )

            if "end" in date_range:
                try:
                    end_dt = datetime.fromisoformat(date_range["end"])
                    q = q.filter(Document.modified_ts <= end_dt)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid date_range.end format",
                        extra={"value": date_range.get("end")},
                    )

        # Apply keyword filters (simple text search)
        # For production, consider using full-text search extensions
        keywords = query.lower().split()
        for keyword in keywords:
            q = q.filter(
                or_(
                    Chunk.text.ilike(f"%{keyword}%"),
                    Document.title.ilike(f"%{keyword}%"),
                )
            )

        # Execute query
        rows = q.limit(top_k).all()

        results = []
        for chunk, doc in rows:
            # Calculate simple relevance score based on keyword matches
            matches = sum(1 for kw in keywords if kw.lower() in chunk.text.lower())
            relevance_score = min(1.0, matches / max(1, len(keywords)))

            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    metadata={
                        "doc_id": chunk.document_id,
                        "path": doc.path,
                        "seq": chunk.seq,
                        "page": chunk.page,
                        "section": chunk.section,
                    },
                    score=relevance_score,
                    document_path=doc.path,
                    document_id=chunk.document_id,
                    seq=chunk.seq,
                    page=chunk.page,
                )
            )

        logger.debug(
            "Keyword search completed",
            extra={"results_count": len(results)},
        )
        return results

    except Exception as e:
        logger.error(
            "Keyword search error",
            extra={"error": str(e)},
            exc_info=True,
        )
        return []


def _hybrid_search(
    query: str,
    top_k: int,
    filters: dict[str, Any],
    db: Session,
) -> list[SearchResult]:
    """
    Perform hybrid search combining semantic and keyword search.

    Uses configured weights from settings:
    - semantic_weight: weight for semantic search (default 0.7)
    - keyword_weight: weight for keyword search (default 0.3)

    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Metadata filters
        db: Database session

    Returns:
        List of SearchResult objects, ranked by combined score
    """
    try:
        logger.info(
            "Starting hybrid search",
            extra={
                "query": query,
                "top_k": top_k,
                "semantic_weight": settings.semantic_weight,
                "keyword_weight": settings.keyword_weight,
            },
        )

        # Get query embedding for semantic search
        query_embedding = _get_query_embedding(query)

        # Perform both searches
        semantic_results = _semantic_search(query_embedding, top_k, filters, db)
        keyword_results = _keyword_search(query, top_k, filters, db)

        # Combine results with weighted scoring
        # Use chunk_id as key for deduplication
        combined: dict[int, SearchResult] = {}

        # Add semantic results
        for result in semantic_results:
            weighted_score = result.score * settings.semantic_weight
            result.score = weighted_score
            combined[result.chunk_id] = result

        # Merge keyword results
        for result in keyword_results:
            weighted_score = result.score * settings.keyword_weight
            if result.chunk_id in combined:
                # Average the scores for duplicate results
                combined[result.chunk_id].score = (
                    combined[result.chunk_id].score + weighted_score
                ) / 2
                logger.debug(
                    "Merged duplicate search result",
                    extra={"chunk_id": result.chunk_id},
                )
            else:
                result.score = weighted_score
                combined[result.chunk_id] = result

        # Sort by combined score (descending)
        final_results = sorted(combined.values(), key=lambda r: r.score, reverse=True)

        # Return top_k results
        final_results = final_results[:top_k]

        logger.info(
            "Hybrid search completed",
            extra={
                "total_results": len(final_results),
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results),
            },
        )

        return final_results

    except Exception as e:
        logger.error(
            "Hybrid search error",
            extra={"error": str(e), "query": query},
            exc_info=True,
        )
        raise


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(20, ge=1, le=100, description="Number of results to return"),
    filters: str | None = Query(None, description="JSON filters: {path, doc_id, date_range}"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Hybrid semantic and keyword search endpoint.

    Combines vector similarity search with keyword matching for comprehensive results.

    Query Parameters:
        q: Search query (required, 1-1000 characters)
        top_k: Number of results to return (default: 20, max: 100)
        filters: JSON object with optional filters:
            - path: str - Filter by document path (exact match)
            - doc_id: int - Filter by document ID
            - date_range: object with optional 'start' and 'end' ISO datetime strings

    Returns:
        JSON response with:
            - success: bool - Whether search succeeded
            - query: str - The original query
            - results: list - Search results with:
                - chunk_id: int
                - text: str - Chunk text content
                - metadata: dict - Chunk metadata
                - score: float - Combined relevance score (0-1)
                - document_path: str
                - citation: str - Citation in format solvx://doc/{doc_id}#chunk={seq}[&page={page}]
            - count: int - Number of results returned
            - total_documents: int - Total indexed documents
            - total_chunks: int - Total indexed chunks

    Error responses:
        400: Bad input (invalid query, invalid JSON filters)
        500: Server error (search failed)

    Example:
        GET /api/search?q=machine+learning&top_k=10
        GET /api/search?q=python&filters={"doc_id":5}
        GET /api/search?q=data&filters={"date_range":{"start":"2024-01-01","end":"2024-12-31"}}
    """
    request_id = None
    try:
        # Parse and validate filters
        try:
            parsed_filters = _parse_filters(filters)
            logger.debug(
                "Filters parsed successfully",
                extra={"filter_keys": list(parsed_filters.keys())},
            )
        except ValueError as e:
            logger.warning(
                "Invalid filters provided",
                extra={"error": str(e), "filters": filters},
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filters: {str(e)}",
            )

        # Validate query
        if not q or not q.strip():
            logger.warning("Empty query provided")
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty",
            )

        query = q.strip()
        logger.info(
            "Search request received",
            extra={
                "query": query,
                "top_k": top_k,
                "has_filters": bool(parsed_filters),
            },
        )

        # Perform hybrid search
        try:
            search_results = _hybrid_search(query, top_k, parsed_filters, db)
        except Exception as e:
            logger.error(
                "Search execution failed",
                extra={"error": str(e), "query": query},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}",
            )

        # Get database statistics
        doc_count = db.query(func.count(Document.id)).scalar() or 0
        chunk_count = db.query(func.count(Chunk.id)).scalar() or 0

        # Format response
        results_data = [result.to_dict() for result in search_results]

        response = {
            "success": True,
            "query": query,
            "results": results_data,
            "count": len(results_data),
            "total_documents": doc_count,
            "total_chunks": chunk_count,
        }

        logger.info(
            "Search completed successfully",
            extra={
                "query": query,
                "results_count": len(results_data),
                "doc_count": doc_count,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in search endpoint",
            extra={"error": str(e), "query": q},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during search",
        )
