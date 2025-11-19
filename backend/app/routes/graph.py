"""Graph API routes for knowledge graph visualization."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.graph.build_graph import build_graph
from backend.app.logging import get_logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graph")
def get_knowledge_graph(
    similarity_threshold: float = Query(
        0.55,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for including edges (0.0-1.0)",
    ),
    max_nodes: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of nodes to include in graph",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Get knowledge graph with nodes and edges in D3.js compatible format.

    The graph represents chunks as nodes and semantic relationships as edges.
    Uses semantic similarity to determine edge connections between chunks.

    Query Parameters:
        similarity_threshold: Minimum similarity score for edges (default: 0.55)
        max_nodes: Maximum number of nodes to return (default: 100)

    Returns:
        JSON response with:
            - success: bool - Whether the request succeeded
            - nodes: list - Graph nodes with:
                - id: str - Node identifier (chunk ID)
                - label: str - Short chunk text preview
                - text: str - Full chunk text
                - chunk_id: int - Chunk database ID
                - document_id: int - Document ID
                - doc_path: str - Document path
                - section: str | null - Section name
                - page: int | null - Page number
                - group: int - Community/group identifier for clustering
                - size: int - Relative node size (1-10)
            - edges: list - Graph edges with:
                - source: str - Source node ID (chunk ID)
                - target: str - Target node ID (chunk ID)
                - weight: float - Edge weight (similarity score)
                - distance: float - Computed distance for layout (1/weight)
            - metadata: dict - Graph metadata with:
                - total_nodes: int - Total number of nodes
                - total_edges: int - Total number of edges
                - communities_count: int - Number of detected communities
                - similarity_threshold: float - Applied threshold
                - max_nodes_param: int - Requested max nodes

    Error responses:
        400: Bad request (invalid parameters)
        500: Server error (graph building failed)

    Example:
        GET /api/graph
        GET /api/graph?similarity_threshold=0.6&max_nodes=200
    """
    request_id = None
    try:
        # Validate parameters
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            logger.warning(
                "Invalid similarity threshold provided",
                extra={"similarity_threshold": similarity_threshold},
            )
            raise HTTPException(
                status_code=400,
                detail="similarity_threshold must be between 0.0 and 1.0",
            )

        if max_nodes < 1 or max_nodes > 1000:
            logger.warning(
                "Invalid max_nodes parameter",
                extra={"max_nodes": max_nodes},
            )
            raise HTTPException(
                status_code=400,
                detail="max_nodes must be between 1 and 1000",
            )

        logger.info(
            "Graph request received",
            extra={
                "similarity_threshold": similarity_threshold,
                "max_nodes": max_nodes,
            },
        )

        # Get database statistics before building graph
        total_chunks = db.query(func.count(Chunk.id)).scalar() or 0
        total_documents = db.query(func.count(Document.id)).scalar() or 0

        logger.debug(
            "Database statistics retrieved",
            extra={
                "total_chunks": total_chunks,
                "total_documents": total_documents,
            },
        )

        if total_chunks == 0:
            logger.warning("No chunks found in database")
            return {
                "success": True,
                "nodes": [],
                "edges": [],
                "metadata": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "communities_count": 0,
                    "similarity_threshold": similarity_threshold,
                    "max_nodes_param": max_nodes,
                    "total_chunks_available": total_chunks,
                    "total_documents_available": total_documents,
                },
            }

        # Build graph using the graph builder
        try:
            logger.debug(
                "Building knowledge graph",
                extra={
                    "similarity_threshold": similarity_threshold,
                    "max_nodes": max_nodes,
                },
            )
            graph_data = build_graph(
                db=db,
                similarity_threshold=similarity_threshold,
                max_nodes=max_nodes,
            )
            logger.debug(
                "Knowledge graph built successfully",
                extra={
                    "nodes_count": len(graph_data.get("nodes", [])),
                    "edges_count": len(graph_data.get("edges", [])),
                },
            )
        except Exception as e:
            logger.error(
                "Failed to build knowledge graph",
                extra={
                    "error": str(e),
                    "similarity_threshold": similarity_threshold,
                    "max_nodes": max_nodes,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build knowledge graph: {str(e)}",
            )

        # Extract nodes and edges with D3.js compatible formatting
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        communities = graph_data.get("communities", {})

        # Count unique communities
        unique_communities = len(set(communities.values())) if communities else 0

        logger.info(
            "Knowledge graph retrieved successfully",
            extra={
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "communities_count": unique_communities,
                "similarity_threshold": similarity_threshold,
            },
        )

        # Format response with comprehensive metadata
        response = {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "communities_count": unique_communities,
                "similarity_threshold": similarity_threshold,
                "max_nodes_param": max_nodes,
                "total_chunks_available": total_chunks,
                "total_documents_available": total_documents,
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in graph endpoint",
            extra={
                "error": str(e),
                "similarity_threshold": similarity_threshold,
                "max_nodes": max_nodes,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while retrieving the knowledge graph",
        )
