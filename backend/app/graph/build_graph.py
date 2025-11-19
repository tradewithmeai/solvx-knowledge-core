"""Knowledge graph builder for SolVX Knowledge Core.

Builds a graph from document embeddings with:
- Document nodes (with centroids computed from chunk embeddings)
- Tag nodes extracted from document metadata
- Edges based on cosine similarity between document centroids
- Graph metrics: communities (Louvain), degrees, betweenness centrality
- 1-hour result caching for performance
- Comprehensive error handling and logging
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any

import networkx as nx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.logging import get_logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.vectorstore.chroma_store import get_vector_store

logger = get_logger(__name__)

# Global cache for graph results
_graph_cache: dict[str, Any] = {
    "data": None,
    "timestamp": None,
    "threshold": None,
}
_cache_ttl_seconds = 3600  # 1 hour


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}", exc_info=True)
        return 0.0


def _get_chunk_embeddings(db: Session, doc_id: int) -> list[list[float]]:
    """Retrieve embeddings for all chunks of a document from ChromaDB.

    Args:
        db: Database session
        doc_id: Document ID

    Returns:
        List of embedding vectors for the document's chunks
    """
    embeddings = []

    try:
        # Get chunk IDs for this document
        stmt = select(Chunk.id).where(Chunk.document_id == doc_id).order_by(Chunk.seq)
        result = db.execute(stmt)
        chunk_ids = [str(row[0]) for row in result.all()]

        if not chunk_ids:
            logger.debug(f"No chunks found for document {doc_id}")
            return []

        # Get embeddings from ChromaDB
        vector_store = get_vector_store()
        if not vector_store.is_healthy:
            logger.warning("Vector store not healthy, cannot retrieve embeddings")
            return []

        try:
            # Query ChromaDB for these specific chunk IDs
            results = vector_store._collection.get(
                ids=chunk_ids,
                include=["embeddings"]
            )

            if results and results.get("embeddings"):
                embeddings = results["embeddings"]
                logger.debug(f"Retrieved {len(embeddings)} embeddings for document {doc_id}")
            else:
                logger.debug(f"No embeddings found in ChromaDB for document {doc_id}")

        except Exception as e:
            logger.error(f"Error retrieving embeddings from ChromaDB: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error getting chunk embeddings for document {doc_id}: {e}", exc_info=True)

    return embeddings


def _compute_document_centroid(embeddings: list[list[float]]) -> np.ndarray | None:
    """Compute the centroid (mean) of chunk embeddings for a document.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Centroid vector as numpy array, or None if computation fails
    """
    try:
        if not embeddings:
            return None

        # Convert to numpy array and compute mean
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)

        return centroid

    except Exception as e:
        logger.error(f"Error computing document centroid: {e}", exc_info=True)
        return None


def _extract_tags(tags_json: str | None) -> list[str]:
    """Extract tags from JSON string.

    Args:
        tags_json: JSON string containing tags array

    Returns:
        List of tag strings
    """
    try:
        if not tags_json:
            return []

        tags = json.loads(tags_json)
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if tag]
        return []

    except Exception as e:
        logger.debug(f"Error parsing tags: {e}")
        return []


def _compute_graph_metrics(G: nx.Graph) -> dict[str, Any]:
    """Compute graph metrics including communities, degrees, and centrality.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary with:
            - communities: dict mapping node_id to community_id
            - degrees: dict mapping node_id to degree
            - betweenness: dict mapping node_id to betweenness centrality
    """
    metrics = {
        "communities": {},
        "degrees": {},
        "betweenness": {},
    }

    try:
        # Compute node degrees
        degrees = dict(G.degree())
        metrics["degrees"] = {str(k): v for k, v in degrees.items()}
        logger.debug(f"Computed degrees for {len(degrees)} nodes")

    except Exception as e:
        logger.error(f"Error computing node degrees: {e}", exc_info=True)

    try:
        # Compute betweenness centrality
        # For large graphs, use sampling to speed up computation
        if len(G.nodes()) > 100:
            # Sample-based approximation for large graphs
            k = min(100, len(G.nodes()))
            betweenness = nx.betweenness_centrality(G, k=k)
            logger.debug(f"Computed approximate betweenness centrality (k={k})")
        else:
            betweenness = nx.betweenness_centrality(G)
            logger.debug(f"Computed exact betweenness centrality")

        metrics["betweenness"] = {str(k): float(v) for k, v in betweenness.items()}

    except Exception as e:
        logger.error(f"Error computing betweenness centrality: {e}", exc_info=True)

    try:
        # Detect communities using Louvain algorithm
        if len(G.nodes()) > 0 and len(G.edges()) > 0:
            # Import here to handle optional dependency gracefully
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(G, seed=42)

                # Map node to community ID
                community_map = {}
                for comm_id, community in enumerate(communities):
                    for node in community:
                        community_map[str(node)] = comm_id

                metrics["communities"] = community_map
                logger.info(f"Detected {len(communities)} communities using Louvain algorithm")

            except ImportError:
                logger.warning("Louvain algorithm not available, skipping community detection")
            except Exception as e:
                logger.error(f"Error in Louvain community detection: {e}", exc_info=True)
        else:
            logger.debug("Graph too small for community detection")

    except Exception as e:
        logger.error(f"Error in community detection: {e}", exc_info=True)

    return metrics


def build_graph(db: Session, similarity_threshold: float = 0.55) -> dict[str, Any]:
    """Build knowledge graph from document embeddings.

    Creates a graph with:
    - Document nodes (with centroids from chunk embeddings)
    - Tag nodes (extracted from document metadata)
    - Edges between documents based on cosine similarity > threshold
    - Graph metrics: communities (Louvain), degrees, betweenness centrality

    Results are cached for 1 hour to improve performance.

    Args:
        db: Database session
        similarity_threshold: Minimum cosine similarity for edge creation (default: 0.55)

    Returns:
        Dictionary with:
            - nodes: List of node dicts with {id, label, type, score, metadata}
            - edges: List of edge dicts with {source, target, weight}
            - metrics: Graph metrics (communities, degrees, betweenness)
            - stats: Statistics about the graph
            - cached: Whether result was from cache
    """
    global _graph_cache

    # Check cache
    try:
        if (_graph_cache["data"] is not None and
            _graph_cache["timestamp"] is not None and
            _graph_cache["threshold"] == similarity_threshold):

            cache_age = datetime.now() - _graph_cache["timestamp"]
            if cache_age < timedelta(seconds=_cache_ttl_seconds):
                logger.info(f"Returning cached graph (age: {cache_age.total_seconds():.0f}s)")
                result = _graph_cache["data"].copy()
                result["cached"] = True
                return result

    except Exception as e:
        logger.debug(f"Error checking cache: {e}")

    logger.info(f"Building knowledge graph with similarity_threshold={similarity_threshold}")
    start_time = time.time()

    # Initialize result structure
    result = {
        "nodes": [],
        "edges": [],
        "metrics": {
            "communities": {},
            "degrees": {},
            "betweenness": {},
        },
        "stats": {
            "total_documents": 0,
            "total_tags": 0,
            "total_edges": 0,
            "similarity_threshold": similarity_threshold,
            "build_time_seconds": 0,
        },
        "cached": False,
    }

    try:
        # Step 1: Fetch all successfully embedded documents
        logger.debug("Fetching documents from database")
        stmt = select(Document).where(Document.status == "embedded").order_by(Document.id)
        documents = db.execute(stmt).scalars().all()

        if not documents:
            logger.warning("No embedded documents found in database")
            return result

        logger.info(f"Found {len(documents)} embedded documents")
        result["stats"]["total_documents"] = len(documents)

        # Step 2: Compute document centroids and collect tags
        doc_centroids: dict[int, np.ndarray] = {}
        all_tags: set[str] = set()

        for doc in documents:
            try:
                # Get chunk embeddings
                embeddings = _get_chunk_embeddings(db, doc.id)

                if embeddings:
                    centroid = _compute_document_centroid(embeddings)
                    if centroid is not None:
                        doc_centroids[doc.id] = centroid
                    else:
                        logger.debug(f"Could not compute centroid for document {doc.id}")
                else:
                    logger.debug(f"No embeddings found for document {doc.id}")

                # Extract tags
                tags = _extract_tags(doc.tags)
                all_tags.update(tags)

            except Exception as e:
                logger.error(f"Error processing document {doc.id}: {e}", exc_info=True)
                continue

        logger.info(f"Computed centroids for {len(doc_centroids)} documents")
        logger.info(f"Found {len(all_tags)} unique tags")
        result["stats"]["total_tags"] = len(all_tags)

        # Step 3: Build NetworkX graph
        G = nx.Graph()

        # Add document nodes
        for doc in documents:
            try:
                if doc.id not in doc_centroids:
                    continue

                node_id = f"doc_{doc.id}"

                # Compute initial score (can be refined with centrality later)
                score = 1.0

                # Extract metadata
                metadata = {
                    "document_id": doc.id,
                    "path": doc.path,
                    "title": doc.title or "",
                    "tags": _extract_tags(doc.tags),
                    "modified_ts": doc.modified_ts.isoformat() if doc.modified_ts else None,
                    "indexed_ts": doc.indexed_ts.isoformat() if doc.indexed_ts else None,
                    "size_bytes": doc.size_bytes,
                }

                G.add_node(
                    node_id,
                    label=doc.title or doc.path.split("/")[-1],
                    type="document",
                    score=score,
                    metadata=metadata,
                )

            except Exception as e:
                logger.error(f"Error adding document node {doc.id}: {e}", exc_info=True)
                continue

        # Add tag nodes
        for tag in all_tags:
            try:
                node_id = f"tag_{tag}"
                G.add_node(
                    node_id,
                    label=tag,
                    type="tag",
                    score=1.0,
                    metadata={"tag_name": tag},
                )
            except Exception as e:
                logger.error(f"Error adding tag node {tag}: {e}", exc_info=True)
                continue

        logger.info(f"Added {G.number_of_nodes()} nodes to graph")

        # Step 4: Add edges between documents based on similarity
        doc_ids = list(doc_centroids.keys())
        edge_count = 0

        for i, doc_id_1 in enumerate(doc_ids):
            for doc_id_2 in doc_ids[i+1:]:
                try:
                    centroid_1 = doc_centroids[doc_id_1]
                    centroid_2 = doc_centroids[doc_id_2]

                    similarity = _cosine_similarity(centroid_1, centroid_2)

                    if similarity >= similarity_threshold:
                        G.add_edge(
                            f"doc_{doc_id_1}",
                            f"doc_{doc_id_2}",
                            weight=float(similarity),
                        )
                        edge_count += 1

                except Exception as e:
                    logger.error(f"Error computing similarity between {doc_id_1} and {doc_id_2}: {e}")
                    continue

        logger.info(f"Added {edge_count} edges based on similarity threshold {similarity_threshold}")
        result["stats"]["total_edges"] = edge_count

        # Step 5: Add edges between documents and tags
        for doc in documents:
            try:
                doc_node_id = f"doc_{doc.id}"
                if doc_node_id not in G:
                    continue

                tags = _extract_tags(doc.tags)
                for tag in tags:
                    tag_node_id = f"tag_{tag}"
                    if tag_node_id in G:
                        G.add_edge(doc_node_id, tag_node_id, weight=1.0)
                        edge_count += 1

            except Exception as e:
                logger.error(f"Error adding tag edges for document {doc.id}: {e}", exc_info=True)
                continue

        logger.info(f"Total edges after adding tag connections: {G.number_of_edges()}")
        result["stats"]["total_edges"] = G.number_of_edges()

        # Step 6: Compute graph metrics
        logger.debug("Computing graph metrics")
        metrics = _compute_graph_metrics(G)
        result["metrics"] = metrics

        # Step 7: Extract nodes with enriched metrics
        for node_id, node_data in G.nodes(data=True):
            try:
                node_dict = {
                    "id": str(node_id),
                    "label": node_data.get("label", ""),
                    "type": node_data.get("type", "unknown"),
                    "score": node_data.get("score", 1.0),
                    "metadata": node_data.get("metadata", {}),
                }

                # Enrich with computed metrics
                if str(node_id) in metrics["degrees"]:
                    node_dict["metadata"]["degree"] = metrics["degrees"][str(node_id)]

                if str(node_id) in metrics["betweenness"]:
                    node_dict["metadata"]["betweenness"] = metrics["betweenness"][str(node_id)]

                if str(node_id) in metrics["communities"]:
                    node_dict["metadata"]["community"] = metrics["communities"][str(node_id)]

                result["nodes"].append(node_dict)

            except Exception as e:
                logger.error(f"Error processing node {node_id}: {e}", exc_info=True)
                continue

        # Step 8: Extract edges
        for source, target, edge_data in G.edges(data=True):
            try:
                edge_dict = {
                    "source": str(source),
                    "target": str(target),
                    "weight": float(edge_data.get("weight", 1.0)),
                }
                result["edges"].append(edge_dict)

            except Exception as e:
                logger.error(f"Error processing edge {source}->{target}: {e}", exc_info=True)
                continue

        # Step 9: Calculate build time
        build_time = time.time() - start_time
        result["stats"]["build_time_seconds"] = round(build_time, 2)

        logger.info(
            f"Graph built successfully in {build_time:.2f}s: "
            f"{len(result['nodes'])} nodes, {len(result['edges'])} edges"
        )

        # Step 10: Update cache
        try:
            _graph_cache["data"] = result.copy()
            _graph_cache["timestamp"] = datetime.now()
            _graph_cache["threshold"] = similarity_threshold
            logger.debug("Graph cached successfully")
        except Exception as e:
            logger.warning(f"Error updating cache: {e}")

    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}", exc_info=True)
        # Return partial result rather than crashing
        result["stats"]["error"] = str(e)

    return result


def clear_graph_cache() -> None:
    """Clear the graph cache, forcing rebuild on next call.

    This is useful when documents are added/updated/deleted.
    """
    global _graph_cache

    try:
        _graph_cache["data"] = None
        _graph_cache["timestamp"] = None
        _graph_cache["threshold"] = None
        logger.info("Graph cache cleared")
    except Exception as e:
        logger.error(f"Error clearing graph cache: {e}", exc_info=True)
