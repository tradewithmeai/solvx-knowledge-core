"""RAG context assembly with token budget tracking and diversity control.

This module provides context assembly for RAG (Retrieval-Augmented Generation) by:
- Assembling relevant chunks from search results into formatted context
- Tracking token budget using tiktoken for accurate counting
- Ensuring diversity by limiting chunks per document
- Deduplicating chunks by ID
- Adding source metadata for citations
- Never crashes - comprehensive error handling throughout
- Providing detailed logging for monitoring
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from backend.app.logging import get_logger

logger = get_logger(__name__)

# Lazy import tiktoken to avoid import errors if not installed
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available, falling back to approximate token counting. "
        "Install with: pip install tiktoken"
    )


# ============================================================================
# Token Counting
# ============================================================================


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.

    Uses the cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo) by default.
    Falls back to word-based approximation if tiktoken is not available.

    Args:
        text: Input text to count tokens for
        encoding_name: Tiktoken encoding name (default: cl100k_base)

    Returns:
        int: Token count

    Raises:
        Never raises - returns approximate count on error
    """
    if not text:
        return 0

    try:
        if TIKTOKEN_AVAILABLE:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        else:
            # Fallback: approximate using word count * 1.3
            words = len(text.split())
            return int(words * 1.3)

    except Exception as e:
        logger.warning(
            f"Token counting failed, using word-based approximation: {e}",
            extra={"error": str(e), "text_length": len(text)}
        )
        # Fallback approximation
        words = len(text.split())
        return int(words * 1.3)


# ============================================================================
# Context Assembly
# ============================================================================


def assemble_context(
    search_results: List[Dict[str, Any]],
    max_tokens: int = 4000,
    max_chunks_per_doc: int = 3,
    encoding_name: str = "cl100k_base",
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Assemble context from search results with token budget and diversity control.

    This function takes search results and assembles them into a formatted context
    string suitable for RAG. It ensures diversity by limiting chunks per document,
    tracks token budget accurately, deduplicates chunks, and formats with clear
    separators.

    Args:
        search_results: List of search result dictionaries from hybrid_search.
            Each result should have:
            - chunk_id: int - Unique chunk identifier
            - text: str - Chunk text content
            - combined_score: float - Relevance score
            - source: str - Search source ('keyword', 'semantic', or 'both')
            - metadata: dict - Chunk and document metadata
        max_tokens: Maximum token budget for the assembled context (default: 4000)
        max_chunks_per_doc: Maximum chunks to include per document (default: 3)
        encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4)
        include_metadata: Whether to include source metadata in formatted chunks

    Returns:
        dict: Assembly result with:
            - success: bool - Whether assembly succeeded
            - context: str - Formatted context string ready for RAG
            - metadata: dict - Assembly statistics including:
                - total_tokens: int - Total tokens in assembled context
                - chunks_used: int - Number of chunks included
                - chunks_filtered: int - Number of chunks filtered out
                - sources_used: list - Document paths used
                - diversity_enforced: bool - Whether diversity filtering was applied
                - encoding: str - Token encoding used
            - error: str - Error message (if failed)

    Example:
        >>> from backend.app.search.hybrid import hybrid_search
        >>> from backend.app.db.session import get_db
        >>> db = next(get_db())
        >>> results = hybrid_search("machine learning", top_k=20, db=db)
        >>> context_result = assemble_context(results["results"], max_tokens=4000)
        >>> if context_result["success"]:
        ...     print(f"Context: {context_result['context']}")
        ...     print(f"Tokens used: {context_result['metadata']['total_tokens']}")
    """
    # Validate inputs
    if not search_results:
        logger.info("No search results provided for context assembly")
        return {
            "success": True,
            "context": "",
            "metadata": {
                "total_tokens": 0,
                "chunks_used": 0,
                "chunks_filtered": 0,
                "sources_used": [],
                "diversity_enforced": False,
                "encoding": encoding_name,
            }
        }

    if max_tokens <= 0:
        logger.warning(f"Invalid max_tokens: {max_tokens}, using default 4000")
        max_tokens = 4000

    if max_chunks_per_doc <= 0:
        logger.warning(f"Invalid max_chunks_per_doc: {max_chunks_per_doc}, using default 3")
        max_chunks_per_doc = 3

    try:
        logger.info(
            f"Starting context assembly",
            extra={
                "total_results": len(search_results),
                "max_tokens": max_tokens,
                "max_chunks_per_doc": max_chunks_per_doc,
            }
        )

        # Step 1: Sort by relevance score (descending)
        sorted_results = sorted(
            search_results,
            key=lambda x: x.get("combined_score", 0.0),
            reverse=True
        )

        logger.debug(
            f"Sorted {len(sorted_results)} results by relevance",
            extra={
                "top_score": sorted_results[0].get("combined_score", 0.0) if sorted_results else 0.0,
                "bottom_score": sorted_results[-1].get("combined_score", 0.0) if sorted_results else 0.0,
            }
        )

        # Step 2: Deduplication by chunk ID
        seen_chunk_ids = set()
        deduplicated_results = []

        for result in sorted_results:
            chunk_id = result.get("chunk_id")
            if chunk_id is None:
                logger.warning("Result missing chunk_id, skipping", extra={"result": result})
                continue

            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                deduplicated_results.append(result)

        duplicates_removed = len(sorted_results) - len(deduplicated_results)
        if duplicates_removed > 0:
            logger.debug(f"Removed {duplicates_removed} duplicate chunks")

        # Step 3: Enforce diversity - limit chunks per document
        doc_chunk_counts = defaultdict(int)
        diverse_results = []
        filtered_for_diversity = 0

        for result in deduplicated_results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("doc_id")

            if doc_id is None:
                logger.warning("Result missing doc_id in metadata", extra={"chunk_id": result.get("chunk_id")})
                # Include chunks without doc_id (shouldn't happen, but be resilient)
                diverse_results.append(result)
                continue

            # Check if we've reached the limit for this document
            if doc_chunk_counts[doc_id] < max_chunks_per_doc:
                doc_chunk_counts[doc_id] += 1
                diverse_results.append(result)
            else:
                filtered_for_diversity += 1

        diversity_enforced = filtered_for_diversity > 0

        logger.debug(
            f"Diversity filtering: kept {len(diverse_results)} chunks, filtered {filtered_for_diversity}",
            extra={
                "unique_documents": len(doc_chunk_counts),
                "diversity_enforced": diversity_enforced,
            }
        )

        # Step 4: Assemble context within token budget
        assembled_chunks = []
        total_tokens = 0
        sources_used = set()
        chunks_filtered_by_budget = 0

        # Reserve tokens for separators and formatting
        separator = "\n\n---\n\n"
        separator_tokens = count_tokens(separator, encoding_name)

        # Reserve some tokens for system overhead (prompt templates, etc.)
        reserved_tokens = 100
        available_tokens = max_tokens - reserved_tokens

        for i, result in enumerate(diverse_results):
            chunk_text = result.get("text", "")
            if not chunk_text:
                logger.warning(f"Empty chunk text at index {i}", extra={"chunk_id": result.get("chunk_id")})
                continue

            metadata = result.get("metadata", {})

            # Format chunk with metadata if requested
            if include_metadata:
                formatted_chunk = _format_chunk_with_metadata(result)
            else:
                formatted_chunk = chunk_text

            # Count tokens for this chunk (including separator if not first)
            chunk_tokens = count_tokens(formatted_chunk, encoding_name)
            if i > 0:
                chunk_tokens += separator_tokens

            # Check if we have budget for this chunk
            if total_tokens + chunk_tokens <= available_tokens:
                assembled_chunks.append(formatted_chunk)
                total_tokens += chunk_tokens

                # Track source document
                doc_path = metadata.get("path")
                if doc_path:
                    sources_used.add(doc_path)

                logger.debug(
                    f"Added chunk {i+1}: {chunk_tokens} tokens (total: {total_tokens}/{available_tokens})",
                    extra={
                        "chunk_id": result.get("chunk_id"),
                        "score": result.get("combined_score"),
                        "doc_path": doc_path,
                    }
                )
            else:
                # Budget exceeded
                chunks_filtered_by_budget += 1
                logger.debug(
                    f"Skipping chunk {i+1}: would exceed budget ({total_tokens + chunk_tokens} > {available_tokens})",
                    extra={
                        "chunk_id": result.get("chunk_id"),
                        "chunk_tokens": chunk_tokens,
                        "remaining_budget": available_tokens - total_tokens,
                    }
                )
                # Don't process more chunks if we've hit the budget
                # Continue to count how many we filtered
                continue

        # Step 5: Join chunks with separators
        context = separator.join(assembled_chunks)

        # Add final token count (should be close to our running total)
        final_token_count = count_tokens(context, encoding_name)

        logger.info(
            f"Context assembly completed successfully",
            extra={
                "chunks_used": len(assembled_chunks),
                "chunks_filtered_duplicates": duplicates_removed,
                "chunks_filtered_diversity": filtered_for_diversity,
                "chunks_filtered_budget": chunks_filtered_by_budget,
                "total_tokens": final_token_count,
                "token_budget": max_tokens,
                "utilization": f"{(final_token_count / max_tokens * 100):.1f}%",
                "sources_used": len(sources_used),
                "diversity_enforced": diversity_enforced,
            }
        )

        return {
            "success": True,
            "context": context,
            "metadata": {
                "total_tokens": final_token_count,
                "max_tokens": max_tokens,
                "chunks_used": len(assembled_chunks),
                "chunks_filtered": (
                    duplicates_removed + filtered_for_diversity + chunks_filtered_by_budget
                ),
                "chunks_filtered_duplicates": duplicates_removed,
                "chunks_filtered_diversity": filtered_for_diversity,
                "chunks_filtered_budget": chunks_filtered_by_budget,
                "sources_used": sorted(list(sources_used)),
                "unique_documents": len(sources_used),
                "diversity_enforced": diversity_enforced,
                "encoding": encoding_name,
                "tiktoken_available": TIKTOKEN_AVAILABLE,
            }
        }

    except Exception as e:
        # Never crash - catch all errors
        logger.error(
            f"Context assembly failed: {e}",
            exc_info=True,
            extra={
                "error": str(e),
                "total_results": len(search_results) if search_results else 0,
                "max_tokens": max_tokens,
            }
        )

        return {
            "success": False,
            "context": "",
            "metadata": {
                "total_tokens": 0,
                "chunks_used": 0,
                "chunks_filtered": 0,
                "sources_used": [],
                "diversity_enforced": False,
                "encoding": encoding_name,
            },
            "error": f"Context assembly failed: {str(e)}"
        }


def _format_chunk_with_metadata(result: Dict[str, Any]) -> str:
    """Format a chunk with source metadata for citations.

    Args:
        result: Search result dictionary

    Returns:
        str: Formatted chunk with metadata header
    """
    try:
        chunk_text = result.get("text", "")
        metadata = result.get("metadata", {})

        # Extract key metadata
        doc_path = metadata.get("path", "Unknown")
        doc_title = metadata.get("title") or doc_path.split("/")[-1] if doc_path != "Unknown" else "Unknown"
        page = metadata.get("page")
        section = metadata.get("section")
        score = result.get("combined_score", 0.0)

        # Build metadata header
        header_parts = [f"Source: {doc_title}"]

        if page is not None:
            header_parts.append(f"Page {page}")

        if section:
            header_parts.append(f"Section: {section}")

        header_parts.append(f"Relevance: {score:.2f}")

        header = " | ".join(header_parts)

        # Format: header + newline + content
        formatted = f"[{header}]\n{chunk_text}"

        return formatted

    except Exception as e:
        logger.warning(
            f"Failed to format chunk with metadata: {e}",
            extra={"error": str(e), "chunk_id": result.get("chunk_id")}
        )
        # Fallback: return just the text
        return result.get("text", "")


# ============================================================================
# Utility Functions
# ============================================================================


def estimate_context_size(
    search_results: List[Dict[str, Any]],
    encoding_name: str = "cl100k_base"
) -> Dict[str, Any]:
    """Estimate the size of context that would be assembled from search results.

    This is useful for determining appropriate max_tokens before assembly.

    Args:
        search_results: List of search result dictionaries
        encoding_name: Tiktoken encoding name

    Returns:
        dict: Size estimates with:
            - total_tokens: Total tokens if all results included
            - avg_tokens_per_chunk: Average tokens per chunk
            - min_tokens_per_chunk: Minimum tokens in a chunk
            - max_tokens_per_chunk: Maximum tokens in a chunk
            - total_chunks: Total number of chunks
    """
    if not search_results:
        return {
            "total_tokens": 0,
            "avg_tokens_per_chunk": 0,
            "min_tokens_per_chunk": 0,
            "max_tokens_per_chunk": 0,
            "total_chunks": 0,
        }

    try:
        token_counts = []

        for result in search_results:
            chunk_text = result.get("text", "")
            if chunk_text:
                tokens = count_tokens(chunk_text, encoding_name)
                token_counts.append(tokens)

        if not token_counts:
            return {
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens_per_chunk": 0,
                "max_tokens_per_chunk": 0,
                "total_chunks": 0,
            }

        return {
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens_per_chunk": min(token_counts),
            "max_tokens_per_chunk": max(token_counts),
            "total_chunks": len(token_counts),
        }

    except Exception as e:
        logger.error(
            f"Failed to estimate context size: {e}",
            exc_info=True,
            extra={"error": str(e)}
        )
        return {
            "total_tokens": 0,
            "avg_tokens_per_chunk": 0,
            "min_tokens_per_chunk": 0,
            "max_tokens_per_chunk": 0,
            "total_chunks": 0,
            "error": str(e),
        }


def get_available_encodings() -> List[str]:
    """Get list of available tiktoken encodings.

    Returns:
        list: Available encoding names, or empty list if tiktoken unavailable
    """
    if not TIKTOKEN_AVAILABLE:
        return []

    try:
        # Common tiktoken encodings
        return [
            "cl100k_base",  # GPT-4, GPT-3.5-turbo, text-embedding-ada-002
            "p50k_base",    # Codex models, text-davinci-002, text-davinci-003
            "r50k_base",    # GPT-3 models like davinci
            "gpt2",         # GPT-2
        ]
    except Exception as e:
        logger.error(f"Failed to get encodings: {e}")
        return []
