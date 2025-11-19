"""Chat API route with SSE streaming for RAG-powered conversations.

This module provides a streaming chat endpoint that:
- Performs hybrid search to retrieve relevant context
- Assembles context from search results
- Streams responses from Claude using Server-Sent Events (SSE)
- Returns citations in solvx:// format
- Handles errors gracefully with comprehensive logging
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from anthropic import Anthropic, APIError
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.logging import get_logger
from backend.app.rag.prompts import format_user_prompt, get_system_prompt
from backend.app.search.hybrid import hybrid_search

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatRequest(BaseModel):
    """Chat request with query and optional search parameters."""

    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of context chunks")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Search filters (path, doc_id, date_range)"
    )


# ============================================================================
# Context Assembly
# ============================================================================


def assemble_context(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assemble context chunks from search results for RAG.

    Converts search results into a format suitable for the RAG prompt,
    extracting the necessary fields and formatting citations.

    Args:
        search_results: List of search result dictionaries from hybrid_search

    Returns:
        List of context dictionaries with:
            - content: str - The chunk text
            - source: str - The document path/filename
            - page: int (optional) - The page number if available
            - citation: str - The solvx:// citation URI
            - score: float - The relevance score

    Example:
        >>> results = hybrid_search("machine learning", top_k=5, db=db)
        >>> context = assemble_context(results["results"])
        >>> print(context[0]["content"][:50])
    """
    try:
        context_chunks = []

        for result in search_results:
            # Extract metadata
            metadata = result.get("metadata", {})
            doc_id = metadata.get("doc_id", 0)
            path = metadata.get("path", "unknown")
            seq = metadata.get("seq", 0)
            page = metadata.get("page")

            # Extract filename from path for cleaner citations
            filename = path.split("/")[-1] if "/" in path else path

            # Build solvx:// citation URI
            citation = f"solvx://doc/{doc_id}#chunk={seq}"
            if page is not None:
                citation += f"&page={page}"

            # Assemble context chunk
            chunk = {
                "content": result.get("text", ""),
                "source": filename,
                "citation": citation,
                "score": result.get("combined_score", 0.0),
            }

            # Add page if available
            if page is not None:
                chunk["page"] = page

            context_chunks.append(chunk)

        logger.debug(
            f"Assembled {len(context_chunks)} context chunks",
            extra={"chunk_count": len(context_chunks)},
        )

        return context_chunks

    except Exception as e:
        logger.error(
            f"Failed to assemble context: {e}",
            exc_info=True,
            extra={"error": str(e), "result_count": len(search_results)},
        )
        # Return empty context on error - don't crash
        return []


# ============================================================================
# Claude Streaming Client
# ============================================================================


async def chat_stream(
    query: str, context_chunks: List[Dict[str, Any]]
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream chat responses from Claude using the Anthropic API.

    Creates a streaming chat completion using Claude, with the system prompt
    defining RAG behavior and the user prompt containing the query and context.

    Args:
        query: The user's question/query
        context_chunks: List of context chunks from assemble_context

    Yields:
        Dict[str, Any]: Stream events with:
            - type: "chunk" - Text chunk from Claude
                - content: str - The text content
            - type: "done" - Stream complete
                - sources: List[Dict] - Source citations used
            - type: "error" - Error occurred
                - message: str - Error description

    Raises:
        HTTPException: If Claude API key is not configured

    Example:
        >>> async for event in chat_stream("What is ML?", context):
        ...     if event["type"] == "chunk":
        ...         print(event["content"], end="")
        ...     elif event["type"] == "done":
        ...         print(f"\\n\\nSources: {event['sources']}")
    """
    # Validate API key
    if not settings.claude_api_key:
        logger.error("Claude API key not configured")
        raise HTTPException(
            status_code=500,
            detail="Claude API key not configured. Please set CLAUDE_API_KEY in .env file.",
        )

    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=settings.claude_api_key)

        # Format prompts
        system_prompt = get_system_prompt()
        user_prompt = format_user_prompt(query, context_chunks)

        logger.info(
            "Starting Claude stream",
            extra={
                "query_length": len(query),
                "context_chunks": len(context_chunks),
                "prompt_length": len(user_prompt),
            },
        )

        # Stream from Claude
        stream_started = False
        total_tokens = 0

        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
            max_tokens=4096,
            temperature=0.0,  # Deterministic for factual responses
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            stream_started = True

            for text in stream.text_stream:
                if text:
                    total_tokens += len(text.split())
                    yield {
                        "type": "chunk",
                        "content": text,
                    }

        # Stream complete - send sources
        logger.info(
            "Claude stream completed",
            extra={
                "total_tokens": total_tokens,
                "sources_count": len(context_chunks),
            },
        )

        # Prepare source citations
        sources = [
            {
                "citation": chunk["citation"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "score": chunk["score"],
            }
            for chunk in context_chunks
        ]

        yield {
            "type": "done",
            "sources": sources,
        }

    except APIError as e:
        error_msg = f"Claude API error: {str(e)}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "error_type": type(e).__name__,
                "error": str(e),
                "query": query[:100],
            },
        )
        yield {
            "type": "error",
            "message": error_msg,
        }

    except Exception as e:
        error_msg = f"Unexpected error during streaming: {str(e)}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "error_type": type(e).__name__,
                "error": str(e),
                "query": query[:100],
                "stream_started": stream_started,
            },
        )
        yield {
            "type": "error",
            "message": error_msg,
        }


# ============================================================================
# SSE Event Formatter
# ============================================================================


async def format_sse_events(
    query: str, context_chunks: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """Format chat stream events as Server-Sent Events (SSE).

    Wraps the chat_stream generator to format events according to SSE protocol.
    Each event is sent as "data: {json}\\n\\n".

    Args:
        query: The user's question/query
        context_chunks: List of context chunks from assemble_context

    Yields:
        str: SSE-formatted event strings

    Example SSE events:
        data: {"type": "chunk", "content": "Machine learning is"}
        data: {"type": "chunk", "content": " a subset of AI"}
        data: {"type": "done", "sources": [...]}
    """
    try:
        async for event in chat_stream(query, context_chunks):
            # Format as SSE event
            event_data = json.dumps(event, ensure_ascii=False)
            yield f"data: {event_data}\n\n"

    except Exception as e:
        # Send error event if something goes wrong
        error_event = {
            "type": "error",
            "message": f"Stream error: {str(e)}",
        }
        error_data = json.dumps(error_event, ensure_ascii=False)
        yield f"data: {error_data}\n\n"


# ============================================================================
# Chat Endpoint
# ============================================================================


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """SSE streaming chat endpoint powered by RAG and Claude.

    This endpoint:
    1. Performs hybrid search to retrieve relevant context
    2. Assembles context from top results
    3. Streams response from Claude using the context
    4. Returns citations for all sources used

    Request Body:
        - query: str - User's question (1-5000 characters, required)
        - top_k: int - Number of context chunks to retrieve (1-50, default: 10)
        - filters: dict - Optional search filters:
            - path: str - Filter by document path
            - doc_id: int - Filter by document ID
            - date_range: dict with 'start' and/or 'end' ISO datetime strings

    Response:
        Server-Sent Events (SSE) stream with:
        - Chunk events: {"type": "chunk", "content": "..."}
        - Done event: {"type": "done", "sources": [...]}
        - Error events: {"type": "error", "message": "..."}

    Sources format:
        {
            "citation": "solvx://doc/123#chunk=5&page=12",
            "source": "document.pdf",
            "page": 12,  // optional
            "score": 0.85
        }

    Error Responses:
        400: Invalid request (empty query, invalid filters)
        500: Server error (search failed, streaming failed, API key not configured)

    Example:
        POST /api/chat
        {
            "query": "What is machine learning?",
            "top_k": 10,
            "filters": {"doc_id": 5}
        }

        Response (SSE stream):
        data: {"type": "chunk", "content": "Machine learning is"}
        data: {"type": "chunk", "content": " a subset of artificial"}
        data: {"type": "chunk", "content": " intelligence [source: ml-guide.pdf p.5]"}
        data: {"type": "done", "sources": [{"citation": "solvx://doc/5#chunk=2&page=5", ...}]}
    """
    try:
        # Validate query
        if not request.query or not request.query.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        query = request.query.strip()

        logger.info(
            "Chat request received",
            extra={
                "query_length": len(query),
                "top_k": request.top_k,
                "has_filters": bool(request.filters),
            },
        )

        # Perform hybrid search to get relevant context
        logger.debug("Performing hybrid search", extra={"query": query[:100]})

        search_result = hybrid_search(
            query=query,
            top_k=request.top_k,
            filters=request.filters,
            db=db,
        )

        if not search_result.get("success"):
            error_msg = search_result.get("error", "Search failed")
            logger.error(
                f"Hybrid search failed: {error_msg}",
                extra={"query": query[:100], "error": error_msg},
            )
            raise HTTPException(status_code=500, detail=f"Search failed: {error_msg}")

        results = search_result.get("results", [])

        logger.info(
            "Search completed",
            extra={
                "result_count": len(results),
                "query": query[:100],
                "duration_ms": search_result.get("metadata", {}).get("duration_ms", 0),
            },
        )

        # Check if we have any results
        if not results:
            logger.warning(
                "No search results found for query",
                extra={"query": query[:100]},
            )
            # Continue with empty context - Claude will indicate no information found

        # Assemble context chunks
        context_chunks = assemble_context(results)

        logger.info(
            "Context assembled, starting stream",
            extra={
                "context_chunks": len(context_chunks),
                "query": query[:100],
            },
        )

        # Return SSE streaming response
        return StreamingResponse(
            format_sse_events(query, context_chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(
            "Unexpected error in chat endpoint",
            exc_info=True,
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "query": request.query[:100] if request.query else None,
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {str(e)}",
        )
