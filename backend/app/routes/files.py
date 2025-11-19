"""File management API routes for watch paths, scanning, and document management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.logging import get_logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.task import Task

# Scanner functions would be imported from scanner.py when it exists
# from backend.app.scanner import (
#     scan_directory,
#     scan_path,
#     add_watch_path,
#     remove_watch_path,
#     get_watch_paths,
# )

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["files"])


# Pydantic models for request/response
class AddPathRequest(BaseModel):
    """Request model for adding a watch path."""

    path: str = Field(..., description="Absolute path to directory to watch")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that path is absolute and exists."""
        path_obj = Path(v)
        if not path_obj.is_absolute():
            raise ValueError("Path must be absolute")
        return v


class PathResponse(BaseModel):
    """Response model for a watch path."""

    id: int
    path: str
    enabled: bool
    last_scan: str | None
    created_at: str
    updated_at: str


class ScanResponse(BaseModel):
    """Response model for scan operations."""

    task_id: int
    status: str
    message: str


class DocumentFilter(BaseModel):
    """Filters for document listing."""

    status: str | None = Field(None, description="Filter by document status")
    source_type: str | None = Field(None, description="Filter by source type")
    search: str | None = Field(None, description="Search in path or title")


class DocumentResponse(BaseModel):
    """Response model for a document."""

    id: int
    path: str
    source_type: str
    hash: str
    title: str | None
    status: str
    size_bytes: int
    modified_ts: str | None
    indexed_ts: str | None
    created_at: str
    chunk_count: int | None = None


class DocumentListResponse(BaseModel):
    """Response model for document listing with pagination."""

    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# Watch Path Endpoints


@router.post("/paths", response_model=PathResponse, status_code=status.HTTP_201_CREATED)
def add_watch_path(
    request: AddPathRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Add a new watch path for file monitoring and scanning.

    Args:
        request: Path information
        db: Database session

    Returns:
        Created watch path information

    Raises:
        HTTPException: If path is invalid, already exists, or other errors occur
    """
    logger.info(f"Adding watch path: {request.path}")

    try:
        # Validate path exists
        path_obj = Path(request.path)
        if not path_obj.exists():
            logger.error(f"Path does not exist: {request.path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path does not exist: {request.path}",
            )

        if not path_obj.is_dir():
            logger.error(f"Path is not a directory: {request.path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {request.path}",
            )

        # TODO: Implement WatchPath model and add_watch_path function
        # For now, create a placeholder response
        # watch_path = add_watch_path(db, str(path_obj.resolve()))

        logger.warning("WatchPath model not yet implemented - returning mock response")
        return {
            "id": 1,
            "path": str(path_obj.resolve()),
            "enabled": True,
            "last_scan": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add watch path: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add watch path: {str(e)}",
        )


@router.delete("/paths/{path_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_watch_path(
    path_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    Remove a watch path.

    Args:
        path_id: ID of the watch path to remove
        db: Database session

    Raises:
        HTTPException: If path not found or removal fails
    """
    logger.info(f"Removing watch path with ID: {path_id}")

    try:
        # TODO: Implement WatchPath model and remove_watch_path function
        # watch_path = db.query(WatchPath).filter(WatchPath.id == path_id).first()
        # if not watch_path:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail=f"Watch path with ID {path_id} not found"
        #     )
        # remove_watch_path(db, path_id)

        logger.warning("WatchPath model not yet implemented - operation skipped")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove watch path {path_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove watch path: {str(e)}",
        )


@router.get("/paths", response_model=list[PathResponse])
def list_watch_paths(
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """
    List all watch paths.

    Args:
        db: Database session

    Returns:
        List of watch paths with their details
    """
    logger.info("Listing all watch paths")

    try:
        # TODO: Implement WatchPath model and get_watch_paths function
        # watch_paths = get_watch_paths(db)
        # return [
        #     {
        #         "id": wp.id,
        #         "path": wp.path,
        #         "enabled": wp.enabled,
        #         "last_scan": wp.last_scan.isoformat() if wp.last_scan else None,
        #         "created_at": wp.created_at.isoformat(),
        #         "updated_at": wp.updated_at.isoformat(),
        #     }
        #     for wp in watch_paths
        # ]

        logger.warning("WatchPath model not yet implemented - returning empty list")
        return []

    except Exception as e:
        logger.error(f"Failed to list watch paths: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list watch paths: {str(e)}",
        )


# Scan Endpoints


@router.post("/scan", response_model=ScanResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_full_scan(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Trigger a full scan of all watch paths.

    Creates a background task to scan all enabled watch paths and ingest
    new or modified documents.

    Args:
        db: Database session

    Returns:
        Task information for tracking scan progress
    """
    logger.info("Triggering full scan of all watch paths")

    try:
        # Create a task for tracking
        task = Task(
            name="full_scan",
            status="pending",
            details=json.dumps({"scan_type": "full", "started_at": datetime.utcnow().isoformat()}),
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        logger.info(f"Created scan task with ID: {task.id}")

        # TODO: Implement scan_directory function and trigger background scan
        # This would typically use a background task queue (Celery, RQ, etc.)
        # For now, we'll just create the task record
        # schedule_scan_task(task.id, "full")

        return {
            "task_id": task.id,
            "status": "pending",
            "message": "Full scan task created and queued for processing",
        }

    except Exception as e:
        logger.error(f"Failed to trigger full scan: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger scan: {str(e)}",
        )


@router.post("/scan/{path_id}", response_model=ScanResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_path_scan(
    path_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Trigger a scan of a specific watch path.

    Creates a background task to scan the specified watch path and ingest
    new or modified documents.

    Args:
        path_id: ID of the watch path to scan
        db: Database session

    Returns:
        Task information for tracking scan progress

    Raises:
        HTTPException: If path not found or scan fails to start
    """
    logger.info(f"Triggering scan for watch path ID: {path_id}")

    try:
        # TODO: Verify watch path exists
        # watch_path = db.query(WatchPath).filter(WatchPath.id == path_id).first()
        # if not watch_path:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail=f"Watch path with ID {path_id} not found"
        #     )

        # Create a task for tracking
        task = Task(
            name="path_scan",
            status="pending",
            details=json.dumps({
                "scan_type": "path",
                "path_id": path_id,
                "started_at": datetime.utcnow().isoformat(),
            }),
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        logger.info(f"Created scan task with ID: {task.id} for path ID: {path_id}")

        # TODO: Implement scan_path function and trigger background scan
        # schedule_scan_task(task.id, "path", path_id=path_id)

        return {
            "task_id": task.id,
            "status": "pending",
            "message": f"Path scan task created for watch path {path_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger path scan {path_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger scan: {str(e)}",
        )


# Document Management Endpoints


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=500, description="Number of items per page"),
    status: str | None = Query(None, description="Filter by status"),
    source_type: str | None = Query(None, description="Filter by source type"),
    search: str | None = Query(None, description="Search in path or title"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    List documents with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        status: Filter by document status (new, parsed, embedded, error, deleted)
        source_type: Filter by source type (pdf, txt, md, etc.)
        search: Search term for path or title
        db: Database session

    Returns:
        Paginated list of documents with metadata
    """
    logger.info(f"Listing documents: page={page}, page_size={page_size}, status={status}, search={search}")

    try:
        # Build query
        query = db.query(Document)

        # Apply filters
        if status:
            query = query.filter(Document.status == status)

        if source_type:
            query = query.filter(Document.source_type == source_type)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Document.path.ilike(search_term)) | (Document.title.ilike(search_term))
            )

        # Get total count
        total = query.count()

        # Calculate pagination
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size

        # Get documents with chunk counts
        documents = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size).all()

        # Build response with chunk counts
        doc_responses = []
        for doc in documents:
            chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc.id).scalar()
            doc_responses.append({
                "id": doc.id,
                "path": doc.path,
                "source_type": doc.source_type,
                "hash": doc.hash,
                "title": doc.title,
                "status": doc.status,
                "size_bytes": doc.size_bytes,
                "modified_ts": doc.modified_ts.isoformat() if doc.modified_ts else None,
                "indexed_ts": doc.indexed_ts.isoformat() if doc.indexed_ts else None,
                "created_at": doc.created_at.isoformat(),
                "chunk_count": chunk_count or 0,
            })

        logger.info(f"Found {total} documents, returning page {page}/{total_pages}")

        return {
            "documents": doc_responses,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a document and all its associated chunks.

    This will remove the document from the database and delete all associated
    chunks. Vector embeddings in the vector store should be cleaned up separately.

    Args:
        doc_id: ID of the document to delete
        db: Database session

    Raises:
        HTTPException: If document not found or deletion fails
    """
    logger.info(f"Deleting document with ID: {doc_id}")

    try:
        # Find document
        document = db.query(Document).filter(Document.id == doc_id).first()

        if not document:
            logger.warning(f"Document {doc_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found",
            )

        # Get chunk count before deletion
        chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc_id).scalar()

        logger.info(f"Deleting document {doc_id} ({document.path}) with {chunk_count} chunks")

        # Delete chunks (CASCADE should handle this, but explicit is better)
        db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

        # Delete document
        db.delete(document)
        db.commit()

        logger.info(f"Successfully deleted document {doc_id} and {chunk_count} chunks")

        # TODO: Also delete from vector store
        # This would require integration with the vectorstore module
        # delete_from_vectorstore(doc_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(
    doc_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Get detailed information about a specific document.

    Args:
        doc_id: ID of the document
        db: Database session

    Returns:
        Document details including chunk count

    Raises:
        HTTPException: If document not found
    """
    logger.info(f"Fetching document with ID: {doc_id}")

    try:
        document = db.query(Document).filter(Document.id == doc_id).first()

        if not document:
            logger.warning(f"Document {doc_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found",
            )

        chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc_id).scalar()

        return {
            "id": document.id,
            "path": document.path,
            "source_type": document.source_type,
            "hash": document.hash,
            "title": document.title,
            "status": document.status,
            "size_bytes": document.size_bytes,
            "modified_ts": document.modified_ts.isoformat() if document.modified_ts else None,
            "indexed_ts": document.indexed_ts.isoformat() if document.indexed_ts else None,
            "created_at": document.created_at.isoformat(),
            "chunk_count": chunk_count or 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}",
        )
