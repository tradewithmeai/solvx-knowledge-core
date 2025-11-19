"""Admin routes for health checks, tasks, and system management."""

import sys
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import __version__
from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.task import Task

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Health check endpoint with system information and statistics.

    Returns:
        System health information including version, counts, and status.
    """
    # Get database counts
    doc_count = db.query(func.count(Document.id)).scalar() or 0
    chunk_count = db.query(func.count(Chunk.id)).scalar() or 0
    task_count = db.query(func.count(Task.id)).scalar() or 0

    # Get recent task statuses
    recent_tasks = (
        db.query(Task)
        .order_by(Task.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "status": "ok",
        "app": {
            "name": settings.app_name,
            "version": __version__,
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "timestamp": datetime.utcnow().isoformat(),
        },
        "database": {
            "documents": doc_count,
            "chunks": chunk_count,
            "tasks": task_count,
        },
        "config": {
            "embed_provider": settings.embed_provider,
            "enable_rerank": settings.enable_rerank,
            "enable_ocr": settings.enable_ocr,
        },
        "recent_tasks": [
            {
                "name": task.name,
                "status": task.status,
                "created_at": task.created_at.isoformat(),
            }
            for task in recent_tasks
        ],
    }


@router.get("/tasks")
def list_tasks(
    limit: int = 20,
    status: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """
    List background tasks with optional filtering.

    Args:
        limit: Maximum number of tasks to return
        status: Filter by task status (pending, running, completed, failed)
        db: Database session

    Returns:
        List of tasks with details.
    """
    query = db.query(Task).order_by(Task.created_at.desc())

    if status:
        query = query.filter(Task.status == status)

    tasks = query.limit(limit).all()

    return {
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "status": task.status,
                "started": task.started.isoformat() if task.started else None,
                "finished": task.finished.isoformat() if task.finished else None,
                "created_at": task.created_at.isoformat(),
                "details": task.details,
            }
            for task in tasks
        ],
        "total": len(tasks),
    }
