"""Insights API routes for weekly digests and reflection prompts.

This module provides endpoints for generating and retrieving:
- Weekly knowledge digests from the past week's documents
- Reflection prompts for personal learning
- Cached insight data with markdown content and metadata

Features:
- Hybrid search to identify key topics
- Claude-powered digest and reflection generation
- Database persistence with caching
- Error handling and comprehensive logging
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.insights import weekly, reflection
from backend.app.logging import get_logger
from backend.app.models.insight import Insight

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["insights"])


# ============================================================================
# Request/Response Models
# ============================================================================


class WeeklyDigestResponse(BaseModel):
    """Response model for weekly digest."""

    id: int = Field(..., description="Unique digest ID")
    type: str = Field(..., description="Insight type (weekly)")
    period_start: datetime = Field(..., description="Week start date")
    period_end: datetime = Field(..., description="Week end date")
    markdown: str = Field(..., description="Digest content in markdown format")
    created_at: datetime = Field(..., description="When digest was created")
    updated_at: datetime = Field(..., description="When digest was last updated")

    class Config:
        from_attributes = True


class ReflectionPromptResponse(BaseModel):
    """Response model for reflection prompt."""

    id: int = Field(..., description="Unique insight ID")
    type: str = Field(..., description="Insight type (reflection)")
    markdown: str = Field(..., description="Reflection content in markdown format")
    created_at: datetime = Field(..., description="When reflection was created")
    updated_at: datetime = Field(..., description="When reflection was last updated")

    class Config:
        from_attributes = True


class GenerateWeeklyRequest(BaseModel):
    """Request model for triggering weekly digest generation."""

    period_start: Optional[datetime] = Field(
        default=None,
        description="Optional start date (defaults to Monday of current week)",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="Optional end date (defaults to Sunday of current week)",
    )


class GenerateReflectionRequest(BaseModel):
    """Request model for triggering reflection generation."""

    focus_area: Optional[str] = Field(
        default=None,
        description="Optional focus area for reflection (e.g., 'machine learning', 'productivity')",
    )


class GenerateResponse(BaseModel):
    """Response model for generation endpoints."""

    success: bool = Field(..., description="Whether generation succeeded")
    message: str = Field(..., description="Status message")
    insight_id: Optional[int] = Field(None, description="ID of generated insight")


class InsightListResponse(BaseModel):
    """Response model for listing insights."""

    success: bool = Field(..., description="Whether request succeeded")
    count: int = Field(..., description="Number of insights returned")
    insights: list[ReflectionPromptResponse] = Field(
        ..., description="List of insights"
    )


# ============================================================================
# Weekly Digest Endpoints
# ============================================================================


@router.get("/insights/weekly/latest", response_model=WeeklyDigestResponse)
def get_latest_weekly_digest(db: Session = Depends(get_db)):
    """Retrieve the latest weekly digest.

    Returns the most recently generated weekly digest, including the
    week's key insights, notable documents, and learning summary.

    Returns:
        WeeklyDigestResponse: The latest weekly digest with markdown content

    Raises:
        HTTPException: 404 if no digest exists
        HTTPException: 500 on database errors

    Example:
        GET /api/insights/weekly/latest
        Response:
            {
                "id": 42,
                "type": "weekly",
                "period_start": "2024-11-11T00:00:00",
                "period_end": "2024-11-17T23:59:59",
                "markdown": "# Weekly Digest...",
                "created_at": "2024-11-18T10:30:00",
                "updated_at": "2024-11-18T10:30:00"
            }
    """
    try:
        logger.info("Fetching latest weekly digest")

        # Query latest weekly insight
        latest_digest = (
            db.query(Insight)
            .filter(Insight.type == "weekly")
            .order_by(desc(Insight.period_end))
            .first()
        )

        if not latest_digest:
            logger.warning("No weekly digest found in database")
            raise HTTPException(
                status_code=404,
                detail="No weekly digest has been generated yet",
            )

        logger.info(
            "Latest weekly digest retrieved",
            extra={
                "insight_id": latest_digest.id,
                "period_start": latest_digest.period_start.isoformat()
                if latest_digest.period_start
                else None,
                "period_end": latest_digest.period_end.isoformat()
                if latest_digest.period_end
                else None,
            },
        )

        return latest_digest

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Failed to retrieve latest weekly digest",
            exc_info=True,
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve weekly digest",
        )


@router.get("/insights/weekly/{week_id}", response_model=WeeklyDigestResponse)
def get_weekly_digest(week_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific week's digest by ID.

    Fetches a previously generated weekly digest from the database
    using its unique identifier.

    Path Parameters:
        week_id: Unique digest ID

    Returns:
        WeeklyDigestResponse: The requested weekly digest

    Raises:
        HTTPException: 404 if digest not found
        HTTPException: 500 on database errors

    Example:
        GET /api/insights/weekly/42
        Response:
            {
                "id": 42,
                "type": "weekly",
                "period_start": "2024-11-11T00:00:00",
                "period_end": "2024-11-17T23:59:59",
                "markdown": "# Weekly Digest...",
                "created_at": "2024-11-18T10:30:00",
                "updated_at": "2024-11-18T10:30:00"
            }
    """
    try:
        logger.info(
            "Fetching weekly digest by ID",
            extra={"week_id": week_id},
        )

        # Query digest by ID
        digest = db.query(Insight).filter(Insight.id == week_id).first()

        if not digest:
            logger.warning(
                "Weekly digest not found",
                extra={"week_id": week_id},
            )
            raise HTTPException(
                status_code=404,
                detail=f"Weekly digest with ID {week_id} not found",
            )

        if digest.type != "weekly":
            logger.warning(
                "Insight is not a weekly digest",
                extra={"week_id": week_id, "type": digest.type},
            )
            raise HTTPException(
                status_code=404,
                detail=f"Insight {week_id} is not a weekly digest",
            )

        logger.info(
            "Weekly digest retrieved",
            extra={
                "week_id": week_id,
                "period_start": digest.period_start.isoformat()
                if digest.period_start
                else None,
            },
        )

        return digest

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Failed to retrieve weekly digest",
            exc_info=True,
            extra={"error": str(e), "week_id": week_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve weekly digest",
        )


@router.post("/insights/weekly/generate", response_model=GenerateResponse)
def generate_weekly_digest(
    request: GenerateWeeklyRequest, db: Session = Depends(get_db)
):
    """Trigger generation of a weekly digest.

    Analyzes documents from the specified week (or current week if not specified)
    and generates a markdown digest with key insights, notable topics, and
    learning summary. The digest is stored in the database.

    Request Body:
        period_start: Optional ISO datetime for week start
        period_end: Optional ISO datetime for week end

    Returns:
        GenerateResponse: Generation status and insight ID

    Raises:
        HTTPException: 400 on invalid date parameters
        HTTPException: 500 on generation failure

    Example:
        POST /api/insights/weekly/generate
        {
            "period_start": "2024-11-11T00:00:00",
            "period_end": "2024-11-17T23:59:59"
        }

        Response:
        {
            "success": true,
            "message": "Weekly digest generated successfully",
            "insight_id": 43
        }
    """
    try:
        logger.info("Starting weekly digest generation")

        # Determine period
        if request.period_start and request.period_end:
            period_start = request.period_start
            period_end = request.period_end

            # Validate dates
            if period_start >= period_end:
                logger.warning(
                    "Invalid date range provided",
                    extra={
                        "period_start": period_start.isoformat(),
                        "period_end": period_end.isoformat(),
                    },
                )
                raise HTTPException(
                    status_code=400,
                    detail="period_start must be before period_end",
                )
        else:
            # Use current week (Monday-Sunday)
            today = datetime.utcnow().date()
            monday = today - timedelta(days=today.weekday())
            sunday = monday + timedelta(days=6)

            period_start = datetime.combine(monday, datetime.min.time())
            period_end = datetime.combine(sunday, datetime.max.time())

            logger.debug(
                "Using current week for digest",
                extra={
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
            )

        # Check if digest already exists for this period
        existing = (
            db.query(Insight)
            .filter(
                Insight.type == "weekly",
                Insight.period_start == period_start,
                Insight.period_end == period_end,
            )
            .first()
        )

        if existing:
            logger.info(
                "Weekly digest already exists for this period",
                extra={
                    "insight_id": existing.id,
                    "period_start": period_start.isoformat(),
                },
            )
            return GenerateResponse(
                success=True,
                message="Weekly digest already exists for this period",
                insight_id=existing.id,
            )

        # Generate digest
        logger.info(
            "Generating weekly digest",
            extra={
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
            },
        )

        markdown_content = weekly.generate_digest(
            period_start=period_start,
            period_end=period_end,
            db=db,
        )

        if not markdown_content:
            logger.warning(
                "No markdown content generated for weekly digest",
                extra={"period_start": period_start.isoformat()},
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to generate digest content",
            )

        # Store in database
        insight = Insight(
            type="weekly",
            period_start=period_start,
            period_end=period_end,
            markdown=markdown_content,
        )
        db.add(insight)
        db.commit()
        db.refresh(insight)

        logger.info(
            "Weekly digest generated and stored",
            extra={
                "insight_id": insight.id,
                "period_start": period_start.isoformat(),
                "markdown_length": len(markdown_content),
            },
        )

        return GenerateResponse(
            success=True,
            message="Weekly digest generated successfully",
            insight_id=insight.id,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Failed to generate weekly digest",
            exc_info=True,
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "period_start": request.period_start.isoformat()
                if request.period_start
                else None,
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate weekly digest: {str(e)}",
        )


# ============================================================================
# Reflection Endpoints
# ============================================================================


@router.get(
    "/insights/reflections",
    response_model=InsightListResponse,
)
def list_reflection_prompts(
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    db: Session = Depends(get_db),
):
    """List all reflection prompts.

    Retrieves a list of previously generated reflection prompts,
    ordered by most recent first.

    Query Parameters:
        limit: Maximum number of reflections to return (1-100, default: 10)

    Returns:
        InsightListResponse: List of reflection prompts with metadata

    Raises:
        HTTPException: 500 on database errors

    Example:
        GET /api/insights/reflections?limit=5
        Response:
        {
            "success": true,
            "count": 3,
            "insights": [
                {
                    "id": 10,
                    "type": "reflection",
                    "markdown": "# Reflection Prompt: Machine Learning...",
                    "created_at": "2024-11-18T09:00:00",
                    "updated_at": "2024-11-18T09:00:00"
                }
            ]
        }
    """
    try:
        logger.info(
            "Listing reflection prompts",
            extra={"limit": limit},
        )

        # Query reflections ordered by most recent
        reflections = (
            db.query(Insight)
            .filter(Insight.type == "reflection")
            .order_by(desc(Insight.created_at))
            .limit(limit)
            .all()
        )

        logger.info(
            "Reflection prompts retrieved",
            extra={"count": len(reflections)},
        )

        return InsightListResponse(
            success=True,
            count=len(reflections),
            insights=reflections,
        )

    except Exception as e:
        logger.error(
            "Failed to list reflection prompts",
            exc_info=True,
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve reflection prompts",
        )


@router.post(
    "/insights/reflections/generate",
    response_model=GenerateResponse,
)
def generate_reflection(
    request: GenerateReflectionRequest, db: Session = Depends(get_db)
):
    """Trigger generation of a reflection prompt.

    Analyzes recent documents and generates a personalized reflection prompt
    to encourage deeper learning. Can optionally focus on a specific area.
    The reflection is stored in the database.

    Request Body:
        focus_area: Optional topic area to focus on (e.g., 'machine learning')

    Returns:
        GenerateResponse: Generation status and insight ID

    Raises:
        HTTPException: 500 on generation failure

    Example:
        POST /api/insights/reflections/generate
        {
            "focus_area": "machine learning"
        }

        Response:
        {
            "success": true,
            "message": "Reflection prompt generated successfully",
            "insight_id": 11
        }
    """
    try:
        logger.info(
            "Starting reflection generation",
            extra={"focus_area": request.focus_area},
        )

        # Generate reflection
        markdown_content = reflection.generate_prompt(
            focus_area=request.focus_area,
            db=db,
        )

        if not markdown_content:
            logger.warning(
                "No markdown content generated for reflection",
                extra={"focus_area": request.focus_area},
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to generate reflection content",
            )

        # Store in database
        insight = Insight(
            type="reflection",
            markdown=markdown_content,
        )
        db.add(insight)
        db.commit()
        db.refresh(insight)

        logger.info(
            "Reflection generated and stored",
            extra={
                "insight_id": insight.id,
                "focus_area": request.focus_area,
                "markdown_length": len(markdown_content),
            },
        )

        return GenerateResponse(
            success=True,
            message="Reflection prompt generated successfully",
            insight_id=insight.id,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Failed to generate reflection",
            exc_info=True,
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "focus_area": request.focus_area,
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate reflection: {str(e)}",
        )
