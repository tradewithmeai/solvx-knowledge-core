"""Structured logging configuration with request tracking."""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from backend.app.config import settings

# Context variable for request ID
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON-like structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Format as compact JSON-like string
        parts = [f"{k}={v}" for k, v in log_data.items()]
        return " ".join(parts)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to track request IDs."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add request ID to context and response headers."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def setup_logging() -> None:
    """Configure application logging."""
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create logs directory
    log_file = settings.logs_path / f"solvx_{datetime.now():%Y%m%d}.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # File handler with structured formatting and rotation
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logger.info(f"Logging configured level={settings.log_level} file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
