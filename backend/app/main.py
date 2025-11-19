"""Main FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.app.config import settings
from backend.app.db.session import init_db
from backend.app.logging import RequestIDMiddleware, get_logger, setup_logging
from backend.app.routes import admin, files, search

# Set up logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Data directory: {settings.solvx_data_dir}")
    logger.info(f"Database path: {settings.solvx_db_path}")
    logger.info(f"Embedding provider: {settings.embed_provider}")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Local-first personal knowledge base with RAG, hybrid search, and intelligent insights",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestIDMiddleware)

# CORS configuration (disabled by default for security)
if settings.debug:
    logger.warning("CORS enabled in debug mode - do not use in production!")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(admin.router)
app.include_router(files.router)
app.include_router(search.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": __version__,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
