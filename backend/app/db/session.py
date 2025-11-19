"""Database session management with SQLAlchemy."""

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import settings
from backend.app.logging import get_logger

logger = get_logger(__name__)

# Create SQLite engine with WAL mode for better concurrency
DATABASE_URL = f"sqlite:///{settings.solvx_db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Allow multi-threading
        "timeout": 30,  # Timeout for acquiring locks
    },
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL debugging
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas for optimal performance and safety."""
    cursor = dbapi_conn.cursor()

    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys=ON")

    # Synchronous mode for durability
    cursor.execute("PRAGMA synchronous=NORMAL")

    # Memory-mapped I/O for performance
    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB

    # Cache size (negative = KB, positive = pages)
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB

    cursor.close()


# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session.

    Yields:
        Database session that is automatically closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize the database (create tables if they don't exist)."""
    from backend.app.models.base import Base

    logger.info("Initialising database tables")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised successfully")
