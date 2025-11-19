"""Production-ready file scanner and watcher for automatic document ingestion.

This module provides:
- Directory scanning for supported file types
- Hash-based change detection (BLAKE3)
- Database tracking of watched paths
- Real-time file monitoring with watchdog (optional)
- Automatic parsing, chunking, and embedding pipeline
- Comprehensive error handling and logging
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from sqlalchemy import Integer, String, Boolean, DateTime, Text, select
from sqlalchemy.orm import Session, Mapped, mapped_column
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from backend.app.config import settings
from backend.app.db.session import SessionLocal
from backend.app.embeddings.embedder import embed_chunks
from backend.app.ingestion.chunking import chunk_document
from backend.app.ingestion.parsers import (
    TextParser,
    MarkdownParser,
    PDFParser,
    HTMLParser,
    DOCXParser,
    CSVParser,
    IPYNBParser,
    CodeParser,
    OCRImageParser,
    ParserBase,
)
from backend.app.logging import get_logger
from backend.app.models.base import Base, TimestampMixin
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.utils.hash import hash_file
from backend.app.utils.io import (
    is_supported_file,
    should_ignore,
    get_mime_type,
    get_file_type_category,
)
from backend.app.vectorstore.chroma_store import get_vector_store

logger = get_logger(__name__)


# =============================================================================
# Database Model
# =============================================================================


class WatchedPath(Base, TimestampMixin):
    """Tracked directory path for automatic document scanning."""

    __tablename__ = "watched_paths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    recursive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ignore_patterns: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    last_scan_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<WatchedPath(id={self.id}, path={self.path}, enabled={self.enabled})>"


# =============================================================================
# Parser Factory
# =============================================================================


def get_parser(file_path: Path, mime_type: Optional[str] = None) -> Optional[ParserBase]:
    """
    Get appropriate parser for a file.

    Args:
        file_path: Path to the file
        mime_type: Optional MIME type (will be detected if not provided)

    Returns:
        Parser instance if supported, None otherwise
    """
    if mime_type is None:
        mime_type = get_mime_type(file_path)

    # Order matters - more specific parsers first
    parsers = [
        PDFParser(),
        MarkdownParser(),
        DOCXParser(),
        HTMLParser(),
        IPYNBParser(),
        CSVParser(),
        CodeParser(),
        OCRImageParser(),
        TextParser(),  # Most generic, last
    ]

    for parser in parsers:
        try:
            if parser.can_parse(file_path, mime_type):
                logger.debug(f"Selected parser {parser.__class__.__name__} for {file_path}")
                return parser
        except Exception as e:
            logger.warning(f"Parser {parser.__class__.__name__} check failed: {e}")
            continue

    logger.warning(f"No parser found for {file_path} (mime: {mime_type})")
    return None


# =============================================================================
# Core Scanner Functions
# =============================================================================


def add_watch_path(
    path: str | Path,
    db: Session,
    recursive: bool = True,
    ignore_patterns: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Optional[WatchedPath]:
    """
    Add a directory path to watch list.

    Args:
        path: Directory path to watch
        db: Database session
        recursive: Whether to scan subdirectories (default: True)
        ignore_patterns: List of glob patterns to ignore (optional)
        notes: Optional notes about this watched path

    Returns:
        WatchedPath object if successful, None on error
    """
    try:
        path = Path(path).resolve()

        # Validate path exists and is a directory
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return None

        if not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return None

        # Check if already watched
        existing = db.execute(
            select(WatchedPath).where(WatchedPath.path == str(path))
        ).scalar_one_or_none()

        if existing:
            logger.warning(f"Path already watched: {path}")
            # Update existing record
            existing.enabled = True
            existing.recursive = recursive
            if ignore_patterns:
                import json
                existing.ignore_patterns = json.dumps(ignore_patterns)
            if notes:
                existing.notes = notes
            db.commit()
            db.refresh(existing)
            logger.info(f"Updated watched path: {path}")
            return existing

        # Create new watched path
        import json
        watched_path = WatchedPath(
            path=str(path),
            enabled=True,
            recursive=recursive,
            ignore_patterns=json.dumps(ignore_patterns) if ignore_patterns else None,
            notes=notes,
        )

        db.add(watched_path)
        db.commit()
        db.refresh(watched_path)

        logger.info(
            f"Added watched path: {path} (recursive={recursive}, "
            f"ignore_patterns={len(ignore_patterns) if ignore_patterns else 0})"
        )

        return watched_path

    except Exception as e:
        logger.error(f"Failed to add watch path {path}: {e}", exc_info=True)
        db.rollback()
        return None


def remove_watch_path(path: str | Path, db: Session) -> bool:
    """
    Remove a directory path from watch list.

    Args:
        path: Directory path to remove
        db: Database session

    Returns:
        True if removed, False otherwise
    """
    try:
        path = Path(path).resolve()

        watched_path = db.execute(
            select(WatchedPath).where(WatchedPath.path == str(path))
        ).scalar_one_or_none()

        if not watched_path:
            logger.warning(f"Path not found in watch list: {path}")
            return False

        db.delete(watched_path)
        db.commit()

        logger.info(f"Removed watched path: {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to remove watch path {path}: {e}", exc_info=True)
        db.rollback()
        return False


def scan_path(
    path: str | Path,
    db: Session,
    recursive: bool = True,
    ignore_patterns: Optional[List[str]] = None,
    force_reindex: bool = False,
) -> dict:
    """
    Scan a directory path and process new/modified files.

    Args:
        path: Directory path to scan
        db: Database session
        recursive: Whether to scan subdirectories (default: True)
        ignore_patterns: List of glob patterns to ignore (optional)
        force_reindex: If True, reprocess all files even if unchanged (default: False)

    Returns:
        Dictionary with scan statistics:
            - scanned: Number of files scanned
            - new: Number of new files added
            - modified: Number of modified files updated
            - deleted: Number of deleted files removed
            - errors: Number of files that failed processing
            - skipped: Number of files skipped (ignored/unsupported)
    """
    stats = {
        "scanned": 0,
        "new": 0,
        "modified": 0,
        "deleted": 0,
        "errors": 0,
        "skipped": 0,
    }

    try:
        path = Path(path).resolve()

        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return stats

        if not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return stats

        logger.info(f"Starting scan of {path} (recursive={recursive}, force={force_reindex})")

        # Get all existing documents under this path
        existing_docs = {}
        for doc in db.execute(
            select(Document).where(Document.path.like(f"{str(path)}%"))
        ).scalars():
            existing_docs[doc.path] = doc

        # Track which files we've seen (to detect deletions)
        seen_paths = set()

        # Scan directory
        if recursive:
            file_paths = path.rglob("*")
        else:
            file_paths = path.glob("*")

        for file_path in file_paths:
            try:
                # Skip directories
                if not file_path.is_file():
                    continue

                stats["scanned"] += 1
                seen_paths.add(str(file_path))

                # Skip if should be ignored
                if should_ignore(file_path, ignore_patterns):
                    logger.debug(f"Ignoring file (pattern match): {file_path}")
                    stats["skipped"] += 1
                    continue

                # Skip if not supported
                if not is_supported_file(file_path):
                    logger.debug(f"Skipping unsupported file: {file_path}")
                    stats["skipped"] += 1
                    continue

                # Process file
                result = _process_file(
                    file_path=file_path,
                    db=db,
                    existing_doc=existing_docs.get(str(file_path)),
                    force_reindex=force_reindex,
                )

                if result == "new":
                    stats["new"] += 1
                elif result == "modified":
                    stats["modified"] += 1
                elif result == "error":
                    stats["errors"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                stats["errors"] += 1
                continue

        # Detect deleted files
        for doc_path, doc in existing_docs.items():
            if doc_path not in seen_paths:
                try:
                    _handle_deleted_file(doc, db)
                    stats["deleted"] += 1
                except Exception as e:
                    logger.error(f"Error handling deleted file {doc_path}: {e}", exc_info=True)
                    stats["errors"] += 1

        logger.info(
            f"Scan complete: {stats['scanned']} scanned, {stats['new']} new, "
            f"{stats['modified']} modified, {stats['deleted']} deleted, "
            f"{stats['errors']} errors, {stats['skipped']} skipped"
        )

        # Update last_scan_ts for watched path
        watched_path = db.execute(
            select(WatchedPath).where(WatchedPath.path == str(path))
        ).scalar_one_or_none()
        if watched_path:
            watched_path.last_scan_ts = datetime.utcnow()
            db.commit()

        return stats

    except Exception as e:
        logger.error(f"Failed to scan path {path}: {e}", exc_info=True)
        return stats


def scan_all_paths(db: Session, force_reindex: bool = False) -> dict:
    """
    Scan all enabled watched paths.

    Args:
        db: Database session
        force_reindex: If True, reprocess all files even if unchanged (default: False)

    Returns:
        Dictionary with aggregated scan statistics
    """
    total_stats = {
        "scanned": 0,
        "new": 0,
        "modified": 0,
        "deleted": 0,
        "errors": 0,
        "skipped": 0,
        "paths_scanned": 0,
    }

    try:
        # Get all enabled watched paths
        watched_paths = db.execute(
            select(WatchedPath).where(WatchedPath.enabled == True)
        ).scalars().all()

        if not watched_paths:
            logger.warning("No enabled watched paths found")
            return total_stats

        logger.info(f"Scanning {len(watched_paths)} watched paths")

        for watched_path in watched_paths:
            try:
                # Parse ignore patterns
                ignore_patterns = None
                if watched_path.ignore_patterns:
                    import json
                    ignore_patterns = json.loads(watched_path.ignore_patterns)

                # Scan the path
                stats = scan_path(
                    path=watched_path.path,
                    db=db,
                    recursive=watched_path.recursive,
                    ignore_patterns=ignore_patterns,
                    force_reindex=force_reindex,
                )

                # Aggregate stats
                for key in ["scanned", "new", "modified", "deleted", "errors", "skipped"]:
                    total_stats[key] += stats.get(key, 0)
                total_stats["paths_scanned"] += 1

            except Exception as e:
                logger.error(f"Error scanning path {watched_path.path}: {e}", exc_info=True)
                total_stats["errors"] += 1
                continue

        logger.info(
            f"All paths scan complete: {total_stats['paths_scanned']} paths, "
            f"{total_stats['scanned']} files scanned, {total_stats['new']} new, "
            f"{total_stats['modified']} modified, {total_stats['deleted']} deleted"
        )

        return total_stats

    except Exception as e:
        logger.error(f"Failed to scan all paths: {e}", exc_info=True)
        return total_stats


# =============================================================================
# File Processing
# =============================================================================


def _process_file(
    file_path: Path,
    db: Session,
    existing_doc: Optional[Document] = None,
    force_reindex: bool = False,
) -> str:
    """
    Process a single file: parse, chunk, embed, and store.

    Args:
        file_path: Path to the file
        db: Database session
        existing_doc: Existing document record if any
        force_reindex: Force reprocessing even if file hasn't changed

    Returns:
        Result status: "new", "modified", "unchanged", "error", "skipped"
    """
    try:
        # Check file access
        if not os.access(file_path, os.R_OK):
            logger.warning(f"No read permission for file: {file_path}")
            return "skipped"

        # Get file stats
        try:
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
        except Exception as e:
            logger.error(f"Failed to get file stats for {file_path}: {e}")
            return "error"

        # Skip empty files
        if file_size == 0:
            logger.debug(f"Skipping empty file: {file_path}")
            return "skipped"

        # Skip very large files (>100MB)
        if file_size > 100 * 1024 * 1024:
            logger.warning(f"Skipping very large file ({file_size / (1024*1024):.1f}MB): {file_path}")
            return "skipped"

        # Compute file hash
        try:
            file_hash = hash_file(file_path)
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            return "error"

        # Check if file has changed
        if existing_doc and not force_reindex:
            if existing_doc.hash == file_hash:
                logger.debug(f"File unchanged, skipping: {file_path}")
                return "unchanged"

        # Get parser
        parser = get_parser(file_path)
        if not parser:
            logger.warning(f"No parser available for {file_path}")
            return "skipped"

        # Parse the file
        logger.info(f"Parsing file: {file_path}")
        try:
            parsed_doc = parser.parse(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}", exc_info=True)
            # Update document status to error if it exists
            if existing_doc:
                existing_doc.status = "error"
                existing_doc.error = str(e)
                db.commit()
            return "error"

        if not parsed_doc or not parsed_doc.blocks:
            logger.warning(f"No content extracted from {file_path}")
            return "skipped"

        # Create or update document record
        is_new = existing_doc is None
        if is_new:
            doc = Document(
                path=str(file_path),
                source_type=get_file_type_category(file_path),
                hash=file_hash,
                title=parsed_doc.title or file_path.name,
                modified_ts=file_mtime,
                size_bytes=file_size,
                status="parsed",
            )
            db.add(doc)
            db.flush()  # Get the ID
            logger.info(f"Created new document record: {file_path} (id={doc.id})")
        else:
            doc = existing_doc
            # Delete old chunks and embeddings
            _delete_document_chunks(doc.id, db)
            # Update document
            doc.hash = file_hash
            doc.title = parsed_doc.title or file_path.name
            doc.modified_ts = file_mtime
            doc.size_bytes = file_size
            doc.status = "parsed"
            doc.error = None
            db.flush()
            logger.info(f"Updated document record: {file_path} (id={doc.id})")

        # Chunk the document
        try:
            chunks_data = chunk_document(
                document_id=doc.id,
                blocks=parsed_doc.blocks,
                target_size=settings.chunk_size_tokens,
                overlap=settings.chunk_overlap,
            )
        except Exception as e:
            logger.error(f"Failed to chunk {file_path}: {e}", exc_info=True)
            doc.status = "error"
            doc.error = f"Chunking failed: {e}"
            db.commit()
            return "error"

        if not chunks_data:
            logger.warning(f"No chunks created for {file_path}")
            doc.status = "error"
            doc.error = "No chunks created"
            db.commit()
            return "error"

        logger.info(f"Created {len(chunks_data)} chunks for {file_path}")

        # Create chunk records in database
        chunk_records = []
        for chunk_data in chunks_data:
            chunk = Chunk(
                document_id=doc.id,
                seq=chunk_data["seq"],
                text=chunk_data["text"],
                token_count=chunk_data["token_count"],
                page=chunk_data.get("page"),
                section=chunk_data.get("section"),
                offset_start=chunk_data.get("offset_start"),
                offset_end=chunk_data.get("offset_end"),
            )
            db.add(chunk)
            chunk_records.append(chunk)

        db.flush()  # Get chunk IDs
        logger.info(f"Saved {len(chunk_records)} chunk records to database")

        # Prepare chunks for embedding (add IDs)
        for i, chunk_data in enumerate(chunks_data):
            chunk_data["id"] = chunk_records[i].id
            chunk_data["doc_id"] = doc.id
            chunk_data["path"] = str(file_path)
            chunk_data["modified_ts"] = file_mtime

        # Generate embeddings
        try:
            embedded_chunks = embed_chunks(chunks_data, show_progress=False)
        except Exception as e:
            logger.error(f"Failed to embed chunks for {file_path}: {e}", exc_info=True)
            doc.status = "error"
            doc.error = f"Embedding failed: {e}"
            db.commit()
            return "error"

        # Filter out chunks that failed to embed
        valid_chunks = [c for c in embedded_chunks if c.get("embedding") is not None]
        if not valid_chunks:
            logger.error(f"No valid embeddings generated for {file_path}")
            doc.status = "error"
            doc.error = "All embeddings failed"
            db.commit()
            return "error"

        if len(valid_chunks) < len(embedded_chunks):
            logger.warning(
                f"Only {len(valid_chunks)}/{len(embedded_chunks)} chunks "
                f"successfully embedded for {file_path}"
            )

        # Store embeddings in vector store
        try:
            vector_store = get_vector_store()
            result = vector_store.upsert_chunks(valid_chunks)
            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))
            logger.info(f"Stored {result.get('chunks_upserted', 0)} embeddings in vector store")
        except Exception as e:
            logger.error(f"Failed to store embeddings for {file_path}: {e}", exc_info=True)
            doc.status = "error"
            doc.error = f"Vector store failed: {e}"
            db.commit()
            return "error"

        # Update document status
        doc.status = "embedded"
        doc.indexed_ts = datetime.utcnow()
        db.commit()

        logger.info(
            f"Successfully processed {file_path}: "
            f"{len(chunk_records)} chunks, {len(valid_chunks)} embedded"
        )

        return "new" if is_new else "modified"

    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
        db.rollback()
        return "error"


def _delete_document_chunks(document_id: int, db: Session) -> None:
    """
    Delete all chunks for a document from both database and vector store.

    Args:
        document_id: Document ID
        db: Database session
    """
    try:
        # Delete from vector store
        vector_store = get_vector_store()
        vector_store.delete_by_document_id(document_id)
        logger.debug(f"Deleted vector embeddings for document {document_id}")

        # Delete from database
        db.execute(
            Chunk.__table__.delete().where(Chunk.document_id == document_id)
        )
        db.flush()
        logger.debug(f"Deleted chunk records for document {document_id}")

    except Exception as e:
        logger.error(f"Error deleting chunks for document {document_id}: {e}", exc_info=True)
        # Don't raise - this is a cleanup operation


def _handle_deleted_file(doc: Document, db: Session) -> None:
    """
    Handle a file that has been deleted from the filesystem.

    Args:
        doc: Document record
        db: Database session
    """
    try:
        logger.info(f"Handling deleted file: {doc.path}")

        # Delete chunks and embeddings
        _delete_document_chunks(doc.id, db)

        # Mark document as deleted (or delete it entirely)
        doc.status = "deleted"
        db.commit()

        logger.info(f"Marked document as deleted: {doc.path}")

    except Exception as e:
        logger.error(f"Error handling deleted file {doc.path}: {e}", exc_info=True)
        db.rollback()
        raise


# =============================================================================
# Real-time File Monitoring (Watchdog)
# =============================================================================


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events using watchdog."""

    def __init__(self, watched_path: WatchedPath, debounce_seconds: float = 2.0):
        """
        Initialize file change handler.

        Args:
            watched_path: WatchedPath record
            debounce_seconds: Seconds to wait before processing a change (default: 2.0)
        """
        super().__init__()
        self.watched_path = watched_path
        self.debounce_seconds = debounce_seconds
        self._pending_changes: dict[str, float] = {}  # path -> timestamp
        self._ignore_patterns = None

        if watched_path.ignore_patterns:
            import json
            self._ignore_patterns = json.loads(watched_path.ignore_patterns)

    def _should_process(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        try:
            # Skip directories
            if file_path.is_dir():
                return False

            # Skip if ignored
            if should_ignore(file_path, self._ignore_patterns):
                return False

            # Skip if not supported
            if not is_supported_file(file_path):
                return False

            return True

        except Exception as e:
            logger.debug(f"Error checking if should process {file_path}: {e}")
            return False

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        if event.is_directory:
            return

        try:
            file_path = Path(event.src_path)
            if self._should_process(file_path):
                logger.debug(f"File created: {file_path}")
                self._pending_changes[str(file_path)] = time.time()
        except Exception as e:
            logger.error(f"Error handling created event: {e}", exc_info=True)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return

        try:
            file_path = Path(event.src_path)
            if self._should_process(file_path):
                logger.debug(f"File modified: {file_path}")
                self._pending_changes[str(file_path)] = time.time()
        except Exception as e:
            logger.error(f"Error handling modified event: {e}", exc_info=True)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if event.is_directory:
            return

        try:
            file_path = Path(event.src_path)
            logger.debug(f"File deleted: {file_path}")

            # Process immediately (no debounce for deletions)
            db = SessionLocal()
            try:
                doc = db.execute(
                    select(Document).where(Document.path == str(file_path))
                ).scalar_one_or_none()

                if doc:
                    _handle_deleted_file(doc, db)
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error handling deleted event: {e}", exc_info=True)

    def process_pending_changes(self) -> None:
        """Process pending file changes (debounced)."""
        if not self._pending_changes:
            return

        current_time = time.time()
        to_process = []

        # Find changes that have passed debounce period
        for file_path, timestamp in list(self._pending_changes.items()):
            if current_time - timestamp >= self.debounce_seconds:
                to_process.append(file_path)
                del self._pending_changes[file_path]

        # Process each file
        for file_path_str in to_process:
            try:
                file_path = Path(file_path_str)

                # Verify file still exists
                if not file_path.exists():
                    logger.debug(f"File no longer exists, skipping: {file_path}")
                    continue

                db = SessionLocal()
                try:
                    # Get existing document if any
                    existing_doc = db.execute(
                        select(Document).where(Document.path == str(file_path))
                    ).scalar_one_or_none()

                    # Process the file
                    result = _process_file(
                        file_path=file_path,
                        db=db,
                        existing_doc=existing_doc,
                        force_reindex=False,
                    )

                    logger.info(f"Processed file {file_path}: {result}")

                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Error processing pending change {file_path_str}: {e}", exc_info=True)


class FileWatcher:
    """Real-time file system watcher using watchdog."""

    def __init__(self, db: Session, check_interval: float = 5.0):
        """
        Initialize file watcher.

        Args:
            db: Database session for loading watched paths
            check_interval: Seconds between pending changes checks (default: 5.0)
        """
        self.db = db
        self.check_interval = check_interval
        self.observer = Observer()
        self.handlers: dict[str, FileChangeHandler] = {}
        self._running = False

    def start(self) -> None:
        """Start watching all enabled paths."""
        try:
            # Get all enabled watched paths
            watched_paths = self.db.execute(
                select(WatchedPath).where(WatchedPath.enabled == True)
            ).scalars().all()

            if not watched_paths:
                logger.warning("No enabled watched paths to monitor")
                return

            logger.info(f"Starting file watcher for {len(watched_paths)} paths")

            # Set up handlers for each path
            for watched_path in watched_paths:
                try:
                    path = Path(watched_path.path)
                    if not path.exists():
                        logger.warning(f"Watched path does not exist: {path}")
                        continue

                    handler = FileChangeHandler(watched_path)
                    self.observer.schedule(
                        handler,
                        str(path),
                        recursive=watched_path.recursive,
                    )
                    self.handlers[str(path)] = handler

                    logger.info(f"Watching path: {path} (recursive={watched_path.recursive})")

                except Exception as e:
                    logger.error(f"Failed to watch path {watched_path.path}: {e}", exc_info=True)
                    continue

            # Start the observer
            self.observer.start()
            self._running = True
            logger.info("File watcher started")

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}", exc_info=True)
            raise

    def stop(self) -> None:
        """Stop watching files."""
        try:
            if self._running:
                self.observer.stop()
                self.observer.join(timeout=10.0)
                self._running = False
                logger.info("File watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}", exc_info=True)

    def process_pending_changes(self) -> None:
        """Process pending changes for all handlers (debounced)."""
        for handler in self.handlers.values():
            try:
                handler.process_pending_changes()
            except Exception as e:
                logger.error(f"Error processing pending changes: {e}", exc_info=True)

    def is_alive(self) -> bool:
        """Check if watcher is running."""
        return self._running and self.observer.is_alive()
