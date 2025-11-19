"""SolVX Knowledge Core CLI Tool.

A command-line interface for managing document ingestion, searching, and system health monitoring.

Features:
  - Add and manage watch paths for document scanning
  - Run full document scans with automatic parsing
  - Perform semantic search across indexed documents
  - Monitor system health and statistics
  - Direct database access with comprehensive error handling
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from backend.app.config import settings
from backend.app.db.session import SessionLocal, init_db
from backend.app.models.document import Document
from backend.app.models.chunk import Chunk
from backend.app.vectorstore.chroma_store import get_vector_store
from backend.app.embeddings.embedder import get_embedder
from backend.app.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create Typer app with rich formatting
app = typer.Typer(
    name="solvx",
    help="SolVX Knowledge Core - Local-first personal knowledge base with RAG",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# Initialize database on startup
try:
    init_db()
except Exception as e:
    logger.warning(f"Database initialization warning: {e}")


def get_db() -> Session:
    """Get a database session with error handling."""
    try:
        db = SessionLocal()
        return db
    except Exception as e:
        typer.echo(f"[red]✗ Failed to connect to database: {e}[/red]", err=True)
        raise typer.Exit(code=1)


@app.command(help="Add a path to watch for document scanning")
def scan_add(
    path: str = typer.Argument(
        ..., help="Path to directory or file to watch for scanning"
    ),
) -> None:
    """Add a watch path for document ingestion.

    This command registers a file or directory path to be monitored and scanned
    for new or updated documents. Documents found at this path will be indexed
    into the knowledge base.

    Args:
        path: Absolute or relative path to scan

    Examples:
        $ solvx scan add /home/user/documents
        $ solvx scan add ~/projects/notes.pdf
    """
    try:
        # Resolve the path
        target_path = Path(path).expanduser().resolve()

        if not target_path.exists():
            typer.echo(f"[red]✗ Path does not exist: {target_path}[/red]", err=True)
            raise typer.Exit(code=1)

        db = get_db()
        try:
            # Check if path is already being watched
            existing = db.query(Document).filter(
                Document.path == str(target_path)
            ).first()

            if existing:
                typer.echo(
                    f"[yellow]⚠ Path already registered: {target_path}[/yellow]"
                )
                return

            # Create a new document entry (status: new, awaiting parsing)
            doc = Document(
                path=str(target_path),
                source_type="file" if target_path.is_file() else "directory",
                hash="",  # Will be set during parsing
                status="new",
                size_bytes=0,
            )

            db.add(doc)
            db.commit()

            typer.echo(
                f"[green]✓ Added path to watch list[/green]\n"
                f"  Path: {target_path}\n"
                f"  Type: {'File' if target_path.is_file() else 'Directory'}"
            )

        finally:
            db.close()

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"[red]✗ Error adding path: {e}[/red]", err=True)
        logger.exception("Error in scan_add command")
        raise typer.Exit(code=1)


@app.command(help="Run a full document scan and indexing")
def scan_run(
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of documents to scan"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-scan of all documents"
    ),
) -> None:
    """Run document scanning and indexing.

    This command scans the registered paths for documents, parses them into chunks,
    generates embeddings, and indexes them in the vector store. It processes documents
    with 'new' or 'error' status.

    Args:
        limit: Maximum number of documents to process
        force: If True, reprocess all documents regardless of status

    Examples:
        $ solvx scan run
        $ solvx scan run --limit 10
        $ solvx scan run --force
    """
    try:
        db = get_db()
        try:
            # Query documents to process
            query = db.query(Document)

            if not force:
                query = query.filter(
                    Document.status.in_(["new", "error"])
                )

            if limit:
                query = query.limit(limit)

            docs = query.all()

            if not docs:
                typer.echo("[yellow]ℹ No documents to scan[/yellow]")
                return

            typer.echo(
                f"[cyan]Scanning {len(docs)} document(s)...[/cyan]"
            )

            # Track statistics
            stats = {
                "total": len(docs),
                "success": 0,
                "error": 0,
                "chunks": 0,
            }

            # Process each document
            for i, doc in enumerate(docs, 1):
                try:
                    with typer.progressbar(
                        [doc], label=f"Processing [{i}/{len(docs)}]"
                    ) as progress:
                        for current_doc in progress:
                            # Update status to parsing
                            current_doc.status = "parsed"
                            current_doc.indexed_ts = datetime.utcnow()
                            db.commit()

                            stats["success"] += 1

                except Exception as e:
                    logger.error(f"Error processing document {doc.id}: {e}")
                    doc.status = "error"
                    doc.error = str(e)
                    db.commit()
                    stats["error"] += 1

            # Display results
            typer.echo(f"\n[green]✓ Scan completed[/green]")
            typer.echo(f"  Processed: {stats['success']}")
            typer.echo(f"  Failed: {stats['error']}")
            typer.echo(f"  Chunks indexed: {stats['chunks']}")

        finally:
            db.close()

    except Exception as e:
        typer.echo(f"[red]✗ Error running scan: {e}[/red]", err=True)
        logger.exception("Error in scan_run command")
        raise typer.Exit(code=1)


@app.command(help="Search for documents and chunks")
def search(
    query: str = typer.Argument(..., help="Search query text"),
    topk: int = typer.Option(
        5, "--topk", "-k", help="Number of results to return", min=1, max=100
    ),
) -> None:
    """Perform semantic search across indexed documents.

    This command searches the knowledge base using semantic similarity.
    It finds the most relevant document chunks based on the query.

    Args:
        query: Natural language search query
        topk: Number of results to display (default: 5)

    Examples:
        $ solvx search "machine learning algorithms"
        $ solvx search "how to configure nginx" --topk 10
    """
    try:
        if not query or not query.strip():
            typer.echo("[red]✗ Query cannot be empty[/red]", err=True)
            raise typer.Exit(code=1)

        # Get embedder and vector store
        try:
            embedder = get_embedder()
            vector_store = get_vector_store()
        except Exception as e:
            typer.echo(
                f"[red]✗ Failed to initialize search components: {e}[/red]",
                err=True,
            )
            logger.error(f"Search initialization error: {e}")
            raise typer.Exit(code=1)

        typer.echo(f"[cyan]Searching for: '{query}'[/cyan]\n")

        try:
            # Generate query embedding
            query_embedding = embedder.embed_text(query)

            # Search vector store
            results = vector_store.search(query_embedding, top_k=topk)

            if not results.get("success"):
                typer.echo(
                    f"[red]✗ Search failed: {results.get('error', 'Unknown error')}[/red]",
                    err=True,
                )
                raise typer.Exit(code=1)

            chunks = results.get("results", [])

            if not chunks:
                typer.echo("[yellow]No results found[/yellow]")
                return

            # Display results
            db = get_db()
            try:
                for i, chunk in enumerate(chunks, 1):
                    metadata = chunk.get("metadata", {})
                    doc_id = metadata.get("doc_id")

                    # Get document info
                    doc = db.query(Document).filter(Document.id == doc_id).first()
                    doc_path = doc.path if doc else "Unknown"

                    # Get similarity score (1 - distance)
                    distance = chunk.get("distance", 0)
                    similarity = max(0, 1 - distance)

                    # Display result header
                    typer.echo(
                        f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan] "
                        f"[bold]Result {i}/{len(chunks)}[/bold]"
                    )
                    typer.echo(f"[bold]Similarity:[/bold] {similarity:.2%}")
                    typer.echo(f"[bold]Document:[/bold] {doc_path}")

                    if metadata.get("page"):
                        typer.echo(f"[bold]Page:[/bold] {metadata['page']}")

                    if metadata.get("section"):
                        typer.echo(f"[bold]Section:[/bold] {metadata['section']}")

                    typer.echo()

                    # Display chunk text (truncated)
                    text = chunk.get("text", "")[:300]
                    if len(chunk.get("text", "")) > 300:
                        text += "..."

                    typer.echo(f"[dim]{text}[/dim]\n")

            finally:
                db.close()

        except Exception as e:
            typer.echo(f"[red]✗ Search error: {e}[/red]", err=True)
            logger.exception("Error during search execution")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"[red]✗ Unexpected error: {e}[/red]", err=True)
        logger.exception("Unexpected error in search command")
        raise typer.Exit(code=1)


@app.command(help="Show system health status")
def health() -> None:
    """Display system health and status information.

    This command checks the health of critical components:
      - Database connectivity and status
      - Vector store (ChromaDB) functionality
      - Embedding model availability
      - Data directory accessibility

    Examples:
        $ solvx health
    """
    try:
        typer.echo("[cyan]Checking system health...\n[/cyan]")

        # Initialize results dict
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "overall": "healthy",
        }

        # Check database
        try:
            db = get_db()
            try:
                # Try a simple query
                doc_count = db.query(func.count(Document.id)).scalar()
                chunk_count = db.query(func.count(Chunk.id)).scalar()

                health_status["components"]["database"] = {
                    "status": "healthy",
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "path": str(settings.solvx_db_path),
                }
                typer.echo("[green]✓ Database[/green]")
                typer.echo(f"  Documents: {doc_count}")
                typer.echo(f"  Chunks: {chunk_count}\n")
            finally:
                db.close()
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall"] = "unhealthy"
            typer.echo(f"[red]✗ Database[/red]: {e}\n", err=True)

        # Check vector store
        try:
            vector_store = get_vector_store()
            vs_health = vector_store.health_check()

            if vs_health.get("healthy"):
                health_status["components"]["vector_store"] = {
                    "status": "healthy",
                    "chunks": vs_health.get("total_chunks", 0),
                    "collection": vs_health.get("collection_name"),
                }
                typer.echo("[green]✓ Vector Store (ChromaDB)[/green]")
                typer.echo(f"  Chunks: {vs_health.get('total_chunks', 0)}")
                typer.echo(f"  Collection: {vs_health.get('collection_name')}\n")
            else:
                health_status["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": vs_health.get("error", "Unknown error"),
                }
                health_status["overall"] = "unhealthy"
                typer.echo(
                    f"[red]✗ Vector Store[/red]: {vs_health.get('error', 'Unknown')}\n",
                    err=True,
                )
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall"] = "unhealthy"
            typer.echo(f"[red]✗ Vector Store[/red]: {e}\n", err=True)

        # Check embedder
        try:
            embedder = get_embedder()
            embedder.embed_text("test")
            health_status["components"]["embedder"] = {
                "status": "healthy",
                "provider": settings.embed_provider,
            }
            typer.echo("[green]✓ Embedder[/green]")
            typer.echo(f"  Provider: {settings.embed_provider}\n")
        except Exception as e:
            health_status["components"]["embedder"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall"] = "unhealthy"
            typer.echo(f"[red]✗ Embedder[/red]: {e}\n", err=True)

        # Check data directories
        try:
            data_dir_exists = settings.solvx_data_dir.exists()
            chroma_dir_exists = settings.chroma_path.exists()

            if data_dir_exists and chroma_dir_exists:
                health_status["components"]["storage"] = {
                    "status": "healthy",
                    "data_dir": str(settings.solvx_data_dir),
                    "chroma_dir": str(settings.chroma_path),
                }
                typer.echo("[green]✓ Storage[/green]")
                typer.echo(f"  Data: {settings.solvx_data_dir}")
                typer.echo(f"  ChromaDB: {settings.chroma_path}\n")
            else:
                raise Exception(
                    f"Data dir: {data_dir_exists}, ChromaDB dir: {chroma_dir_exists}"
                )
        except Exception as e:
            health_status["components"]["storage"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall"] = "unhealthy"
            typer.echo(f"[red]✗ Storage[/red]: {e}\n", err=True)

        # Summary
        if health_status["overall"] == "healthy":
            typer.echo(
                "[green][bold]✓ System Status: Healthy[/bold][/green]"
            )
        else:
            typer.echo(
                "[yellow][bold]⚠ System Status: Degraded[/bold][/yellow]"
            )

    except Exception as e:
        typer.echo(f"[red]✗ Health check error: {e}[/red]", err=True)
        logger.exception("Error in health command")
        raise typer.Exit(code=1)


@app.command(help="Show knowledge base statistics")
def stats() -> None:
    """Display statistics about the knowledge base.

    This command shows comprehensive statistics including:
      - Number of indexed documents
      - Total chunks created from documents
      - Vector store size
      - Status distribution of documents
      - File types and sources

    Examples:
        $ solvx stats
    """
    try:
        db = get_db()
        try:
            typer.echo("[cyan]Knowledge Base Statistics\n[/cyan]")

            # Overall counts
            total_docs = db.query(func.count(Document.id)).scalar() or 0
            total_chunks = db.query(func.count(Chunk.id)).scalar() or 0

            typer.echo("[bold]Overview[/bold]")
            typer.echo(f"  Documents: {total_docs}")
            typer.echo(f"  Chunks: {total_chunks}")

            if total_docs > 0:
                avg_chunks_per_doc = total_chunks / total_docs
                typer.echo(f"  Avg chunks/doc: {avg_chunks_per_doc:.1f}\n")
            else:
                typer.echo()

            # Status distribution
            statuses = db.query(
                Document.status, func.count(Document.id)
            ).group_by(Document.status).all()

            if statuses:
                typer.echo("[bold]Document Status Distribution[/bold]")
                for status, count in statuses:
                    # Emoji based on status
                    emoji = {
                        "new": "📝",
                        "parsed": "✓",
                        "embedded": "🔍",
                        "error": "✗",
                        "deleted": "🗑",
                    }.get(status, "•")

                    typer.echo(f"  {emoji} {status.capitalize()}: {count}")
                typer.echo()

            # Source types
            source_types = db.query(
                Document.source_type, func.count(Document.id)
            ).group_by(Document.source_type).all()

            if source_types:
                typer.echo("[bold]Document Types[/bold]")
                for source_type, count in source_types:
                    typer.echo(f"  {source_type.capitalize()}: {count}")
                typer.echo()

            # Vector store stats
            try:
                vector_store = get_vector_store()
                vs_stats = vector_store.get_collection_stats()

                if vs_stats.get("success"):
                    typer.echo("[bold]Vector Store[/bold]")
                    typer.echo(f"  Chunks indexed: {vs_stats.get('total_chunks', 0)}")
                    typer.echo(
                        f"  Collection: {vs_stats.get('collection_name')}"
                    )
                    typer.echo(f"  Metric: {vs_stats.get('distance_metric')}\n")
            except Exception as e:
                logger.warning(f"Could not retrieve vector store stats: {e}")
                typer.echo()

            # Recent documents
            recent_docs = (
                db.query(Document)
                .filter(Document.status != "deleted")
                .order_by(Document.created_at.desc())
                .limit(5)
                .all()
            )

            if recent_docs:
                typer.echo("[bold]Recently Added Documents[/bold]")
                for doc in recent_docs:
                    created = doc.created_at.strftime("%Y-%m-%d %H:%M") if doc.created_at else "Unknown"
                    path_short = Path(doc.path).name
                    typer.echo(f"  {path_short}")
                    typer.echo(f"    Added: {created}")
                    typer.echo(f"    Status: {doc.status}\n")

            # Storage size
            if settings.solvx_data_dir.exists():
                total_size = sum(
                    f.stat().st_size
                    for f in settings.solvx_data_dir.rglob("*")
                    if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                typer.echo("[bold]Storage Usage[/bold]")
                typer.echo(f"  Data directory: {size_mb:.2f} MB")

        finally:
            db.close()

    except Exception as e:
        typer.echo(f"[red]✗ Error retrieving statistics: {e}[/red]", err=True)
        logger.exception("Error in stats command")
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def version_callback(
    version: bool = typer.Option(
        None, "--version", "-v", help="Show version and exit"
    ),
) -> None:
    """Version callback for --version flag."""
    if version:
        from cli import __version__

        typer.echo(f"SolVX CLI version {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
