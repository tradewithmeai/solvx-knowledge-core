"""Document model for tracking ingested files."""

from datetime import datetime

from sqlalchemy import Integer, String, Text, DateTime, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """Document record tracking an ingested file."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)  # BLAKE3 hash
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array as text
    modified_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    indexed_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="new", index=True
    )  # new, parsed, embedded, error, deleted
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, path={self.path}, status={self.status})>"
