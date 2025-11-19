"""Chunk model for storing text segments."""

from sqlalchemy import Integer, String, Text, ForeignKey, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin


class Chunk(Base, TimestampMixin):
    """Text chunk extracted from a document."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # Sequence number in document
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    offset_start: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    offset_end: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    lang: Mapped[str | None] = mapped_column(String(10), nullable=True)

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, seq={self.seq})>"
