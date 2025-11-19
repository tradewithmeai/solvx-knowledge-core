"""Task model for background job tracking."""

from datetime import datetime

from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin


class Task(Base, TimestampMixin):
    """Background task tracking."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, running, completed, failed
    started: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON details

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, name={self.name}, status={self.status})>"
