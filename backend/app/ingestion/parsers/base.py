"""Base parser class and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ParsedBlock:
    """A block of text extracted from a document."""

    text: str
    page: int | None = None
    section: str | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed document with blocks and metadata."""

    blocks: List[ParsedBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    language: str | None = None


class ParserBase(ABC):
    """Base class for all document parsers."""

    @classmethod
    @abstractmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse the file and extract text blocks.

        Args:
            file_path: Path to the file to parse

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            ParsingError: If parsing fails
        """
        pass

    def _create_block(
        self,
        text: str,
        page: int | None = None,
        section: str | None = None,
        offset_start: int | None = None,
        offset_end: int | None = None,
        **metadata,
    ) -> ParsedBlock:
        """
        Helper to create a ParsedBlock.

        Args:
            text: The text content
            page: Optional page number
            section: Optional section/heading
            offset_start: Optional byte offset start
            offset_end: Optional byte offset end
            **metadata: Additional metadata for the block

        Returns:
            ParsedBlock instance
        """
        return ParsedBlock(
            text=text,
            page=page,
            section=section,
            offset_start=offset_start,
            offset_end=offset_end,
            metadata=metadata,
        )
