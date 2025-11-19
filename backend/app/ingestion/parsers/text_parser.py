"""Plain text file parser."""

from pathlib import Path

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError
from backend.app.utils.io import read_file_safe

logger = get_logger(__name__)


class TextParser(ParserBase):
    """Parser for plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".text", ".log", ".rst", ".org"}

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """Check if file is a supported text file."""
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and mime_type.startswith("text/plain"):
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse plain text file.

        Args:
            file_path: Path to text file

        Returns:
            ParsedDocument with text content

        Raises:
            ParsingError: If file cannot be read
        """
        try:
            logger.info(f"Parsing text file: {file_path}")

            # Read file with encoding fallback
            content = read_file_safe(file_path)

            if not content or not content.strip():
                logger.warning(f"Empty text file: {file_path}")
                return ParsedDocument(blocks=[], metadata={"empty": True})

            # Create single block for entire content
            blocks = [
                self._create_block(
                    text=content,
                    offset_start=0,
                    offset_end=len(content.encode("utf-8")),
                )
            ]

            # Extract title from first line if it looks like a title
            title = None
            lines = content.split("\n")
            if lines and len(lines[0].strip()) < 100:
                title = lines[0].strip()

            metadata = {
                "file_size": file_path.stat().st_size,
                "line_count": len(lines),
            }

            logger.info(f"Successfully parsed text file: {file_path} ({len(lines)} lines)")

            return ParsedDocument(
                blocks=blocks,
                title=title,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {e}")
            raise ParsingError(f"Failed to parse text file: {e}", {"file_path": str(file_path)})
