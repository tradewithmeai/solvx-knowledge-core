"""Markdown file parser with front-matter support."""

import re
from pathlib import Path
from typing import Any, Dict, List

import mistune
from markdown import markdown

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError
from backend.app.utils.io import read_file_safe

logger = get_logger(__name__)


class MarkdownParser(ParserBase):
    """Parser for Markdown files."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """Check if file is a Markdown file."""
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and "markdown" in mime_type.lower():
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse Markdown file with front-matter and heading structure.

        Args:
            file_path: Path to Markdown file

        Returns:
            ParsedDocument with extracted text and structure

        Raises:
            ParsingError: If file cannot be parsed
        """
        try:
            logger.info(f"Parsing Markdown file: {file_path}")

            # Read file
            content = read_file_safe(file_path)

            if not content or not content.strip():
                logger.warning(f"Empty Markdown file: {file_path}")
                return ParsedDocument(blocks=[], metadata={"empty": True})

            # Extract front-matter (YAML)
            front_matter, body = self._extract_front_matter(content)

            # Parse heading structure
            blocks = self._parse_sections(body)

            # Extract title (from front-matter or first heading)
            title = front_matter.get("title")
            if not title and blocks:
                # Try first heading
                for block in blocks:
                    if block.section and block.section.startswith("#"):
                        title = block.section.lstrip("#").strip()
                        break

            metadata = {
                "file_size": file_path.stat().st_size,
                "front_matter": front_matter,
                "heading_count": sum(1 for b in blocks if b.section),
            }

            logger.info(f"Successfully parsed Markdown file: {file_path} ({len(blocks)} blocks)")

            return ParsedDocument(
                blocks=blocks,
                title=title,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse Markdown file {file_path}: {e}")
            raise ParsingError(f"Failed to parse Markdown file: {e}", {"file_path": str(file_path)})

    def _extract_front_matter(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extract YAML front-matter from Markdown.

        Args:
            content: Markdown content

        Returns:
            Tuple of (front_matter dict, body content)
        """
        # Match YAML front-matter (--- ... ---)
        front_matter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(front_matter_pattern, content, re.DOTALL)

        if not match:
            return {}, content

        yaml_content = match.group(1)
        body = content[match.end():]

        # Parse YAML (simple key: value parsing)
        front_matter = {}
        for line in yaml_content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                front_matter[key.strip()] = value.strip()

        return front_matter, body

    def _parse_sections(self, content: str) -> List:
        """
        Parse Markdown into sections based on headings.

        Args:
            content: Markdown body content

        Returns:
            List of ParsedBlocks organized by section
        """
        blocks = []
        lines = content.split("\n")

        current_section = None
        current_text = []
        offset = 0

        for line in lines:
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline

            # Check if line is a heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section if exists
                if current_text:
                    text = "\n".join(current_text).strip()
                    if text:
                        blocks.append(
                            self._create_block(
                                text=text,
                                section=current_section,
                                offset_start=offset - sum(len(l.encode("utf-8")) + 1 for l in current_text),
                                offset_end=offset,
                            )
                        )
                    current_text = []

                # Start new section
                current_section = line
                offset += line_bytes

            else:
                # Add to current section
                current_text.append(line)
                offset += line_bytes

        # Add final section
        if current_text:
            text = "\n".join(current_text).strip()
            if text:
                blocks.append(
                    self._create_block(
                        text=text,
                        section=current_section,
                        offset_start=offset - sum(len(l.encode("utf-8")) + 1 for l in current_text),
                        offset_end=offset,
                    )
                )

        # If no sections, create single block
        if not blocks:
            blocks.append(
                self._create_block(
                    text=content,
                    offset_start=0,
                    offset_end=len(content.encode("utf-8")),
                )
            )

        return blocks
