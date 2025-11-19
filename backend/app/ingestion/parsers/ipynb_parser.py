"""Jupyter Notebook (.ipynb) parser with cell extraction and comment parsing."""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError
from backend.app.utils.io import read_file_safe

logger = get_logger(__name__)


class IPYNBParser(ParserBase):
    """Parser for Jupyter Notebook files (.ipynb)."""

    SUPPORTED_EXTENSIONS = {".ipynb"}

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if file is a Jupyter Notebook file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and "notebook" in mime_type.lower():
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse Jupyter Notebook and extract text and code blocks.

        Args:
            file_path: Path to the Jupyter Notebook file

        Returns:
            ParsedDocument with extracted cells and metadata

        Raises:
            ParsingError: If file cannot be parsed
        """
        try:
            logger.info(f"Parsing Jupyter Notebook: {file_path}")

            # Read file content
            content = read_file_safe(file_path)

            if not content or not content.strip():
                logger.warning(f"Empty Jupyter Notebook: {file_path}")
                return ParsedDocument(blocks=[], metadata={"empty": True})

            # Parse JSON
            try:
                notebook = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in notebook {file_path}: {e}")
                raise ParsingError(
                    f"Invalid JSON format in notebook: {e}",
                    {"file_path": str(file_path), "error": str(e)},
                )

            # Extract metadata
            metadata = self._extract_notebook_metadata(notebook)
            cells = notebook.get("cells", [])

            if not cells:
                logger.warning(f"No cells found in notebook: {file_path}")
                return ParsedDocument(blocks=[], metadata=metadata)

            # Parse cells
            blocks = self._parse_cells(cells)

            # Extract title from notebook metadata or first markdown heading
            title = self._extract_title(notebook, blocks)

            metadata.update(
                {
                    "file_size": file_path.stat().st_size,
                    "cell_count": len(cells),
                    "parsed_cells": len(blocks),
                }
            )

            logger.info(
                f"Successfully parsed Jupyter Notebook: {file_path} "
                f"({len(blocks)} blocks from {len(cells)} cells)"
            )

            return ParsedDocument(
                blocks=blocks,
                title=title,
                language="python",
                metadata=metadata,
            )

        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse Jupyter Notebook {file_path}: {e}")
            raise ParsingError(
                f"Failed to parse Jupyter Notebook: {e}",
                {"file_path": str(file_path), "error": str(e)},
            )

    def _extract_notebook_metadata(self, notebook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract notebook-level metadata.

        Args:
            notebook: Parsed notebook JSON object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Extract notebook format version
        if "nbformat" in notebook:
            metadata["notebook_format"] = f"{notebook.get('nbformat')}.{notebook.get('nbformat_minor', 0)}"

        # Extract kernel info
        if "metadata" in notebook and isinstance(notebook["metadata"], dict):
            nb_metadata = notebook["metadata"]
            if "kernelspec" in nb_metadata:
                kernel = nb_metadata["kernelspec"]
                metadata["kernel_name"] = kernel.get("display_name", kernel.get("name", "unknown"))
            if "language_info" in nb_metadata:
                lang = nb_metadata["language_info"]
                metadata["language_info"] = lang.get("name", "unknown")

        return metadata

    def _parse_cells(self, cells: List[Dict[str, Any]]) -> List:
        """
        Parse notebook cells into blocks.

        Args:
            cells: List of cell objects from notebook

        Returns:
            List of ParsedBlocks
        """
        blocks = []
        offset = 0

        for cell_number, cell in enumerate(cells, start=1):
            try:
                cell_type = cell.get("cell_type", "unknown")

                # Extract cell content
                source = cell.get("source", [])
                cell_text = self._join_source_lines(source)

                if not cell_text or not cell_text.strip():
                    continue

                # Process based on cell type
                if cell_type == "markdown":
                    block = self._process_markdown_cell(cell_text, cell_number, offset)
                    if block:
                        blocks.append(block)
                        offset += len(cell_text.encode("utf-8"))

                elif cell_type == "code":
                    # Extract code and comments separately
                    code_blocks = self._process_code_cell(cell_text, cell_number, offset)
                    blocks.extend(code_blocks)
                    offset += len(cell_text.encode("utf-8"))

                else:
                    # Skip unknown cell types
                    logger.debug(f"Skipping unknown cell type: {cell_type}")
                    continue

            except Exception as e:
                logger.warning(f"Failed to parse cell {cell_number}: {e}")
                continue

        return blocks

    def _process_markdown_cell(
        self, text: str, cell_number: int, offset_start: int
    ):
        """
        Process a markdown cell.

        Args:
            text: Cell source text
            cell_number: Cell number (1-indexed)
            offset_start: Byte offset in document

        Returns:
            ParsedBlock for the markdown cell
        """
        text = text.strip()
        if not text:
            return None

        offset_end = offset_start + len(text.encode("utf-8"))

        return self._create_block(
            text=text,
            section=f"Cell {cell_number} (Markdown)",
            offset_start=offset_start,
            offset_end=offset_end,
            cell_number=cell_number,
            cell_type="markdown",
        )

    def _process_code_cell(self, text: str, cell_number: int, offset_start: int) -> List:
        """
        Process a code cell, extracting code and comments.

        Args:
            text: Cell source text
            cell_number: Cell number (1-indexed)
            offset_start: Byte offset in document

        Returns:
            List of ParsedBlocks (one for code, one for comments if found)
        """
        blocks = []
        text = text.strip()

        if not text:
            return blocks

        offset_end = offset_start + len(text.encode("utf-8"))

        # Add the full code block
        code_block = self._create_block(
            text=text,
            section=f"Cell {cell_number} (Code)",
            offset_start=offset_start,
            offset_end=offset_end,
            cell_number=cell_number,
            cell_type="code",
        )
        blocks.append(code_block)

        # Extract and add comments separately
        comments = self._extract_comments(text)
        if comments:
            comment_text = "\n".join(comments)
            comment_block = self._create_block(
                text=comment_text,
                section=f"Cell {cell_number} (Comments)",
                offset_start=offset_start,
                offset_end=offset_end,
                cell_number=cell_number,
                cell_type="code_comments",
            )
            blocks.append(comment_block)

        return blocks

    def _extract_comments(self, code: str) -> List[str]:
        """
        Extract comments from Python code.

        Args:
            code: Python source code

        Returns:
            List of comment lines
        """
        comments = []

        try:
            # Try to parse as Python code to get accurate comment positions
            lines = code.split("\n")
            for line in lines:
                # Strip leading whitespace
                stripped = line.lstrip()
                # Check if line is a comment or contains a comment
                if stripped.startswith("#"):
                    # Extract the comment text (remove the # and leading space)
                    comment_text = stripped[1:].lstrip()
                    if comment_text:  # Skip empty comments
                        comments.append(comment_text)
                elif "#" in line:
                    # Inline comment - extract part after #
                    # But be careful not to match # inside strings
                    try:
                        # Simple heuristic: if # appears outside quotes
                        in_string = False
                        quote_char = None
                        for i, char in enumerate(line):
                            if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                                if not in_string:
                                    in_string = True
                                    quote_char = char
                                elif char == quote_char:
                                    in_string = False
                            elif char == "#" and not in_string:
                                comment_text = line[i + 1 :].strip()
                                if comment_text:
                                    comments.append(comment_text)
                                break
                    except Exception:
                        # If heuristic fails, skip inline comment parsing
                        pass

        except Exception as e:
            logger.debug(f"Failed to extract comments from code: {e}")
            # Fall back to simple regex
            lines = code.split("\n")
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    comment_text = stripped[1:].lstrip()
                    if comment_text:
                        comments.append(comment_text)

        return comments

    def _extract_title(self, notebook: Dict[str, Any], blocks: List) -> str | None:
        """
        Extract notebook title from metadata or first markdown heading.

        Args:
            notebook: Parsed notebook JSON object
            blocks: List of parsed blocks

        Returns:
            Title string or None
        """
        # Try to get from notebook metadata
        if "metadata" in notebook and isinstance(notebook["metadata"], dict):
            if "title" in notebook["metadata"]:
                return notebook["metadata"]["title"]

        # Try to get from first markdown block
        for block in blocks:
            if block.metadata.get("cell_type") == "markdown":
                text = block.text.strip()
                # Look for first heading
                heading_match = re.match(r"^#+\s+(.+)$", text, re.MULTILINE)
                if heading_match:
                    return heading_match.group(1)
                # Use first line if short enough
                first_line = text.split("\n")[0].strip()
                if first_line and len(first_line) < 200:
                    return first_line
                break

        return None

    def _join_source_lines(self, source: List[str] | str) -> str:
        """
        Join source lines into a single string.

        Args:
            source: Source as list of strings or single string

        Returns:
            Joined source text
        """
        if isinstance(source, str):
            return source
        elif isinstance(source, list):
            return "".join(source)
        else:
            return ""
