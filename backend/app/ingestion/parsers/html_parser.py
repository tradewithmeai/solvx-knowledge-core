"""HTML content parser using readability-lxml for main content extraction."""

import html
import re
from pathlib import Path
from typing import List

from lxml import etree
from lxml.html import HtmlElement, clean, fromstring
from readability import Document

from backend.app.ingestion.parsers.base import ParsedBlock, ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError
from backend.app.utils.io import read_file_safe

logger = get_logger(__name__)


class HTMLParser(ParserBase):
    """Parser for HTML documents using readability-lxml for content extraction."""

    SUPPORTED_EXTENSIONS = {".html", ".htm"}

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if file is a supported HTML file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and "html" in mime_type.lower():
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse HTML file and extract content with readability.

        Args:
            file_path: Path to HTML file

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            ParsingError: If parsing fails
        """
        try:
            logger.info(f"Parsing HTML file: {file_path}")

            # Read file with encoding fallback
            raw_content = read_file_safe(file_path)

            if not raw_content or not raw_content.strip():
                logger.warning(f"Empty HTML file: {file_path}")
                return ParsedDocument(blocks=[], metadata={"empty": True})

            # Extract title and main content
            title = self._extract_title(raw_content)

            # Use readability to extract main content
            readable_content = self._extract_readability_content(raw_content)

            # Parse content into blocks, preserving structure
            blocks = self._parse_content_blocks(readable_content)

            # If readability failed to find content, fall back to full HTML parsing
            if not blocks or len("".join(b.text for b in blocks).strip()) < 50:
                logger.debug(f"Readability produced minimal content, using fallback parsing for {file_path}")
                blocks = self._parse_html_fallback(raw_content)

            metadata = {
                "file_size": file_path.stat().st_size,
                "block_count": len(blocks),
                "total_chars": sum(len(block.text) for block in blocks),
            }

            logger.info(f"Successfully parsed HTML file: {file_path} ({len(blocks)} blocks, {len(title or '')} chars title)")

            return ParsedDocument(
                blocks=blocks,
                title=title,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse HTML file {file_path}: {e}")
            raise ParsingError(f"Failed to parse HTML file: {e}", {"file_path": str(file_path)})

    def _extract_title(self, html_content: str) -> str | None:
        """
        Extract title from HTML document.

        Args:
            html_content: Raw HTML content

        Returns:
            Title text or None if not found
        """
        try:
            # Try to extract from <title> tag first
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
            if title_match:
                title_text = title_match.group(1).strip()
                # Decode HTML entities
                title_text = html.unescape(title_text)
                if title_text:
                    return title_text

            # Fallback: try to extract from og:title meta tag
            og_match = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if og_match:
                return html.unescape(og_match.group(1).strip())

            return None
        except Exception as e:
            logger.debug(f"Error extracting title: {e}")
            return None

    def _extract_readability_content(self, html_content: str) -> str:
        """
        Extract main content using readability-lxml.

        Args:
            html_content: Raw HTML content

        Returns:
            Extracted HTML content or original if extraction fails
        """
        try:
            # Use readability Document to extract main content
            doc = Document(html_content, min_text_length=10, min_excerpt_length=10)
            readable_html = doc.summary()

            if readable_html and len(readable_html.strip()) > 20:
                return readable_html
            return html_content
        except Exception as e:
            logger.debug(f"Readability extraction failed, using fallback: {e}")
            return html_content

    def _parse_content_blocks(self, html_content: str) -> List[ParsedBlock]:
        """
        Parse HTML content into text blocks, preserving paragraph structure.

        Args:
            html_content: HTML content to parse

        Returns:
            List of ParsedBlock objects
        """
        blocks = []

        try:
            # Parse HTML safely
            try:
                root = fromstring(html_content)
            except Exception:
                # Try with lxml's more lenient parser
                parser = etree.HTMLParser(recover=True, remove_blank_text=True)
                root = etree.fromstring(html_content, parser)
                if isinstance(root, etree._Element):
                    # Convert to lxml html element
                    from lxml import html as lxml_html
                    root = lxml_html.fromstring(etree.tostring(root))

            if root is None:
                return blocks

            # Extract text from semantic elements
            blocks = self._extract_blocks_from_tree(root)

            return blocks
        except Exception as e:
            logger.debug(f"Error parsing content blocks: {e}")
            return blocks

    def _extract_blocks_from_tree(self, root: HtmlElement) -> List[ParsedBlock]:
        """
        Extract text blocks from parsed HTML tree.

        Args:
            root: Root HTML element

        Returns:
            List of ParsedBlock objects
        """
        blocks = []
        offset = 0

        # Remove script and style elements
        for element in root.xpath(".//script | .//style | .//noscript"):
            element.getparent().remove(element)

        # Process block-level elements in order
        block_selectors = [
            ".//p",  # Paragraphs
            ".//div[@class or @id]",  # Divs with class or id
            ".//article",  # Articles
            ".//section",  # Sections
            ".//blockquote",  # Block quotes
            ".//pre",  # Preformatted
            ".//li",  # List items
            ".//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6",  # Headings
        ]

        processed_elements = set()

        for selector in block_selectors:
            try:
                elements = root.xpath(selector)
                for elem in elements:
                    # Avoid processing the same element twice
                    if id(elem) in processed_elements:
                        continue

                    text = self._extract_text_from_element(elem)
                    text = text.strip()

                    if text and len(text) > 2:  # Skip very short fragments
                        processed_elements.add(id(elem))

                        block = self._create_block(
                            text=text,
                            offset_start=offset,
                            offset_end=offset + len(text.encode("utf-8")),
                        )
                        blocks.append(block)
                        offset += len(text.encode("utf-8")) + 1  # +1 for separator

            except Exception as e:
                logger.debug(f"Error processing selector {selector}: {e}")
                continue

        return blocks

    def _extract_text_from_element(self, elem: HtmlElement) -> str:
        """
        Extract clean text from HTML element, handling entities.

        Args:
            elem: HTML element

        Returns:
            Clean text content
        """
        try:
            # Get all text content including tail text
            text_parts = []

            def collect_text(element):
                """Recursively collect text from element and children."""
                if element.text:
                    text_parts.append(element.text)
                for child in element:
                    collect_text(child)
                    if child.tail:
                        text_parts.append(child.tail)

            collect_text(elem)
            full_text = "".join(text_parts)

            # Decode HTML entities
            full_text = html.unescape(full_text)

            # Normalize whitespace
            full_text = re.sub(r"\s+", " ", full_text)
            full_text = full_text.strip()

            return full_text
        except Exception as e:
            logger.debug(f"Error extracting text from element: {e}")
            # Fallback to direct text content
            try:
                return html.unescape(elem.text_content()).strip()
            except Exception:
                return ""

    def _parse_html_fallback(self, html_content: str) -> List[ParsedBlock]:
        """
        Fallback HTML parser for when readability produces minimal content.

        Args:
            html_content: Raw HTML content

        Returns:
            List of ParsedBlock objects
        """
        blocks = []

        try:
            # Clean HTML and remove scripts/styles
            cleaner = clean.Cleaner(scripts=True, javascript=True, style=True, links=True, meta=True, page_structure=False)

            try:
                html_clean = cleaner.clean_html(html_content)
            except Exception:
                html_clean = html_content

            # Parse cleaned HTML
            try:
                root = fromstring(html_clean)
            except Exception:
                parser = etree.HTMLParser(recover=True)
                root = etree.fromstring(html_clean, parser)
                from lxml import html as lxml_html
                root = lxml_html.fromstring(etree.tostring(root))

            if root is None:
                return blocks

            # Extract all text paragraphs
            offset = 0

            # Get all non-empty text nodes
            for elem in root.iter():
                if elem.tag in {"p", "div", "span", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "blockquote", "pre"}:
                    text = self._extract_text_from_element(elem)
                    if text and len(text) > 2:
                        block = self._create_block(
                            text=text,
                            offset_start=offset,
                            offset_end=offset + len(text.encode("utf-8")),
                        )
                        blocks.append(block)
                        offset += len(text.encode("utf-8")) + 1

            return blocks
        except Exception as e:
            logger.debug(f"Fallback HTML parsing failed: {e}")
            # Final fallback: extract all text
            try:
                text = re.sub(r"<[^>]+>", " ", html_content)
                text = html.unescape(text)
                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    return [
                        self._create_block(
                            text=text,
                            offset_start=0,
                            offset_end=len(text.encode("utf-8")),
                        )
                    ]
            except Exception:
                pass

            return blocks
