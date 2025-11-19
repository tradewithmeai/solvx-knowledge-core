"""Production-ready PDF parser with PyMuPDF and pdfminer.six fallback."""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.ingestion.parsers.base import ParsedBlock, ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError

logger = get_logger(__name__)

# Try to import PyMuPDF (fitz)
try:
    import fitz

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.debug("PyMuPDF (fitz) not available, will use fallback")

# Try to import pdfminer.six
try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams

    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.debug("pdfminer.six not available, will use fallback")


class PDFParser(ParserBase):
    """
    Production-ready PDF parser with multiple backends.

    Uses PyMuPDF (fitz) as primary parser with pdfminer.six as fallback.
    Detects scanned pages and extracts metadata.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    # Thresholds for detecting scanned pages
    IMAGE_RATIO_THRESHOLD = 0.7  # If > 70% of page is images, mark as scanned
    MIN_TEXT_LENGTH = 50  # Minimum characters to consider a page as having real text

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if file is a PDF.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and "pdf" in mime_type.lower():
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF file and extract text, metadata, and page information.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            ParsingError: If PDF cannot be parsed
        """
        try:
            logger.info(f"Parsing PDF file: {file_path}")

            # Validate file exists
            if not file_path.exists():
                raise ParsingError(f"PDF file not found: {file_path}", {"file_path": str(file_path)})

            # Try PyMuPDF first (preferred for better performance)
            if PYMUPDF_AVAILABLE:
                return self._parse_with_pymupdf(file_path)

            # Fall back to pdfminer.six
            if PDFMINER_AVAILABLE:
                logger.info(f"PyMuPDF not available, using pdfminer.six fallback for {file_path}")
                return self._parse_with_pdfminer(file_path)

            # No parsers available
            raise ParsingError(
                "No PDF parser backends available. Install PyMuPDF or pdfminer.six.",
                {"file_path": str(file_path)},
            )

        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise ParsingError(f"Failed to parse PDF: {e}", {"file_path": str(file_path)})

    def _parse_with_pymupdf(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF using PyMuPDF (fitz).

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted content
        """
        try:
            logger.debug(f"Using PyMuPDF to parse: {file_path}")
            doc = fitz.open(file_path)

            blocks: List[ParsedBlock] = []
            metadata_dict: Dict[str, Any] = {}
            scanned_pages: List[int] = []

            try:
                # Extract metadata
                metadata = doc.metadata
                if metadata:
                    metadata_dict["author"] = metadata.get("author")
                    metadata_dict["title"] = metadata.get("title")
                    metadata_dict["subject"] = metadata.get("subject")
                    metadata_dict["creator"] = metadata.get("creator")
                    metadata_dict["producer"] = metadata.get("producer")
                    metadata_dict["creation_date"] = metadata.get("creationDate")
                    metadata_dict["modification_date"] = metadata.get("modDate")

                metadata_dict["page_count"] = doc.page_count
                metadata_dict["file_size"] = file_path.stat().st_size

                # Process each page
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()

                        # Check if page is likely scanned (has lots of images, little text)
                        is_scanned = self._is_scanned_page_pymupdf(page, page_text)
                        if is_scanned:
                            scanned_pages.append(page_num)

                        # Create block for page content
                        if page_text.strip():
                            block_metadata = {
                                "is_scanned": is_scanned,
                                "has_images": len(page.get_images()) > 0,
                                "image_count": len(page.get_images()),
                            }

                            block = self._create_block(
                                text=page_text.strip(),
                                page=page_num + 1,  # 1-indexed for display
                                **block_metadata,
                            )
                            blocks.append(block)
                        elif is_scanned:
                            # Scanned page with no text extract
                            block = self._create_block(
                                text="[Scanned page - OCR required]",
                                page=page_num + 1,
                                is_scanned=True,
                                has_images=len(page.get_images()) > 0,
                                image_count=len(page.get_images()),
                            )
                            blocks.append(block)

                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        # Continue with next page instead of crashing
                        continue

                # Set metadata for scanned pages
                if scanned_pages:
                    metadata_dict["scanned_pages"] = scanned_pages
                    metadata_dict["scan_percentage"] = round(
                        (len(scanned_pages) / doc.page_count * 100), 2
                    )

                # Extract title from metadata if available
                title = metadata_dict.get("title") if metadata else None

                logger.info(
                    f"Successfully parsed PDF with PyMuPDF: {file_path} "
                    f"({doc.page_count} pages, {len(scanned_pages)} scanned)"
                )

                return ParsedDocument(
                    blocks=blocks,
                    metadata=metadata_dict,
                    title=title,
                )

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF parsing failed for {file_path}: {e}")
            raise

    def _parse_with_pdfminer(self, file_path: Path) -> ParsedDocument:
        """
        Parse PDF using pdfminer.six as fallback.

        Note: pdfminer.six has limitations:
        - No easy per-page extraction
        - No metadata extraction
        - Cannot detect scanned pages

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted content
        """
        try:
            logger.debug(f"Using pdfminer.six to parse: {file_path}")

            # Extract text using pdfminer
            output_string = io.StringIO()
            try:
                extract_text_to_fp(
                    open(file_path, "rb"),
                    output_string,
                    laparams=LAParams(),
                )
            except Exception as e:
                logger.warning(f"pdfminer.six extraction failed: {e}")
                # Return minimal document instead of failing
                return ParsedDocument(
                    blocks=[],
                    metadata={"error": str(e), "parser": "pdfminer.six"},
                )

            text = output_string.getvalue()
            output_string.close()

            blocks: List[ParsedBlock] = []
            if text.strip():
                # pdfminer doesn't give per-page info, so create single block
                block = self._create_block(
                    text=text.strip(),
                    metadata={
                        "extraction_method": "pdfminer.six",
                        "note": "Per-page information not available from pdfminer.six",
                    },
                )
                blocks.append(block)

            metadata_dict = {
                "file_size": file_path.stat().st_size,
                "parser": "pdfminer.six",
                "note": "Limited metadata extraction with pdfminer.six. Consider installing PyMuPDF for better results.",
            }

            logger.info(f"Successfully parsed PDF with pdfminer.six: {file_path}")

            return ParsedDocument(
                blocks=blocks,
                metadata=metadata_dict,
            )

        except Exception as e:
            logger.error(f"pdfminer.six parsing failed for {file_path}: {e}")
            raise

    def _is_scanned_page_pymupdf(self, page: Any, text: str) -> bool:
        """
        Detect if a page is scanned (mostly images with little text).

        Args:
            page: PyMuPDF page object
            text: Extracted text from page

        Returns:
            True if page appears to be scanned
        """
        try:
            # Check if page has very little extractable text
            if len(text.strip()) < self.MIN_TEXT_LENGTH:
                # Check if page has images
                images = page.get_images()
                if images:
                    logger.debug(
                        f"Page {page.number + 1}: Detected as scanned "
                        f"(minimal text: {len(text.strip())} chars, {len(images)} images)"
                    )
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error detecting scanned page: {e}")
            return False
