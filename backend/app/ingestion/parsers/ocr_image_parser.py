"""OCR image parser for extracting text from image files using Tesseract."""

from pathlib import Path
from typing import Optional

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError

logger = get_logger(__name__)


class OCRImageParser(ParserBase):
    """Parser for image files using OCR (Optical Character Recognition)."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}
    PYTESSERACT_AVAILABLE = False

    def __init__(self):
        """Initialize the OCR parser and check for pytesseract availability."""
        if not OCRImageParser.PYTESSERACT_AVAILABLE:
            try:
                import pytesseract

                OCRImageParser.PYTESSERACT_AVAILABLE = True
                self.pytesseract = pytesseract
                logger.debug("pytesseract is available")
            except ImportError:
                logger.warning(
                    "pytesseract not installed. OCR parsing will return empty results. "
                    "Install python-tesseract or pytesseract to enable OCR functionality."
                )
                self.pytesseract = None
        else:
            import pytesseract

            self.pytesseract = pytesseract

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if file is a supported image file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        if file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and mime_type.startswith("image/"):
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse an image file using OCR to extract text.

        Args:
            file_path: Path to the image file

        Returns:
            ParsedDocument with OCR-extracted text and image metadata

        Raises:
            ParsingError: If file cannot be read or processed
        """
        try:
            logger.info(f"Parsing image file: {file_path}")

            # Verify file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")

            if not file_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {file_path}")

            # Get image metadata
            image_metadata = self._get_image_metadata(file_path)

            # If pytesseract is not available, return empty document with metadata
            if not self.pytesseract:
                logger.warning(
                    f"pytesseract not available. Skipping OCR for image: {file_path}"
                )
                return ParsedDocument(
                    blocks=[],
                    metadata={
                        **image_metadata,
                        "ocr_available": False,
                        "reason": "pytesseract not installed",
                    },
                    title=file_path.stem,
                )

            # Perform OCR
            extracted_text = self._extract_text_from_image(file_path)

            if not extracted_text or not extracted_text.strip():
                logger.info(f"No text found in image: {file_path}")
                return ParsedDocument(
                    blocks=[],
                    metadata={**image_metadata, "ocr_available": True, "text_found": False},
                    title=file_path.stem,
                )

            # Create a single block with extracted text
            blocks = [
                self._create_block(
                    text=extracted_text.strip(),
                    offset_start=0,
                    offset_end=len(extracted_text.encode("utf-8")),
                    **image_metadata,
                )
            ]

            logger.info(
                f"Successfully parsed image file: {file_path} "
                f"({len(extracted_text)} chars, {image_metadata.get('width')}x{image_metadata.get('height')}px)"
            )

            return ParsedDocument(
                blocks=blocks,
                title=file_path.stem,
                metadata={
                    **image_metadata,
                    "ocr_available": True,
                    "text_found": True,
                    "text_length": len(extracted_text),
                    "file_size": file_path.stat().st_size,
                },
            )

        except FileNotFoundError as e:
            logger.error(f"Image file not found: {file_path}: {e}")
            raise ParsingError(f"Image file not found: {e}", {"file_path": str(file_path)})
        except IsADirectoryError as e:
            logger.error(f"Path is not a file: {file_path}: {e}")
            raise ParsingError(f"Path is not a file: {e}", {"file_path": str(file_path)})
        except Exception as e:
            logger.error(f"Failed to parse image file {file_path}: {e}", exc_info=True)
            raise ParsingError(
                f"Failed to parse image file: {e}", {"file_path": str(file_path)}
            )

    def _get_image_metadata(self, file_path: Path) -> dict:
        """
        Extract metadata from image file (dimensions, format, etc).

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with image metadata
        """
        metadata = {
            "format": file_path.suffix.lower().lstrip(".").upper(),
            "file_name": file_path.name,
        }

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["mode"] = img.mode
                metadata["image_format"] = img.format or "unknown"

                logger.debug(
                    f"Image metadata for {file_path.name}: "
                    f"{img.width}x{img.height} ({img.mode}, {img.format})"
                )

        except ImportError:
            logger.warning(
                "Pillow (PIL) not installed. Image dimension extraction unavailable."
            )
            metadata["image_dimensions_available"] = False

        except Exception as e:
            logger.warning(
                f"Could not extract image metadata for {file_path.name}: {e}"
            )
            metadata["metadata_extraction_error"] = str(e)

        return metadata

    def _extract_text_from_image(self, file_path: Path) -> str:
        """
        Extract text from image using Tesseract OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text from the image

        Raises:
            RuntimeError: If pytesseract is not available or OCR fails
        """
        if not self.pytesseract:
            raise RuntimeError(
                "pytesseract not available. Cannot perform OCR. "
                "Please install python-tesseract or pytesseract."
            )

        try:
            logger.debug(f"Running OCR on image: {file_path}")

            # Perform OCR using pytesseract
            text = self.pytesseract.image_to_string(str(file_path))

            logger.debug(f"OCR completed for {file_path.name}: {len(text)} chars extracted")

            return text

        except self.pytesseract.TesseractNotFoundError as e:
            logger.error(
                f"Tesseract not found. Ensure Tesseract OCR is installed and in PATH: {e}"
            )
            raise RuntimeError(
                "Tesseract OCR not found. Please install Tesseract to enable OCR functionality."
            ) from e

        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}") from e
