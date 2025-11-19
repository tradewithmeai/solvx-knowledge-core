"""Document parsers for various file formats."""

from backend.app.ingestion.parsers.base import ParsedBlock, ParsedDocument, ParserBase
from backend.app.ingestion.parsers.code_parser import CodeParser
from backend.app.ingestion.parsers.csv_parser import CSVParser
from backend.app.ingestion.parsers.docx_parser import DOCXParser
from backend.app.ingestion.parsers.html_parser import HTMLParser
from backend.app.ingestion.parsers.ipynb_parser import IPYNBParser
from backend.app.ingestion.parsers.md_parser import MarkdownParser
from backend.app.ingestion.parsers.ocr_image_parser import OCRImageParser
from backend.app.ingestion.parsers.pdf_parser import PDFParser
from backend.app.ingestion.parsers.text_parser import TextParser

__all__ = [
    "ParserBase",
    "ParsedDocument",
    "ParsedBlock",
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "HTMLParser",
    "DOCXParser",
    "CSVParser",
    "IPYNBParser",
    "CodeParser",
    "OCRImageParser",
]
