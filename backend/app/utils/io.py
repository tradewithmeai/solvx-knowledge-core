"""File I/O and path utilities."""

import mimetypes
from pathlib import Path
from typing import List


# Common document extensions
SUPPORTED_EXTENSIONS = {
    # Text
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".org",
    # Documents
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    # Data
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    # Code
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    # Web
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    # Notebooks
    ".ipynb",
    # Images (for OCR)
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
}


def is_supported_file(file_path: Path) -> bool:
    """
    Check if file extension is supported.

    Args:
        file_path: Path to file

    Returns:
        True if file extension is supported
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def should_ignore(file_path: Path, ignore_patterns: List[str] | None = None) -> bool:
    """
    Check if file should be ignored based on patterns.

    Args:
        file_path: Path to file
        ignore_patterns: List of glob patterns to ignore

    Returns:
        True if file should be ignored
    """
    # Default ignore patterns
    default_ignores = [
        ".*",  # Hidden files
        "__pycache__",
        "node_modules",
        ".git",
        ".svn",
        ".hg",
        "venv",
        "env",
        ".venv",
        "build",
        "dist",
        "target",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dylib",
        "*.dll",
    ]

    patterns = ignore_patterns or default_ignores

    # Check each pattern
    for pattern in patterns:
        if pattern.startswith("*."):
            # Extension pattern
            if file_path.name.endswith(pattern[1:]):
                return True
        elif pattern.startswith("."):
            # Hidden file pattern
            if file_path.name.startswith("."):
                return True
        else:
            # Name pattern
            if pattern in str(file_path):
                return True

    return False


def get_mime_type(file_path: Path) -> str:
    """
    Get MIME type of file.

    Args:
        file_path: Path to file

    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def get_file_type_category(file_path: Path) -> str:
    """
    Categorize file by type.

    Args:
        file_path: Path to file

    Returns:
        Category string (text, pdf, document, code, data, html, notebook, image)
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return "pdf"
    elif ext in {".docx", ".doc", ".odt"}:
        return "document"
    elif ext in {".txt", ".md", ".markdown", ".rst", ".org"}:
        return "text"
    elif ext in {".html", ".htm", ".xml"}:
        return "html"
    elif ext in {".csv", ".tsv", ".json", ".jsonl"}:
        return "data"
    elif ext == ".ipynb":
        return "notebook"
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}:
        return "image"
    elif ext in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".go", ".rs", ".rb"}:
        return "code"
    else:
        return "unknown"


def read_file_safe(file_path: Path, encoding: str = "utf-8", errors: str = "replace") -> str:
    """
    Safely read file contents with error handling.

    Args:
        file_path: Path to file
        encoding: Text encoding
        errors: How to handle encoding errors

    Returns:
        File contents as string
    """
    try:
        return file_path.read_text(encoding=encoding, errors=errors)
    except Exception:
        # Try with different encodings
        for fallback_encoding in ["latin-1", "cp1252", "ascii"]:
            try:
                return file_path.read_text(encoding=fallback_encoding, errors=errors)
            except Exception:
                continue

        # Last resort: read as binary and decode with replace
        return file_path.read_bytes().decode("utf-8", errors="replace")
