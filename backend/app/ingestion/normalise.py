"""Text normalization for consistent processing."""

from backend.app.utils.text import normalize_text


def normalise_text(text: str) -> str:
    """
    Normalise text for embedding and search.

    - Unicode normalization (NFC)
    - Whitespace normalization
    - Remove control characters
    - Consistent newline handling

    Args:
        text: Raw text

    Returns:
        Normalized text
    """
    return normalize_text(text)
