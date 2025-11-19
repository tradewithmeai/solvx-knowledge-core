"""Text processing and normalization utilities."""

import re
import unicodedata
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    - Normalizes Unicode to NFC form
    - Converts multiple whitespace to single space
    - Removes control characters (except newlines)
    - Strips leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except newline and tab
    text = "".join(char for char in text if char in "\n\t" or not unicodedata.category(char).startswith("C"))

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text using simple heuristics.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence splitting on common terminators
    # More sophisticated would use spaCy or nltk
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length, preserving word boundaries.

    Args:
        text: Input text
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Find last word boundary before max_length
    truncate_at = max_length - len(suffix)
    truncated = text[:truncate_at]

    # Trim to last complete word
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + suffix


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace while preserving paragraph structure.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Split into paragraphs
    paragraphs = text.split("\n\n")

    # Clean each paragraph
    cleaned = []
    for para in paragraphs:
        # Replace all whitespace with single spaces
        cleaned_para = " ".join(para.split())
        if cleaned_para:
            cleaned.append(cleaned_para)

    # Join back with double newlines
    return "\n\n".join(cleaned)


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count (rough approximation).

    Uses word count * 1.3 as approximation.
    For accurate counts, use tiktoken.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    words = len(text.split())
    return int(words * 1.3)
