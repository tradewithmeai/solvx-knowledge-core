"""Hashing utilities using BLAKE3."""

from pathlib import Path

import blake3


def hash_text(text: str) -> str:
    """
    Compute BLAKE3 hash of text.

    Args:
        text: Text to hash

    Returns:
        Hexadecimal hash string
    """
    return blake3.blake3(text.encode("utf-8")).hexdigest()


def hash_file(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute BLAKE3 hash of file contents.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 64KB)

    Returns:
        Hexadecimal hash string
    """
    hasher = blake3.blake3()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_bytes(data: bytes) -> str:
    """
    Compute BLAKE3 hash of bytes.

    Args:
        data: Bytes to hash

    Returns:
        Hexadecimal hash string
    """
    return blake3.blake3(data).hexdigest()
