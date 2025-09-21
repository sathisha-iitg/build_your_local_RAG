"""
Utility helpers for logging, text cleaning, and text chunking.
"""

import logging
import re
from typing import List

from src.constants import LOG_FILE_PATH


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
def setup_logging() -> None:
    """
    Configure application-wide logging.

    Logs are written to the path defined in constants, with
    timestamps, severity level, and message text included.
    """
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


# ---------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize OCR-extracted or raw text by removing artifacts.

    Fixes:
        - Hyphenated line breaks (e.g., "exam-\nple" → "example")
        - Newlines within sentences → replaced with spaces
        - Excessive newlines → collapsed into one
        - Extra whitespace → trimmed

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text string.
    """
    # Merge words split by hyphen at line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Convert single newlines inside sentences into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    result = text.strip()
    logging.info("Text cleaned successfully.")
    return result


# ---------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of tokens.

    Args:
        text (str): Input text.
        chunk_size (int): Max number of tokens per chunk.
        overlap (int): Number of tokens shared between consecutive chunks.

    Returns:
        List[str]: List of text chunks.
    """
    normalized = clean_text(text)
    logging.info("Text normalized for chunking.")

    tokens = normalized.split(" ")
    chunks: List[str] = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap handling

    logging.info(
        "Text divided into %d chunks (size=%d, overlap=%d).",
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
