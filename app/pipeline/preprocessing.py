"""
Text Preprocessing

Cleans raw OCR or PDF text before it is sent to the LLM.
Clean input significantly improves extraction accuracy and reduces token usage.

Steps applied:
1. Normalise whitespace and line endings
2. Remove duplicate consecutive lines
3. Strip common OCR artefacts
4. Truncate to the configured chunk size
"""

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)


def normalise_whitespace(text: str) -> str:
    """Replace multiple spaces and tabs with a single space per line."""
    lines = text.splitlines()
    cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned)


def remove_duplicate_lines(text: str) -> str:
    """Remove consecutive duplicate lines that OCR sometimes produces."""
    lines = text.splitlines()
    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
        prev = line
    return "\n".join(deduped)


def remove_ocr_artefacts(text: str) -> str:
    """
    Remove common OCR noise characters.
    Keeps alphanumerics, punctuation useful for invoices, and whitespace.
    """
    # Remove lines that are just symbols or very short noise
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are entirely non-alphanumeric (e.g. "|||||||")
        if stripped and re.search(r"[a-zA-Z0-9]", stripped):
            cleaned.append(line)
    return "\n".join(cleaned)


def truncate_to_chunk(text: str) -> str:
    """
    Limit text to TEXT_CHUNK_SIZE characters.
    This keeps LLM token usage predictable and avoids prompt overflow.
    """
    limit = settings.TEXT_CHUNK_SIZE
    if len(text) > limit:
        logger.debug("Text truncated from %d to %d chars", len(text), limit)
        return text[:limit]
    return text


def preprocess(text: str) -> str:
    """
    Run all preprocessing steps in order and return cleaned text.
    This is the only function the rest of the pipeline calls.
    """
    if not text or not text.strip():
        logger.warning("Preprocessing received empty text")
        return ""

    text = normalise_whitespace(text)
    text = remove_duplicate_lines(text)
    text = remove_ocr_artefacts(text)
    text = truncate_to_chunk(text)

    logger.debug("Preprocessing complete. Output length: %d chars", len(text))
    return text.strip()
