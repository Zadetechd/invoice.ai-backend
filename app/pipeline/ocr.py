"""
OCR Pipeline

Handles text extraction from PDF and image files.

For native PDFs: uses pdfplumber to extract embedded text directly.
For scanned PDFs: converts pages to images then runs pytesseract if available.
For image files: returns empty text so the extractor can route to Gemini vision.

The pipeline automatically detects which method is needed based on
whether meaningful text exists in the PDF layer.
"""

import logging
import os
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)

EMBEDDED_TEXT_THRESHOLD = 50


def extract_text_from_pdf(file_path: str) -> tuple[str, bool]:
    """
    Extract text from a PDF file.

    Tries embedded text extraction first via pdfplumber.
    Falls back to OCR if the embedded text is too short or empty.

    Returns:
        tuple of (extracted_text, ocr_was_used)
    """
    logger.info("Extracting text from PDF: %s", os.path.basename(file_path))

    # Attempt 1: embedded text via pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            embedded_text = "\n".join(pages_text).strip()

        if len(embedded_text) >= EMBEDDED_TEXT_THRESHOLD:
            logger.info("Embedded text extracted (%d chars)", len(embedded_text))
            return embedded_text, False

        logger.info(
            "Embedded text too short (%d chars), falling back to OCR",
            len(embedded_text),
        )

    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)

    # Attempt 2: OCR via pdf2image + pytesseract
    return _ocr_pdf(file_path)


def _ocr_pdf(file_path: str) -> tuple[str, bool]:
    """Convert PDF pages to images and run OCR on each page."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        logger.info("Running OCR on PDF pages")
        images = convert_from_path(file_path, dpi=300)
        page_texts = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            page_texts.append(text)
            logger.debug("OCR completed for page %d", i + 1)

        full_text = "\n".join(page_texts).strip()
        logger.info("OCR produced %d chars from %d pages", len(full_text), len(images))
        return full_text, True

    except Exception as e:
        logger.error("OCR failed for PDF: %s", e)
        return "", True


def extract_text_from_image(file_path: str) -> tuple[str, bool]:
    """
    Attempt tesseract OCR on an image file.

    On hosted environments where tesseract is not installed this will
    return empty text. The extractor detects this and routes the file
    to Gemini vision extraction instead, which requires no local OCR.

    Returns:
        tuple of (extracted_text, ocr_was_used)
    """
    logger.info("Attempting OCR on image: %s", os.path.basename(file_path))
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        logger.info("Image OCR produced %d chars", len(text))
        return text.strip(), True
    except Exception as e:
        logger.warning(
            "Tesseract OCR unavailable (%s). "
            "Extractor will use Gemini vision instead.",
            e,
        )
        return "", True


def extract_text(file_path: str) -> tuple[str, bool]:
    """
    Main entry point for text extraction.
    Routes to the correct method based on file extension.

    For images, empty text signals the extractor to use Gemini vision.

    Returns:
        tuple of (extracted_text, ocr_was_used)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")