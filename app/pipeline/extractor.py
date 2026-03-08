"""
Invoice Extractor

The main pipeline orchestrator.
Connects OCR, preprocessing, LLM extraction, schema validation,
and confidence scoring into a single process_file call.

For native PDFs: pdfplumber extracts text, which goes to the LLM.
For image files: tesseract is attempted first. If unavailable (e.g. on
hosted servers), the image is sent directly to Gemini vision which reads
it without any local OCR dependency.
"""

import logging
import os
from typing import Optional

from pydantic import ValidationError

from app.llm.factory import get_llm_provider
from app.pipeline.ocr import extract_text
from app.pipeline.preprocessing import preprocess
from app.pipeline.scoring import calculate_confidence, score_to_status
from app.schemas.invoice_schema import InvoiceData, InvoiceExtractionResult

logger = logging.getLogger(__name__)

_llm_provider = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _get_provider():
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def _validate_invoice_data(raw_result: dict, file_name: str) -> Optional[InvoiceData]:
    """Validate raw LLM output against the InvoiceData schema."""
    try:
        return InvoiceData(**raw_result)
    except ValidationError as e:
        logger.warning("Schema validation failed for %s: %s", file_name, e)
        safe_fields = {}
        for field_name in InvoiceData.model_fields:
            val = raw_result.get(field_name)
            if val is not None:
                safe_fields[field_name] = val
        try:
            return InvoiceData(**safe_fields)
        except Exception:
            return None


def process_file(file_path: str, file_name: str) -> InvoiceExtractionResult:
    """
    Run the full extraction pipeline on a single invoice file.

    Steps:
    1. Extract raw text via OCR or PDF parser
    2. If image and no text returned, use Gemini vision directly
    3. Preprocess and clean the text (text path only)
    4. Send to LLM for structured extraction
    5. Validate the result against InvoiceData schema
    6. Calculate confidence score
    7. Return InvoiceExtractionResult

    On any failure the function returns a result with status=failed
    rather than raising an exception, so batch processing continues.
    """
    logger.info("Processing file: %s", file_name)
    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in IMAGE_EXTENSIONS

    # Step 1: Text extraction
    try:
        raw_text, ocr_used = extract_text(file_path)
    except Exception as e:
        logger.error("Text extraction failed for %s: %s", file_name, e)
        return InvoiceExtractionResult(
            file_name=file_name,
            status="failed",
            confidence_score=0.0,
            error=f"Text extraction error: {str(e)}",
        )

    provider = _get_provider()
    raw_result: Optional[dict] = None

    # Step 2: Route to Gemini vision if image OCR returned nothing
    if is_image and not raw_text.strip():
        logger.info(
            "No OCR text for image %s — routing to Gemini vision", file_name
        )
        if hasattr(provider, "extract_from_image"):
            raw_result = provider.extract_from_image(file_path)
            ocr_used = False
        else:
            return InvoiceExtractionResult(
                file_name=file_name,
                status="failed",
                confidence_score=0.0,
                error="Image extraction requires a vision-capable LLM provider.",
                ocr_used=True,
            )

    else:
        # Step 2b: Text path — preprocess then send to LLM
        if not raw_text.strip():
            return InvoiceExtractionResult(
                file_name=file_name,
                status="failed",
                confidence_score=0.0,
                error="No text could be extracted from the file.",
                ocr_used=ocr_used,
            )

        clean_text = preprocess(raw_text)
        raw_result = provider.extract(clean_text)

    # Step 3: Check LLM returned something
    if raw_result is None:
        return InvoiceExtractionResult(
            file_name=file_name,
            status="failed",
            confidence_score=0.0,
            error="LLM extraction returned no result.",
            ocr_used=ocr_used,
            raw_text_length=len(raw_text) if raw_text else 0,
        )

    # Step 4: Schema validation
    invoice_data = _validate_invoice_data(raw_result, file_name)

    # Step 5: Confidence scoring
    confidence = calculate_confidence(invoice_data)
    status = score_to_status(confidence)

    logger.info(
        "Extraction complete for %s | status=%s | confidence=%.2f",
        file_name, status, confidence,
    )

    return InvoiceExtractionResult(
        file_name=file_name,
        status=status,
        confidence_score=confidence,
        data=invoice_data,
        ocr_used=ocr_used,
        raw_text_length=len(raw_text) if raw_text else 0,
    )