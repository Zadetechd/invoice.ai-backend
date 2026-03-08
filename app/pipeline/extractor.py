"""
Invoice Extractor

The main pipeline orchestrator.
Connects OCR, preprocessing, LLM extraction, schema validation,
and confidence scoring into a single process_file call.

This is the function every other part of the system calls.
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

# Lazy-load the LLM provider once at module level
_llm_provider = None


def _get_provider():
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def process_file(file_path: str, file_name: str) -> InvoiceExtractionResult:
    """
    Run the full extraction pipeline on a single invoice file.

    Steps:
    1. Extract raw text via OCR or PDF parser
    2. Preprocess and clean the text
    3. Send to LLM for structured extraction
    4. Validate the result against InvoiceData schema
    5. Calculate confidence score
    6. Return InvoiceExtractionResult

    On any failure the function returns a result with status=failed
    rather than raising an exception, so batch processing continues.
    """
    logger.info("Processing file: %s", file_name)

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

    if not raw_text.strip():
        return InvoiceExtractionResult(
            file_name=file_name,
            status="failed",
            confidence_score=0.0,
            error="No text could be extracted from the file.",
            ocr_used=ocr_used,
        )

    # Step 2: Preprocessing
    clean_text = preprocess(raw_text)

    # Step 3: LLM extraction
    provider = _get_provider()
    raw_result = provider.extract(clean_text)

    if raw_result is None:
        return InvoiceExtractionResult(
            file_name=file_name,
            status="failed",
            confidence_score=0.0,
            error="LLM extraction returned no result.",
            ocr_used=ocr_used,
            raw_text_length=len(clean_text),
        )

    # Step 4: Schema validation
    invoice_data: Optional[InvoiceData] = None
    try:
        invoice_data = InvoiceData(**raw_result)
    except ValidationError as e:
        logger.warning("Schema validation failed for %s: %s", file_name, e)
        # Attempt a partial result by loading only valid fields
        safe_fields = {}
        for field_name in InvoiceData.model_fields:
            val = raw_result.get(field_name)
            if val is not None:
                safe_fields[field_name] = val
        try:
            invoice_data = InvoiceData(**safe_fields)
        except Exception:
            invoice_data = None

    # Step 5: Confidence scoring
    confidence = calculate_confidence(invoice_data)
    status = score_to_status(confidence)

    logger.info("Extraction complete for %s | status=%s | confidence=%.2f", file_name, status, confidence)

    return InvoiceExtractionResult(
        file_name=file_name,
        status=status,
        confidence_score=confidence,
        data=invoice_data,
        ocr_used=ocr_used,
        raw_text_length=len(clean_text),
    )
