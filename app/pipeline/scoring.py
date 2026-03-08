"""
Confidence Scoring

Generates a score between 0.0 and 1.0 reflecting how complete and
reliable the extracted invoice data is.

Scoring approach: field completeness weighting.
Core fields carry higher weight than supplementary fields.
The final score is the sum of weights for present fields divided
by the total possible weight.

This approach is deterministic, fast, and does not require an
extra LLM call. It closely mirrors what production document
intelligence systems use for SLA gating.
"""

import logging
from typing import Optional

from app.schemas.invoice_schema import InvoiceData

logger = logging.getLogger(__name__)

# Field weights. Higher weight means the field matters more to the score.
FIELD_WEIGHTS = {
    "vendor_name": 0.20,
    "invoice_number": 0.15,
    "invoice_date": 0.15,
    "total_amount": 0.20,
    "currency": 0.10,
    "line_items": 0.10,
    "due_date": 0.03,
    "subtotal": 0.03,
    "tax_amount": 0.02,
    "bill_to": 0.01,
    "payment_terms": 0.01,
}

TOTAL_WEIGHT = sum(FIELD_WEIGHTS.values())


def calculate_confidence(data: Optional[InvoiceData]) -> float:
    """
    Calculate a confidence score for the extracted invoice data.

    Returns 0.0 if data is None or empty.
    Returns a value between 0.0 and 1.0 based on field completeness.
    """
    if data is None:
        return 0.0

    score = 0.0
    data_dict = data.model_dump()

    for field, weight in FIELD_WEIGHTS.items():
        value = data_dict.get(field)
        if field == "line_items":
            # Award full weight only if at least one line item was extracted
            if value and len(value) > 0:
                score += weight
        elif value is not None and value != "":
            score += weight

    normalised = round(score / TOTAL_WEIGHT, 4)
    logger.debug("Confidence score: %.4f", normalised)
    return normalised


def score_to_status(score: float) -> str:
    """
    Convert a numeric confidence score to a human-readable status label.

    0.75 and above = success
    0.40 to 0.74   = partial
    Below 0.40     = failed
    """
    if score >= 0.75:
        return "success"
    elif score >= 0.40:
        return "partial"
    else:
        return "failed"
