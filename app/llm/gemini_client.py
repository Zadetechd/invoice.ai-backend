"""
Gemini LLM Client

Calls Google Gemini 1.5 Flash via the google-generativeai SDK.
Supports two extraction modes:

  extract(text)            Text-based extraction for native PDFs
  extract_from_image(path) Vision-based extraction for image files,
                           sends the image directly to Gemini without
                           requiring tesseract to be installed
"""

import base64
import json
import logging
import re
from typing import Optional

import google.generativeai as genai

from app.config import settings
from app.llm.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are a document intelligence system specialised in invoice data extraction.

Extract all available invoice fields from the text below and return ONLY a valid JSON object.
Do not include any explanation, markdown, or extra text. Return raw JSON only.

Required fields to extract (use null if not found):
- vendor_name: string
- invoice_number: string
- invoice_date: string (ISO format YYYY-MM-DD if possible)
- due_date: string
- currency: string (3-letter code e.g. USD)
- subtotal: number
- tax_amount: number
- total_amount: number
- bill_to: string
- payment_terms: string
- notes: string
- line_items: array of objects with fields: item (string), quantity (number|null), unit_price (number|null), price (number)

Invoice text:
{text}
"""

VISION_PROMPT = """
You are a document intelligence system specialised in invoice data extraction.

The image attached is an invoice. Extract all available fields and return ONLY a valid JSON object.
Do not include any explanation, markdown, or extra text. Return raw JSON only.

Required fields to extract (use null if not found):
- vendor_name: string
- invoice_number: string
- invoice_date: string (ISO format YYYY-MM-DD if possible)
- due_date: string
- currency: string (3-letter code e.g. USD)
- subtotal: number
- tax_amount: number
- total_amount: number
- bill_to: string
- payment_terms: string
- notes: string
- line_items: array of objects with fields: item (string), quantity (number|null), unit_price (number|null), price (number)
"""


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that the model sometimes adds."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class GeminiClient(BaseLLMProvider):
    """Gemini 1.5 Flash provider for invoice extraction."""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info("GeminiClient initialised with model %s", settings.GEMINI_MODEL)

    def extract(self, text: str) -> Optional[dict]:
        """
        Send invoice text to Gemini and parse the JSON response.
        Used for native PDFs where text has already been extracted.
        Retries up to MAX_RETRIES times if JSON parsing fails.
        """
        prompt = EXTRACTION_PROMPT.format(text=text[:settings.TEXT_CHUNK_SIZE])

        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.debug("Gemini text extraction attempt %d", attempt)
                response = self.model.generate_content(prompt)
                raw = _strip_fences(response.text.strip())
                parsed = json.loads(raw)
                logger.info("Gemini text extraction succeeded on attempt %d", attempt)
                return parsed

            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed on attempt %d: %s", attempt, e)
            except Exception as e:
                logger.error("Gemini API error on attempt %d: %s", attempt, e)
                break

        logger.error("Gemini text extraction failed after %d attempts", settings.MAX_RETRIES)
        return None

    def extract_from_image(self, file_path: str) -> Optional[dict]:
        """
        Send an invoice image directly to Gemini vision for extraction.
        No tesseract or OCR required. Gemini reads the image visually.

        Supports PNG, JPG, and JPEG files.
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/jpeg"

        try:
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to read image file %s: %s", file_path, e)
            return None

        image_part = {"inline_data": {"mime_type": mime_type, "data": image_data}}

        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.debug("Gemini vision extraction attempt %d for %s", attempt, file_path)
                response = self.model.generate_content([VISION_PROMPT, image_part])
                raw = _strip_fences(response.text.strip())
                parsed = json.loads(raw)
                logger.info("Gemini vision extraction succeeded on attempt %d", attempt)
                return parsed

            except json.JSONDecodeError as e:
                logger.warning("Vision JSON parse failed on attempt %d: %s", attempt, e)
            except Exception as e:
                logger.error("Gemini vision API error on attempt %d: %s", attempt, e)
                break

        logger.error("Gemini vision extraction failed after %d attempts", settings.MAX_RETRIES)
        return None

    def health_check(self) -> bool:
        """Verify the Gemini API key is valid with a minimal request."""
        try:
            self.model.generate_content("Reply with the word OK only.")
            return True
        except Exception as e:
            logger.error("Gemini health check failed: %s", e)
            return False