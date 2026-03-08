"""
Gemini LLM Client

Calls Google Gemini 1.5 Flash via the google-generativeai SDK.
Prompts the model to return structured invoice data as JSON.
Retries up to MAX_RETRIES times on parse failures.
"""

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


class GeminiClient(BaseLLMProvider):
    """Gemini 1.5 Flash provider for invoice extraction."""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info("GeminiClient initialised with model %s", settings.GEMINI_MODEL)

    def extract(self, text: str) -> Optional[dict]:
        """
        Send invoice text to Gemini and parse the JSON response.
        Retries up to MAX_RETRIES times if JSON parsing fails.
        """
        prompt = EXTRACTION_PROMPT.format(text=text[:settings.TEXT_CHUNK_SIZE])

        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.debug("Gemini extraction attempt %d", attempt)
                response = self.model.generate_content(prompt)
                raw = response.text.strip()

                # Strip markdown code fences if the model adds them
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)
                logger.info("Gemini extraction succeeded on attempt %d", attempt)
                return parsed

            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed on attempt %d: %s", attempt, e)
            except Exception as e:
                logger.error("Gemini API error on attempt %d: %s", attempt, e)
                break

        logger.error("Gemini extraction failed after %d attempts", settings.MAX_RETRIES)
        return None

    def health_check(self) -> bool:
        """Verify the Gemini API key is valid with a minimal request."""
        try:
            self.model.generate_content("Reply with the word OK only.")
            return True
        except Exception as e:
            logger.error("Gemini health check failed: %s", e)
            return False
