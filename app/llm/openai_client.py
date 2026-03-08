"""
OpenAI LLM Client

Alternative provider using OpenAI GPT models.
Activated by setting LLM_PROVIDER=openai in .env.
Uses the same prompt structure as the Gemini client.
"""

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.llm.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are a document intelligence system specialised in invoice data extraction.

Extract all available invoice fields from the text below and return ONLY a valid JSON object.
Do not include any explanation, markdown, or extra text. Return raw JSON only.

Required fields (use null if not found):
vendor_name, invoice_number, invoice_date (YYYY-MM-DD), due_date, currency (3-letter code),
subtotal, tax_amount, total_amount, bill_to, payment_terms, notes,
line_items (array of: item, quantity, unit_price, price)

Invoice text:
{text}
"""


class OpenAIClient(BaseLLMProvider):
    """OpenAI GPT provider for invoice extraction."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        logger.info("OpenAIClient initialised with model %s", self.model)

    def extract(self, text: str) -> Optional[dict]:
        """Send invoice text to OpenAI and parse the JSON response."""
        prompt = EXTRACTION_PROMPT.format(text=text[:settings.TEXT_CHUNK_SIZE])

        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.debug("OpenAI extraction attempt %d", attempt)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)
                logger.info("OpenAI extraction succeeded on attempt %d", attempt)
                return parsed

            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed on attempt %d: %s", attempt, e)
            except Exception as e:
                logger.error("OpenAI API error on attempt %d: %s", attempt, e)
                break

        return None

    def health_check(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply OK"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error("OpenAI health check failed: %s", e)
            return False
