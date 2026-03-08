"""
LLM Provider Factory

Returns the correct LLM client based on the LLM_PROVIDER setting.
Adding a new provider requires only adding one entry below.
"""

import logging
from app.config import settings
from app.llm.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider() -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    logger.info("Loading LLM provider: %s", provider)

    if provider == "gemini":
        from app.llm.gemini_client import GeminiClient
        return GeminiClient()

    if provider == "openai":
        from app.llm.openai_client import OpenAIClient
        return OpenAIClient()

    raise ValueError(
        f"Unknown LLM provider '{provider}'. Supported values: gemini, openai"
    )
