"""
LLM Provider Abstraction

Defines the interface every LLM provider must implement.
Swapping providers requires only changing LLM_PROVIDER in .env.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Any new provider (Anthropic, Mistral, local vLLM etc.) must
    extend this class and implement the extract method.
    """

    @abstractmethod
    def extract(self, text: str) -> Optional[dict]:
        """
        Send preprocessed invoice text to the LLM and return a dict
        matching the InvoiceData schema.

        Returns None if extraction fails after all retries.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and the key is valid."""
        pass
