"""
Application Configuration

All settings are read from environment variables.
Copy .env.example to .env and fill in your values before running.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    # LLM provider: gemini | openai
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # OpenAI (optional alternative)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Pipeline limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "20"))
    TEXT_CHUNK_SIZE: int = int(os.getenv("TEXT_CHUNK_SIZE", "2000"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))

    # Storage
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/invoice_extractor")
    EXPORT_DIR: str = os.getenv("EXPORT_DIR", "./exports")

    # API metadata
    API_TITLE: str = "AI Invoice Extractor"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = (
        "Document intelligence pipeline for structured invoice extraction "
        "powered by OCR and large language models."
    )

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # PDF uses pdfplumber. Images use Gemini vision directly so
    # tesseract is not required on the server.
    ALLOWED_EXTENSIONS: set = {".pdf", ".png", ".jpg", ".jpeg"}

    def validate(self):
        """Check required keys exist for the configured provider."""
        if self.LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is openai.")

        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.EXPORT_DIR, exist_ok=True)


settings = Settings()