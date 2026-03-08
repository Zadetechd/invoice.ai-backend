"""
AI Invoice Extractor

Entry point for the FastAPI application.
Registers routes, configures logging, and validates settings on startup.

Run with:
    uvicorn app.main:app --reload
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    """Validate configuration and create required directories on startup."""
    logger.info("Starting %s v%s", settings.API_TITLE, settings.API_VERSION)
    try:
        settings.validate()
        logger.info("Configuration validated. LLM provider: %s", settings.LLM_PROVIDER)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        raise


@app.get("/", tags=["System"])
def root():
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
