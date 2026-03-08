"""
File Utilities

Helper functions for validating and saving uploaded files.
All uploaded files are stored temporarily and removed after processing.
"""

import logging
import os
import uuid

from fastapi import HTTPException, UploadFile

from app.config import settings

logger = logging.getLogger(__name__)

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file(file: UploadFile):
    """
    Validate file extension and content type.
    Raises HTTP 400 if the file is not an accepted type.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{ext}' is not supported. "
                f"Accepted types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            ),
        )


async def save_temp_file(file: UploadFile) -> str:
    """
    Save an uploaded file to the temp directory with a unique name.
    Returns the absolute path to the saved file.
    Raises HTTP 413 if the file exceeds MAX_FILE_SIZE_MB.
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename or "file")[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(settings.TEMP_DIR, unique_name)

    content = await file.read()

    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB}MB size limit.",
        )

    with open(dest_path, "wb") as f:
        f.write(content)

    logger.debug("Saved temp file: %s (%d bytes)", dest_path, len(content))
    return dest_path


def cleanup_file(file_path: str):
    """Remove a temp file silently. Errors are logged but not raised."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Cleaned up: %s", file_path)
    except Exception as e:
        logger.warning("Could not remove temp file %s: %s", file_path, e)
