"""
API Routes

Defines all HTTP endpoints.

POST /upload            Single invoice upload and extraction
POST /upload-batch      Multi-file upload that starts an async job
GET  /status/{job_id}   Poll the status of a batch job
GET  /export/json       Export last result as JSON (single or batch)
GET  /export/csv        Export last result as CSV (single or batch)
GET  /health            Health check for the API and LLM provider
"""

import json
import logging
import os
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.pipeline.extractor import process_file
from app.schemas.invoice_schema import (
    BatchExtractionResult,
    InvoiceExtractionResult,
    JobStatus,
)
from app.services.batch_processor import create_job, get_job
from app.utils.file_utils import cleanup_file, save_temp_file, validate_file

router = APIRouter()
logger = logging.getLogger(__name__)

# Holds the most recent result for the export endpoints.
# Updated by both single uploads and completed batch jobs.
_last_batch_result: BatchExtractionResult | None = None


@router.post(
    "/upload",
    response_model=InvoiceExtractionResult,
    summary="Extract data from a single invoice",
    tags=["Extraction"],
)
async def upload_single(file: UploadFile = File(...)):
    """
    Upload one invoice file and receive structured extracted data immediately.
    The result is saved to the export cache so /export/json and /export/csv
    work after a single upload without needing to use the batch endpoint.

    Supported formats: PDF, PNG, JPG, JPEG
    """
    global _last_batch_result

    validate_file(file)
    file_path = await save_temp_file(file)
    try:
        result = process_file(file_path, file.filename or "invoice")
    finally:
        cleanup_file(file_path)

    # Save to export cache
    successful = 1 if result.status in {"success", "partial"} else 0
    _last_batch_result = BatchExtractionResult(
        total_files=1,
        successful=successful,
        failed=1 - successful,
        results=[result],
    )

    return result


@router.post(
    "/upload-batch",
    response_model=dict,
    summary="Upload multiple invoices for async batch processing",
    tags=["Batch"],
)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Upload up to MAX_BATCH_SIZE invoices at once.
    Returns a job_id immediately. Poll /status/{job_id} for results.
    """
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.MAX_BATCH_SIZE} files.",
        )

    file_pairs = []
    for file in files:
        validate_file(file)
        path = await save_temp_file(file)
        file_pairs.append((path, file.filename or "invoice"))

    job_id = create_job(file_pairs)
    return {"job_id": job_id, "message": f"Batch job started for {len(files)} files."}


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Check batch job status",
    tags=["Batch"],
)
def get_status(job_id: str):
    """
    Poll the processing status of a batch job.
    When status is completed the full results are included in the response.
    """
    global _last_batch_result
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == "completed" and job.result:
        _last_batch_result = job.result

    return job


@router.get(
    "/export/json",
    summary="Export last results as JSON",
    tags=["Export"],
)
def export_json():
    """Download the most recent extraction results as a JSON file."""
    if not _last_batch_result:
        raise HTTPException(
            status_code=404,
            detail="No results available yet. Upload an invoice first.",
        )

    os.makedirs(settings.EXPORT_DIR, exist_ok=True)
    export_path = os.path.join(settings.EXPORT_DIR, "results.json")

    with open(export_path, "w") as f:
        json.dump(_last_batch_result.model_dump(), f, indent=2)

    return FileResponse(
        export_path,
        media_type="application/json",
        filename="invoice_results.json",
    )


@router.get(
    "/export/csv",
    summary="Export last results as CSV",
    tags=["Export"],
)
def export_csv():
    """Download the most recent extraction results as a CSV file."""
    if not _last_batch_result:
        raise HTTPException(
            status_code=404,
            detail="No results available yet. Upload an invoice first.",
        )

    os.makedirs(settings.EXPORT_DIR, exist_ok=True)
    export_path = os.path.join(settings.EXPORT_DIR, "results.csv")

    rows = []
    for r in _last_batch_result.results:
        row = {
            "file_name": r.file_name,
            "status": r.status,
            "confidence_score": r.confidence_score,
            "ocr_used": r.ocr_used,
        }
        if r.data:
            row.update({
                "vendor_name": r.data.vendor_name,
                "invoice_number": r.data.invoice_number,
                "invoice_date": r.data.invoice_date,
                "currency": r.data.currency,
                "total_amount": r.data.total_amount,
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)

    return FileResponse(
        export_path,
        media_type="text/csv",
        filename="invoice_results.csv",
    )


@router.get(
    "/health",
    summary="API and LLM provider health check",
    tags=["System"],
)
def health_check():
    """Returns the health status of the API and the configured LLM provider."""
    from app.llm.factory import get_llm_provider
    try:
        provider = get_llm_provider()
        llm_ok = provider.health_check()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "llm": str(e)},
        )

    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_reachable": llm_ok,
    }