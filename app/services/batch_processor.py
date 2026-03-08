"""
Batch Processor

Handles processing of multiple invoice files in a single request.
Tracks job state in memory so the status endpoint can poll progress.

Each job is stored in a dict keyed by job_id.
In a production system this state would live in Redis or a database.
"""

import logging
import os
import threading
import uuid
from typing import Dict

from app.pipeline.extractor import process_file
from app.schemas.invoice_schema import BatchExtractionResult, InvoiceExtractionResult, JobStatus

logger = logging.getLogger(__name__)

# In-memory job store: job_id -> JobStatus
_jobs: Dict[str, JobStatus] = {}
_jobs_lock = threading.Lock()


def create_job(file_paths: list[tuple[str, str]]) -> str:
    """
    Register a new batch job and start processing in a background thread.

    Args:
        file_paths: list of (absolute_path, original_filename) tuples

    Returns:
        job_id string
    """
    job_id = str(uuid.uuid4())
    total = len(file_paths)

    with _jobs_lock:
        _jobs[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            total_files=total,
            processed_files=0,
        )

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, file_paths),
        daemon=True,
    )
    thread.start()
    logger.info("Batch job %s started with %d files", job_id, total)
    return job_id


def _run_job(job_id: str, file_paths: list[tuple[str, str]]):
    """Process each file sequentially and update job state."""
    results = []

    with _jobs_lock:
        _jobs[job_id].status = "processing"

    for file_path, file_name in file_paths:
        try:
            result = process_file(file_path, file_name)
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", file_name, e)
            result = InvoiceExtractionResult(
                file_name=file_name,
                status="failed",
                confidence_score=0.0,
                error=str(e),
            )

        results.append(result)

        with _jobs_lock:
            _jobs[job_id].processed_files += 1

        # Clean up temp file
        try:
            os.remove(file_path)
        except Exception:
            pass

    successful = sum(1 for r in results if r.status in {"success", "partial"})
    failed = len(results) - successful

    batch_result = BatchExtractionResult(
        total_files=len(results),
        successful=successful,
        failed=failed,
        results=results,
    )

    with _jobs_lock:
        _jobs[job_id].status = "completed"
        _jobs[job_id].result = batch_result

    logger.info("Batch job %s completed. %d succeeded, %d failed", job_id, successful, failed)


def get_job(job_id: str) -> JobStatus | None:
    """Return the current state of a job, or None if not found."""
    with _jobs_lock:
        return _jobs.get(job_id)
