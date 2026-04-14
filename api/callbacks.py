"""
Outbound HTTP callbacks from the LeafMachine execution service to Laravel.

Laravel owns the job table in MySQL.  This module is the only place where
the Python service writes back to Laravel.  All calls are fire-and-continue:
a failed callback is logged but never raises, so the pipeline is not aborted.
"""
from __future__ import annotations

import logging

import httpx

from api.auth import callback_headers
from api.config import settings
from api.schemas import FinalCallback, JobStatus, ProgressUpdate

logger = logging.getLogger("lm2.callbacks")

_TIMEOUT = 10.0  # seconds
_callbacks_disabled_due_auth = False


def _mark_callbacks_disabled_for_auth(status_code: int) -> None:
    global _callbacks_disabled_due_auth
    if status_code == 401 and not _callbacks_disabled_due_auth:
        _callbacks_disabled_due_auth = True
        logger.warning(
            "Laravel callback endpoint returned 401 Unauthorized; disabling further callbacks for this process."
        )


def _callback_url(job_id: str) -> str:
    return f"{settings.LARAVEL_CALLBACK_URL.rstrip('/')}/{job_id}/status"


def post_progress(job_id: str, step: str) -> None:
    """Push a pipeline progress update to Laravel.

    Laravel should update the job's ``progress_step`` column in MySQL.
    Safe to call frequently — failures are only logged, never raised.
    """
    if not settings.LARAVEL_CALLBACK_URL:
        logger.debug("Callback disabled (no LARAVEL_CALLBACK_URL). Step: %s", step)
        return
    if _callbacks_disabled_due_auth:
        logger.debug("Callback disabled after prior 401 response. Step: %s", step)
        return

    payload = ProgressUpdate(
        job_id=job_id,
        status=JobStatus.running,
        progress_step=step,
    ).model_dump()

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(_callback_url(job_id), json=payload, headers=callback_headers())
            _mark_callbacks_disabled_for_auth(resp.status_code)
            if resp.status_code >= 400:
                logger.warning(
                    "Progress callback returned HTTP %s for job %s [%s]",
                    resp.status_code,
                    job_id,
                    step,
                )
            else:
                logger.info(
                    "Progress callback delivered for job %s [%s] (HTTP %s)",
                    job_id,
                    step,
                    resp.status_code,
                )
    except Exception as exc:
        logger.warning("Progress callback failed for job %s [%s]: %s", job_id, step, exc)


def post_final(
    job_id: str,
    status: JobStatus,
    result_files: list[str],
    output_path: str,
    error_message: str | None = None,
    results_data: list[dict] | None = None,
    measurement_records: list[dict] | None = None,
    result_artifacts: list[dict] | None = None,
    callback_payload_meta: dict | None = None,
) -> None:
    """Push the final job outcome (completed or failed) to Laravel.

    Laravel should update the job row status and store the result metadata
    so the frontend can display results or error details without ever talking
    directly to this service.
    """
    if not settings.LARAVEL_CALLBACK_URL:
        logger.info(
            "Callback disabled. Final status for job %s: %s  files=%s",
            job_id, status, result_files,
        )
        return
    if _callbacks_disabled_due_auth:
        logger.info(
            "Final callback skipped for job %s because callbacks were disabled after 401 Unauthorized.",
            job_id,
        )
        return

    payload = FinalCallback(
        job_id=job_id,
        status=status,
        result_files=result_files,
        output_path=output_path,
        error_message=error_message,
        results_data=results_data,
        measurement_records=measurement_records,
        result_artifacts=result_artifacts,
        callback_payload_meta=callback_payload_meta,
    ).model_dump()

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(_callback_url(job_id), json=payload, headers=callback_headers())
            _mark_callbacks_disabled_for_auth(resp.status_code)
            if resp.status_code >= 400:
                logger.error("Final callback returned HTTP %s for job %s", resp.status_code, job_id)
            resp.raise_for_status()
            logger.info("Final callback delivered for job %s (HTTP %s)", job_id, resp.status_code)
    except Exception as exc:
        logger.error("Final callback failed for job %s: %s", job_id, exc)
