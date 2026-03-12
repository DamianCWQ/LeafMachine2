from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


class JobStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancel_requested = "cancel_requested"


class JobRequest(BaseModel):
    """Body for POST /api/v1/jobs (path-based submission — files pre-transferred)."""
    job_id: str
    run_name: str
    config_overrides: dict = {}

    @field_validator("job_id")
    @classmethod
    def job_id_safe(cls, v: str) -> str:
        # Prevent path traversal via job_id
        import re
        if not re.fullmatch(r"[\w\-]{1,128}", v):
            raise ValueError("job_id must contain only alphanumeric characters, hyphens, or underscores (max 128 chars)")
        return v


class JobAccepted(BaseModel):
    """Returned immediately (202) when a job is enqueued."""
    job_id: str
    status: JobStatus = JobStatus.accepted
    message: str = "Job accepted and queued for processing"


class ProgressUpdate(BaseModel):
    """Payload sent to Laravel on each pipeline step (progress callback)."""
    job_id: str
    status: JobStatus
    progress_step: str
    error_message: Optional[str] = None


class FinalCallback(BaseModel):
    """Payload sent to Laravel on job completion or failure."""
    job_id: str
    status: JobStatus
    result_files: list[str] = []
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    results_data: Optional[list[dict]] = None
