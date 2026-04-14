from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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


class MeasurementRecord(BaseModel):
    """Normalized measurement record for database persistence downstream."""
    filename: Optional[str] = None
    component_name: Optional[str] = None
    component_type: Optional[str] = None
    area: Optional[float] = None
    perimeter: Optional[float] = None
    bbox_min_long_side: Optional[float] = None
    bbox_min_short_side: Optional[float] = None
    units: Optional[str] = None
    conversion_factor_applied: Optional[float] = None
    aspect_ratio: Optional[float] = None


class ResultArtifact(BaseModel):
    """Single output artifact reference for downstream storage and display."""
    path: str
    kind: str
    media_type: str
    extension: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None


class CallbackPayloadMeta(BaseModel):
    """Delivery metadata used to detect callback payload truncation."""
    result_files_total: int = 0
    result_files_sent: int = 0
    result_files_truncated: bool = False
    results_data_total: int = 0
    results_data_sent: int = 0
    results_data_truncated: bool = False
    measurement_records_total: int = 0
    measurement_records_sent: int = 0
    measurement_records_truncated: bool = False
    result_artifacts_total: int = 0
    result_artifacts_sent: int = 0
    result_artifacts_truncated: bool = False


class FinalCallback(BaseModel):
    """Payload sent to Laravel on job completion or failure."""
    job_id: str
    status: JobStatus
    result_files: list[str] = Field(default_factory=list)
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    results_data: Optional[list[dict]] = None
    measurement_records: Optional[list[MeasurementRecord]] = None
    result_artifacts: Optional[list[ResultArtifact]] = None
    callback_payload_meta: Optional[CallbackPayloadMeta] = None
