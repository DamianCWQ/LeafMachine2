"""
LeafMachine2 pipeline runner.

This module is the single place that calls machine().  It runs as a
FastAPI BackgroundTask (in a threadpool worker, not the event loop) so it
may safely block for the duration of the pipeline without affecting other
API requests.

Responsibilities
----------------
- Validate config_overrides against a blocked-key list
- Deep-merge approved overrides into the default LM2 config
- Force-set controlled paths (input, output, run_name)
- Drive machine() and ferry progress steps back to Laravel via callbacks
- Collect result files and send the final callback
"""
from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

from api import callbacks
from api.config import settings
from api.schemas import JobStatus

logger = logging.getLogger("lm2.runner")

# ── Security: keys that callers must never override ───────────────────────────
# These are set exclusively by this service based on the job record.
_BLOCKED_KEYS: frozenset[str] = frozenset(
    {
        "dir_images_local",
        "dir_output",
        "run_name",
        "dir_home",
    }
)


def _validate_overrides(overrides: dict, path: str = "") -> None:
    """Recursively reject blocked keys and path-like string values.

    Raises ValueError with a descriptive message on the first violation.
    """
    for key, value in overrides.items():
        full_path = f"{path}.{key}" if path else key
        if key in _BLOCKED_KEYS:
            raise ValueError(
                f"Override key '{full_path}' is reserved and cannot be set by callers"
            )
        if isinstance(value, str) and (".." in value or value.startswith(("/", "\\"))):
            raise ValueError(
                f"Override value at '{full_path}' contains an unsafe path-like string"
            )
        if isinstance(value, dict):
            _validate_overrides(value, full_path)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Return a new dict with overrides recursively merged into base."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class _ProgressHook:
    """Translates machine()'s ProgressReport calls into Laravel callbacks.

    machine() expects an object with these methods:
        update_overall(step_name)   — 13 major pipeline steps
        update_batch(step_name)     — one call per batch
        update_batch_part(step_name)— 5 sub-steps per batch
        set_n_batches(n)            — called once after batch count is known
        reset_batch_part()          — called between batches
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def update_overall(self, step_name: str = "") -> None:
        callbacks.post_progress(self.job_id, step_name)

    def update_batch(self, step_name: str = "") -> None:
        callbacks.post_progress(self.job_id, step_name)

    def update_batch_part(self, step_name: str = "") -> None:
        callbacks.post_progress(self.job_id, step_name)

    def set_n_batches(self, n_batches: int) -> None:
        pass  # Batch count is informational; no state needed here

    def reset_batch_part(self) -> None:
        pass  # No per-batch-part state to reset


def _collect_results(output_path: str) -> list[str]:
    """Return relative paths of generated result files under output_path."""
    base = Path(output_path)
    if not base.exists():
        return []
    return [
        str(p.relative_to(base))
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".png", ".pdf"}
    ]


def run_job(job_id: str, input_dir: str, run_name: str, config_overrides: dict) -> None:
    """Execute the LeafMachine2 pipeline for a single job.

    Called from FastAPI BackgroundTasks after the HTTP response is already
    sent (202 Accepted).  Communicates state back to Laravel via callbacks.

    Parameters
    ----------
    job_id:
        UUID assigned by Laravel. Used for output directory naming and
        callback routing.
    input_dir:
        Absolute path to the uploaded images directory. Must already be
        validated as residing under LM2_UPLOAD_DIR by the router.
    run_name:
        Human-readable run identifier forwarded to LeafMachine2.
    config_overrides:
        Caller-provided partial config dict (already validated by the router
        via _validate_overrides before this function is called).  Merged on
        top of LeafMachine2 defaults before the pipeline runs.
    """
    lm2_home = settings.LM2_HOME
    output_path = str(Path(settings.LM2_OUTPUT_DIR) / job_id)

    # Ensure LeafMachine2 root is importable
    if lm2_home not in sys.path:
        sys.path.insert(0, lm2_home)

    # ── Import pipeline modules ──────────────────────────────────────────────
    try:
        from leafmachine2.machine.general_utils import load_config_file
        from leafmachine2.machine.machine import machine
    except ImportError as exc:
        logger.error("LeafMachine2 import failed for job %s: %s", job_id, exc)
        callbacks.post_final(job_id, JobStatus.failed, [], output_path, str(exc))
        return

    # ── Load and prepare config ──────────────────────────────────────────────
    callbacks.post_progress(job_id, "Loading configuration")
    try:
        cfg = load_config_file(lm2_home, None, system="LeafMachine2")
    except Exception as exc:
        logger.error("Config load failed for job %s: %s", job_id, exc)
        callbacks.post_final(job_id, JobStatus.failed, [], output_path, str(exc))
        return

    if config_overrides:
        cfg = _deep_merge(cfg, config_overrides)

    # Force-set controlled paths — these can never come from config_overrides
    cfg["leafmachine"]["project"]["dir_images_local"] = input_dir
    cfg["leafmachine"]["project"]["dir_output"] = output_path
    cfg["leafmachine"]["project"]["run_name"] = run_name

    os.makedirs(output_path, exist_ok=True)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    callbacks.post_progress(job_id, "Starting pipeline")
    hook = _ProgressHook(job_id)

    try:
        machine(cfg_file_path=None, dir_home=lm2_home, cfg_test=cfg, progress_report=hook)
    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        callbacks.post_final(job_id, JobStatus.failed, [], output_path, str(exc))
        return

    # ── Report success ───────────────────────────────────────────────────────
    result_files = _collect_results(output_path)
    logger.info("Job %s completed. %d result files.", job_id, len(result_files))
    callbacks.post_final(job_id, JobStatus.completed, result_files, output_path)


# ── Public validate helper (called by routers before enqueueing) ─────────────

def validate_config_overrides(overrides: dict) -> None:
    """Raise ValueError if overrides contain any blocked or unsafe keys.

    Called by the router *before* accepting the request so callers receive
    an HTTP 422 rather than a silent failure inside the background task.
    """
    _validate_overrides(overrides)
