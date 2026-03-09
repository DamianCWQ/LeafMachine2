"""
Mock Laravel server for local testing of the LeafMachine2 execution service.

This script impersonates the Laravel side of the architecture so you can test
the full callback flow without a real Laravel + MySQL deployment.

What it does
------------
- Receives progress / final callbacks from the FastAPI service
- Lets you submit jobs (file upload or pre-transferred path) via a browser form
- Displays live job status in a browser dashboard (auto-refreshes every 3 s)
- Optional SIMULATION MODE: fakes the entire pipeline locally without calling
  the real FastAPI service (useful when the heavy ML stack is not available)

Quick start
-----------
1. Make sure api/.env has:
       LARAVEL_CALLBACK_URL=http://localhost:8000/api/internal/leafmachine/jobs
       LM2_SERVICE_URL=http://localhost:9000     (the real FastAPI service)

2. Run the mock server:
       python api/mock_laravel.py

3. In a second terminal, run the real service:
       uvicorn api.main:app --reload --port 9000

4. Open http://localhost:8000 in your browser.

Simulation mode (no ML stack needed)
--------------------------------------
Set MOCK_SIMULATE=true in api/.env. The mock will pretend to run the pipeline
locally, using MOCK_STEP_DELAY (seconds per step) and MOCK_FAIL_RATE (0–1).
"""
from __future__ import annotations

import asyncio
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse

# ── Load api/.env ─────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

CALLBACK_TOKEN: str = os.getenv("LARAVEL_CALLBACK_TOKEN", "").strip()
LM2_API_KEY: str = os.getenv("LM2_API_KEY", "").strip()
LM2_SERVICE_URL: str = os.getenv("LM2_SERVICE_URL", "http://localhost:9000").rstrip("/")
MOCK_PORT: int = int(os.getenv("MOCK_PORT", "8000"))
MOCK_STEP_DELAY: float = float(os.getenv("MOCK_STEP_DELAY", "0.8"))
MOCK_FAIL_RATE: float = float(os.getenv("MOCK_FAIL_RATE", "0.10"))
MOCK_SIMULATE: bool = os.getenv("MOCK_SIMULATE", "false").lower() in ("1", "true", "yes")

# ── Simulated pipeline steps (mirrors machine.py's 13 overall steps + batch) ─
_SIM_STEPS = [
    "Loaded config file",
    "Create Output Directory Structure",
    "Validate ML Files",
    "Created Project Storage Object",
    "Validate Input File Names and Types",
    "Save Copy of Config File",
    "Detect Archival Components",
    "Detect Plant Components",
    "Crop Individual Objects from Images",
    "SpecimenCrop Images",
    "Detecting Phenology",
    "Censoring Archival Components",
    "Binarize Labels",
    "Starting Batch 1 of 1",
    "Processing Rulers",
    "Segmenting Leaves",
    "Detecting Landmarks",
    "Building Overlays",
    "Saving Data",
]

# ── In-memory job store ───────────────────────────────────────────────────────
# Keyed by job_id.  Each value is a dict with:
#   job_id, run_name, submitted_at, status, progress_step,
#   error_message, result_files, output_path, history (list of raw payloads)
_jobs: dict[str, dict] = {}

app = FastAPI(title="Mock Laravel Server", docs_url="/api/docs", redoc_url=None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_job(job_id: str, run_name: str = "") -> dict:
    if job_id not in _jobs:
        _jobs[job_id] = {
            "job_id": job_id,
            "run_name": run_name,
            "submitted_at": _now(),
            "status": "pending",
            "progress_step": "",
            "error_message": None,
            "result_files": [],
            "output_path": None,
            "history": [],
        }
    return _jobs[job_id]


def _apply_callback(job_id: str, payload: dict) -> None:
    """Merge an incoming callback payload into the in-memory job record."""
    record = _upsert_job(job_id)
    record["history"].append({"received_at": _now(), **payload})
    if "status" in payload:
        record["status"] = payload["status"]
    if payload.get("progress_step"):
        record["progress_step"] = payload["progress_step"]
    if "error_message" in payload and payload["error_message"] is not None:
        record["error_message"] = payload["error_message"]
    if payload.get("result_files"):
        record["result_files"] = payload["result_files"]
    if payload.get("output_path"):
        record["output_path"] = payload["output_path"]


# ─────────────────────────────────────────────────────────────────────────────
# Callback endpoint  (Laravel equivalent)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/internal/leafmachine/jobs/{job_id}/status",
    status_code=status.HTTP_200_OK,
    summary="Receive a progress or final status callback from the Python service",
)
async def receive_callback(job_id: str, request: Request) -> dict:
    # Validate bearer token
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if CALLBACK_TOKEN and token != CALLBACK_TOKEN:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid callback token")

    payload = await request.json()
    _apply_callback(job_id, payload)

    step = payload.get("progress_step", "")
    job_status = payload.get("status", "?")
    print(f"  [callback] {job_id[:8]}…  status={job_status}  step={step!r}")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Job submission endpoints  (what Laravel's queue worker would do)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/submit/upload", summary="Submit a job by uploading files to the real service")
async def submit_upload(request: Request, background_tasks: BackgroundTasks) -> dict:
    """Multipart-forward: re-sends files + metadata to the real FastAPI /upload endpoint."""
    form = await request.form()
    run_name = form.get("run_name", "mock-test")
    job_id = form.get("job_id") or str(uuid.uuid4())
    config_overrides = form.get("config_overrides", "")

    _upsert_job(job_id, run_name)

    if MOCK_SIMULATE:
        background_tasks.add_task(_simulate_pipeline, job_id)
        return {"job_id": job_id, "status": "accepted", "mode": "simulation"}

    # Forward to real service
    files_raw = form.getlist("files")
    if not files_raw:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No files provided")

    form_data: list[tuple] = [
        ("job_id", (None, job_id)),
        ("run_name", (None, run_name)),
    ]
    if config_overrides:
        form_data.append(("config_overrides", (None, config_overrides)))
    for f in files_raw:
        data = await f.read()
        form_data.append(("files", (f.filename, data, f.content_type or "application/octet-stream")))

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LM2_SERVICE_URL}/api/v1/jobs/upload",
                files=form_data,
                headers={"X-API-Key": LM2_API_KEY},
            )
        resp.raise_for_status()
        _jobs[job_id]["status"] = "accepted"
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not reach LeafMachine service: {exc}")


@app.post("/api/submit/path", summary="Submit a job for pre-transferred files (path mode)")
async def submit_path(request: Request, background_tasks: BackgroundTasks) -> dict:
    body = await request.json()
    job_id = body.get("job_id") or str(uuid.uuid4())
    run_name = body.get("run_name", "mock-test")

    _upsert_job(job_id, run_name)

    if MOCK_SIMULATE:
        background_tasks.add_task(_simulate_pipeline, job_id)
        return {"job_id": job_id, "status": "accepted", "mode": "simulation"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LM2_SERVICE_URL}/api/v1/jobs",
                json=body,
                headers={"X-API-Key": LM2_API_KEY, "Content-Type": "application/json"},
            )
        resp.raise_for_status()
        _jobs[job_id]["status"] = "accepted"
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not reach LeafMachine service: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Job inspection
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/jobs", summary="List all tracked jobs")
def list_jobs() -> list[dict]:
    return [
        {k: v for k, v in job.items() if k != "history"}
        for job in reversed(list(_jobs.values()))
    ]


@app.get("/api/jobs/{job_id}", summary="Full history for one job")
def get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return _jobs[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation mode (no real ML stack needed)
# ─────────────────────────────────────────────────────────────────────────────

async def _simulate_pipeline(job_id: str) -> None:
    """Walk through the pipeline steps locally and self-post the callbacks."""
    await asyncio.sleep(0.2)
    _apply_callback(job_id, {"status": "running", "progress_step": "Starting pipeline"})

    for step in _SIM_STEPS:
        await asyncio.sleep(MOCK_STEP_DELAY)
        if random.random() < MOCK_FAIL_RATE:
            _apply_callback(
                job_id,
                {
                    "status": "failed",
                    "progress_step": step,
                    "error_message": f"[Simulated failure at step: {step}]",
                    "result_files": [],
                },
            )
            print(f"  [simulate] {job_id[:8]}…  FAILED at {step!r}")
            return
        _apply_callback(job_id, {"status": "running", "progress_step": step})
        print(f"  [simulate] {job_id[:8]}…  {step}")

    fake_files = [
        "Measurements__batch_0.csv",
        "Rulers__batch_0.csv",
        "Landmarks__batch_0.csv",
    ]
    _apply_callback(
        job_id,
        {
            "status": "completed",
            "progress_step": "Done",
            "result_files": fake_files,
            "output_path": f"/srv/leafmachine2/outputs/{job_id}",
        },
    )
    print(f"  [simulate] {job_id[:8]}…  COMPLETED")


# ─────────────────────────────────────────────────────────────────────────────
# Browser dashboard
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_COLORS = {
    "pending": "#6c757d",
    "accepted": "#0d6efd",
    "running": "#fd7e14",
    "completed": "#198754",
    "failed": "#dc3545",
    "cancel_requested": "#ffc107",
}

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeafMachine2 — Mock Laravel Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f8f9fa; color: #212529; }}
  header {{ background: #1a3c1a; color: white; padding: 1rem 2rem; }}
  header h1 {{ margin: 0; font-size: 1.4rem; }}
  header small {{ opacity: .7; font-size: .85rem; }}
  main {{ padding: 1.5rem 2rem; }}
  .badge {{ display: inline-block; padding: .2em .6em; border-radius: .3em; font-size: .8rem;
            color: white; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: .5rem;
           overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th {{ background: #e9ecef; text-align: left; padding: .6rem 1rem; font-size: .85rem; }}
  td {{ padding: .6rem 1rem; border-top: 1px solid #dee2e6; font-size: .875rem; vertical-align: middle; }}
  tr:hover td {{ background: #f1f3f5; }}
  .mono {{ font-family: monospace; font-size: .8rem; color: #6c757d; }}
  details summary {{ cursor: pointer; color: #0d6efd; font-size: .85rem; }}
  form {{ background: white; padding: 1.5rem; border-radius: .5rem;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 2rem; max-width: 700px; }}
  form h2 {{ margin-top: 0; font-size: 1.1rem; }}
  label {{ display: block; margin: .5rem 0 .2rem; font-size: .875rem; font-weight: 600; }}
  input, textarea {{ width: 100%; box-sizing: border-box; padding: .4rem .6rem; border: 1px solid #ced4da;
                     border-radius: .3rem; font-size: .875rem; }}
  button {{ margin-top: .8rem; padding: .45rem 1.2rem; background: #1a3c1a; color: white;
            border: none; border-radius: .3rem; cursor: pointer; font-size: .875rem; }}
  button:hover {{ background: #2d5e2d; }}
  .section-title {{ font-size: 1.05rem; font-weight: 700; margin: 1.5rem 0 .7rem; }}
  #refresh-note {{ font-size: .78rem; color: #6c757d; float: right; margin-top: .3rem; }}
  .result-files {{ font-size: .75rem; color: #6c757d; font-family: monospace; }}
  .sim-badge {{ background: #6610f2; color: white; font-size: .7rem; padding: .1em .5em;
                border-radius: .3em; margin-left: .4rem; vertical-align: middle; }}
</style>
</head>
<body>
<header>
  <h1>LeafMachine2 — Mock Laravel Dashboard</h1>
  <small>Simulates the Laravel server for local API testing
    {sim_label}
  </small>
</header>
<main>

<!-- Submit job -->
<div class="section-title">
  Submit a job
  <span id="refresh-note">Dashboard auto-refreshes every 3 s</span>
</div>

<form id="uploadForm" enctype="multipart/form-data">
  <h2>Upload images → POST /api/v1/jobs/upload</h2>
  <label>Job ID (leave blank to auto-generate)</label>
  <input type="text" name="job_id" placeholder="e.g. my-test-job-001">
  <label>Run name</label>
  <input type="text" name="run_name" value="local-test" required>
  <label>Image files</label>
  <input type="file" name="files" multiple accept="image/*">
  <label>config_overrides (JSON, optional)</label>
  <textarea name="config_overrides" rows="2" placeholder='{{"leafmachine":{{"project":{{"batch_size":50}}}}}}'></textarea>
  <button type="submit">Submit upload job</button>
</form>

<div class="section-title">Jobs</div>
<table>
  <thead>
    <tr>
      <th>Job ID</th><th>Run name</th><th>Status</th>
      <th>Progress step</th><th>Submitted</th><th>Result files</th>
    </tr>
  </thead>
  <tbody id="jobTableBody">
    <tr><td colspan="6" style="color:#aaa;text-align:center">Loading…</td></tr>
  </tbody>
</table>

</main>

<script>
const STATUS_COLORS = {json_colors};

function badge(s) {{
  const c = STATUS_COLORS[s] || '#aaa';
  return `<span class="badge" style="background:${{c}}">${{s}}</span>`;
}}

async function loadJobs() {{
  const resp = await fetch('/api/jobs');
  if (!resp.ok) return;
  const jobs = await resp.json();
  const tbody = document.getElementById('jobTableBody');
  if (jobs.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="6" style="color:#aaa;text-align:center">No jobs yet</td></tr>';
    return;
  }}
  tbody.innerHTML = jobs.map(j => {{
    const sub = new Date(j.submitted_at).toLocaleTimeString();
    const files = j.result_files && j.result_files.length
      ? `<span class="result-files">${{j.result_files.join('<br>')}}</span>`
      : '<span style="color:#aaa">—</span>';
    const err = j.error_message
      ? `<br><span style="color:#dc3545;font-size:.78rem">${{j.error_message}}</span>` : '';
    return `<tr>
      <td class="mono">${{j.job_id}}</td>
      <td>${{j.run_name}}</td>
      <td>${{badge(j.status)}}</td>
      <td>${{j.progress_step || '—'}}${{err}}</td>
      <td>${{sub}}</td>
      <td>${{files}}</td>
    </tr>`;
  }}).join('');
}}

// Auto-refresh every 3 s
loadJobs();
setInterval(loadJobs, 3000);

// Upload form submission
document.getElementById('uploadForm').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const fd = new FormData(e.target);
  if (!fd.get('job_id')) fd.set('job_id', crypto.randomUUID());
  try {{
    const resp = await fetch('/api/submit/upload', {{ method: 'POST', body: fd }});
    const data = await resp.json();
    if (!resp.ok) {{ alert('Error: ' + JSON.stringify(data)); return; }}
    alert('Job submitted: ' + data.job_id);
    loadJobs();
  }} catch(err) {{ alert('Request failed: ' + err); }}
}});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    sim_label = '<span class="sim-badge">SIMULATION MODE</span>' if MOCK_SIMULATE else ""
    import json as _json
    return HTMLResponse(
        _DASHBOARD_HTML.format(
            sim_label=sim_label,
            json_colors=_json.dumps(_STATUS_COLORS),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode_label = "SIMULATION (no real ML service needed)" if MOCK_SIMULATE else f"LIVE → {LM2_SERVICE_URL}"
    print("=" * 60)
    print("  Mock Laravel server")
    print(f"  Mode          : {mode_label}")
    print(f"  Dashboard     : http://localhost:{MOCK_PORT}")
    print(f"  Callback URL  : http://localhost:{MOCK_PORT}/api/internal/leafmachine/jobs")
    print()
    print("  Make sure api/.env has:")
    print(f"    LARAVEL_CALLBACK_URL=http://localhost:{MOCK_PORT}/api/internal/leafmachine/jobs")
    if not MOCK_SIMULATE:
        print(f"  Then start the real service:")
        print(f"    uvicorn api.main:app --reload --port 9000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=MOCK_PORT, log_level="warning")
