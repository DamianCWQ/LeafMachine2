# ─────────────────────────────────────────────────────────────────────────────
# LeafMachine2 Execution Service — Dockerfile
#
# Builds a container image for the Python FastAPI microservice only.
# The Laravel application runs on a separate server and is NOT included here.
#
# Build:
#   docker build -f api/Dockerfile -t leafmachine2-api .
#
# Run (development):
#   docker run --env-file api/.env \
#     -v /host/uploads:/srv/leafmachine2/uploads \
#     -v /host/outputs:/srv/leafmachine2/outputs \
#     -p 5000:5000 leafmachine2-api
# ─────────────────────────────────────────────────────────────────────────────

# ── Base image ────────────────────────────────────────────────────────────────
# Use the CUDA-enabled image if a GPU is available on the host. Swap the FROM
# line for `python:3.11-slim` for a CPU-only build.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /srv/leafmachine2

# ── Python dependencies ───────────────────────────────────────────────────────
# Install the microservice-specific deps first (faster layer cache).
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Install the full LeafMachine2 ML stack.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories (override via volume mounts in production) ────────────
RUN mkdir -p /srv/leafmachine2/uploads /srv/leafmachine2/outputs

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
