"""
LeafMachine2 execution microservice entry point.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 9000
"""
import logging
import uvicorn
from fastapi import FastAPI

from api.config import settings
from api.routers import health, jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="LeafMachine2 Execution Service",
    description=(
        "Thin execution microservice for the LeafMachine2 image analysis pipeline. "
        "Receives jobs from a Laravel application, runs the pipeline, and reports "
        "progress and results back via HTTP callbacks. "
        "Laravel + MySQL own all job state; this service is stateless."
    ),
    version="1.0.0",
    # Restrict Swagger UI to internal use — consider disabling in production
    # by setting docs_url=None, redoc_url=None if the port is publicly reachable.
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health.router)
app.include_router(jobs.router)


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
    )
