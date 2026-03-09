from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
def health() -> dict:
    """Returns service liveness information.

    No authentication required — intended for load-balancer / uptime probes.
    """
    return {
        "status": "ok",
        "service": "leafmachine2-execution-service",
        "version": "1.0",
    }
