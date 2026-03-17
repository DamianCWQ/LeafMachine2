from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path relative to this file so it works regardless of where
# uvicorn is launched from.
_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service auth ──────────────────────────────────────────────────────────
    # Authentication is disabled for this service. This value is optional and
    # retained only for backwards-compatible environment files.
    LM2_API_KEY: str = ""

    # ── LeafMachine2 paths ───────────────────────────────────────────────────
    # Root of the LeafMachine2 repository on this server.
    # Defaults to the parent directory of api/ (i.e. the repo root).
    LM2_HOME: str = str(Path(__file__).resolve().parent.parent)

    # Base directory where Laravel uploads land. All uploaded files are
    # stored under  LM2_UPLOAD_DIR/<job_id>/  and never outside this tree.
    LM2_UPLOAD_DIR: str

    # Base directory for pipeline outputs. All outputs are written under
    # LM2_OUTPUT_DIR/<job_id>/  and never outside this tree.
    LM2_OUTPUT_DIR: str

    # ── Network ───────────────────────────────────────────────────────────────
    PORT: int = 5000

    # ── Laravel callback ─────────────────────────────────────────────────────
    # Full base URL for the internal status-update endpoint on the Laravel
    # server, e.g. https://myapp.com/api/internal/leafmachine/jobs
    # Leave empty to disable callbacks (useful for local dev).
    LARAVEL_CALLBACK_URL: str = ""

    # Bearer token that the Python service uses when posting callbacks to
    # Laravel. Unused when callback auth is disabled.
    LARAVEL_CALLBACK_TOKEN: str = ""


settings = Settings()
