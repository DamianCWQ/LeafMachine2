from api.config import settings


def callback_headers() -> dict[str, str]:
    if settings.LARAVEL_CALLBACK_TOKEN:
        return {"Authorization": f"Bearer {settings.LARAVEL_CALLBACK_TOKEN}"}
    return {}
