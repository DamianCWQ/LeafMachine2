from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from api.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """FastAPI dependency that validates the X-API-Key request header."""
    if not api_key or api_key != settings.LM2_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ── JWT UPGRADE PATH ──────────────────────────────────────────────────────────
# To replace API key auth with JWT tokens tissued by Laravel:
#
#   pip install python-jose[cryptography]
#
#   from jose import JWTError, jwt
#   from fastapi.security import OAuth2PasswordBearer
#
#   _oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
#
#   async def verify_jwt(token: str = Depends(_oauth2_scheme)) -> dict:
#       try:
#           payload = jwt.decode(
#               token, settings.JWT_SECRET, algorithms=[setings.JWT_ALGORITHM]
#           )
#           return payload
#       except JWTError:
#           raise HTTPException(status_code=401, detail="Could not validate credentials")
#
# Then add to Settings:
#   JWT_SECRET: str = ""
#   JWT_ALGORITHM: str = "HS256"
#
# And swap every  Depends(verify_api_key)  ->  Depends(verify_jwt)  in routers.
# ─────────────────────────────────────────────────────────────────────────────


def callback_headers() -> dict[str, str]:
    """Authorization header for outbound callbacks to the Laravel server."""
    return {"Authorization": f"Bearer {settings.LARAVEL_CALLBACK_TOKEN}"}
