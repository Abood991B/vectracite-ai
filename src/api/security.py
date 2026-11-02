"""Authentication dependencies for the private document API."""

from typing import Annotated
import hmac

from fastapi import Header, HTTPException

from src.config import settings


def request_api_key(x_api_key: str | None, authorization: str | None) -> str:
    """Extract an API key from the dedicated header or a Bearer token."""
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def matches_secret(candidate: str, expected: str) -> bool:
    """Compare credentials without leaking prefix or timing information."""
    return bool(candidate and expected) and hmac.compare_digest(
        candidate.encode("utf-8"), expected.encode("utf-8")
    )


def _unauthorized(detail: str, status_code: int = 401) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _configured_keys() -> tuple[str, str]:
    return (
        settings.API_READ_KEY.get_secret_value(),
        settings.API_ADMIN_KEY.get_secret_value(),
    )


async def require_read_access(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Allow configured reader or administrator credentials for private reads."""
    if not settings.AUTH_REQUIRED:
        return
    read_key, admin_key = _configured_keys()
    if not read_key or not admin_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    candidate = request_api_key(x_api_key, authorization)
    if not (matches_secret(candidate, read_key) or matches_secret(candidate, admin_key)):
        raise _unauthorized("Valid API credentials are required.")


async def require_admin_access(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require the administrator credential for state-changing operations."""
    if not settings.AUTH_REQUIRED:
        return
    _, admin_key = _configured_keys()
    if not admin_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    candidate = request_api_key(x_api_key, authorization)
    if not matches_secret(candidate, admin_key):
        raise _unauthorized("Administrator credentials are required.", status_code=403)
