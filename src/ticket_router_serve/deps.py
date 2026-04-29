"""FastAPI dependencies — API key authentication."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ticket_router_base.config import API_KEYS
from ticket_router_base.schemas import ErrorResponse

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    key: str | None = Security(api_key_header),
) -> str:
    """Validate X-API-Key header against configured API keys.

    Raises HTTPException 401 if missing or invalid.
    Returns the validated API key string.
    """
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(error="Invalid or missing API key").model_dump(),
        )
    if key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(error="Invalid or missing API key").model_dump(),
        )
    return key
