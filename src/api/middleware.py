"""
API middleware: API key auth, structured request logging.
"""

import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Skip auth for these paths
SKIP_AUTH_PATHS = ("/", "/health", "/docs", "/redoc", "/openapi.json")


def get_api_key_from_request(request: Request) -> Optional[str]:
    """Extract API key from X-API-Key header or Authorization: Bearer <key>."""
    key = request.headers.get("X-API-Key")
    if key:
        return key.strip()
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_api_key(request: Request) -> bool:
    """Return True if request is allowed (no API_KEY set, or valid key provided)."""
    if not API_KEY or not API_KEY.strip():
        return True
    client_key = get_api_key_from_request(request)
    return client_key == API_KEY


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests with 401 when API_KEY is set and request does not provide a valid key."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in SKIP_AUTH_PATHS or request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            return await call_next(request)
        if not require_api_key(request):
            return Response(content='{"detail":"Missing or invalid API key"}', status_code=401, media_type="application/json")
        return await call_next(request)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with method, path, status_code, latency_ms in a structured way."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status_code=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response
