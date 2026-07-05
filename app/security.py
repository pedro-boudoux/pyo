"""Small authentication guards for non-public operational endpoints."""

import secrets

from fastapi import HTTPException, Request

from app.config import MAINTENANCE_API_KEY


def require_maintenance_key(request: Request) -> None:
    """Allow bulk maintenance only when a deployment key is configured."""
    if not MAINTENANCE_API_KEY:
        raise HTTPException(503, "Maintenance endpoints are disabled")

    supplied = request.headers.get("x-maintenance-key", "")
    if not supplied or not secrets.compare_digest(supplied, MAINTENANCE_API_KEY):
        raise HTTPException(403, "Invalid maintenance key")
