"""Authentication guard for public maintenance routes."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import security


def _request(key: str | None = None) -> Request:
    headers = []
    if key is not None:
        headers.append((b"x-maintenance-key", key.encode()))
    return Request({"type": "http", "headers": headers})


def test_maintenance_disabled_without_deployment_key(monkeypatch):
    monkeypatch.setattr(security, "MAINTENANCE_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        security.require_maintenance_key(_request())
    assert exc.value.status_code == 503


def test_maintenance_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(security, "MAINTENANCE_API_KEY", "correct")
    with pytest.raises(HTTPException) as exc:
        security.require_maintenance_key(_request("wrong"))
    assert exc.value.status_code == 403


def test_maintenance_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(security, "MAINTENANCE_API_KEY", "correct")
    assert security.require_maintenance_key(_request("correct")) is None


def test_real_maintenance_route_stops_before_database_work(monkeypatch):
    from app.main import app

    monkeypatch.setattr(security, "MAINTENANCE_API_KEY", None)
    response = TestClient(app).post("/songs/backfill-semantic-embeddings")
    assert response.status_code == 503
    assert response.json()["detail"] == "Maintenance endpoints are disabled"
