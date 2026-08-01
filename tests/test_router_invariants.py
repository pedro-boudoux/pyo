"""
Tier-2 router-level invariant tests via FastAPI TestClient.

Tests cheap-to-mock HTTP invariants across multiple routers:
  - POST /graph/seed with unknown track_id → 404
  - POST /feedback with invalid action string → 400

All DB seams are monkeypatched; init_db is silenced to avoid Postgres on startup.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_fake_get_cursor


# ---------------------------------------------------------------------------
# Shared fixture: silence init_db + build client
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_init_db(monkeypatch):
    """Prevent the startup event from trying to connect to Postgres."""
    monkeypatch.setattr("app.db.init_db", lambda: None)


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /graph/seed — 404 on unknown track_id
# ---------------------------------------------------------------------------

class TestGraphSeedRouter:
    def test_unknown_track_id_returns_404(self, monkeypatch, client):
        """
        /graph/seed with a track_id not present in the songs table → 404.
        The handler does a SELECT before any embedding work.
        """
        monkeypatch.setattr(
            "app.routers.graph.get_cursor",
            make_fake_get_cursor([]),
        )
        response = client.post("/graph/seed", json={"track_id": "nonexistenttrack01"})
        assert response.status_code == 404

    def test_unknown_track_id_error_message(self, monkeypatch, client):
        """404 response includes a human-readable detail field."""
        monkeypatch.setattr(
            "app.routers.graph.get_cursor",
            make_fake_get_cursor([]),
        )
        response = client.post("/graph/seed", json={"track_id": "nope"})
        assert "not found" in response.json()["detail"].lower()

    def test_missing_track_id_field_returns_422(self, client):
        """Pydantic validation: missing required track_id → 422 Unprocessable Entity."""
        response = client.post("/graph/seed", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /feedback — 400 on invalid action
# ---------------------------------------------------------------------------

class TestFeedbackRouter:
    def test_invalid_action_returns_400(self, monkeypatch, client):
        """
        action='like' (not 'accept' or 'reject') → 400 Bad Request.
        The check happens before any DB call in the handler.
        """
        # Patch get_cursor so the songs SELECT doesn't fail (though we expect
        # the action guard to fire first, before any DB access).
        monkeypatch.setattr(
            "app.routers.feedback.get_cursor",
            make_fake_get_cursor([{"id": 1}]),
        )
        response = client.post(
            "/feedback",
            json={"track_id": "sometrackid12345678", "action": "like"},
        )
        assert response.status_code == 400

    def test_invalid_action_error_message(self, monkeypatch, client):
        """400 response contains a message about valid actions."""
        monkeypatch.setattr(
            "app.routers.feedback.get_cursor",
            make_fake_get_cursor([{"id": 1}]),
        )
        response = client.post(
            "/feedback",
            json={"track_id": "sometrackid12345678", "action": "dislike"},
        )
        detail = response.json()["detail"]
        assert "accept" in detail.lower() or "reject" in detail.lower()

    def test_empty_action_returns_400(self, monkeypatch, client):
        """An empty string action is also invalid."""
        monkeypatch.setattr(
            "app.routers.feedback.get_cursor",
            make_fake_get_cursor([{"id": 1}]),
        )
        response = client.post(
            "/feedback",
            json={"track_id": "sometrackid12345678", "action": ""},
        )
        assert response.status_code == 400

    def test_valid_action_accept_not_rejected_by_guard(self, monkeypatch, client):
        """
        'accept' passes the guard. The handler then hits the DB — we mock it
        returning the track row so the handler can proceed (it won't reach
        ann_search since we mock steering and ann_search too, but we only need
        to confirm the 400 guard is NOT triggered).
        """
        from contextlib import contextmanager
        from tests.conftest import FakeCursor

        # Multi-step cursor: songs SELECT → found, feedback INSERT → ok,
        # graph_nodes INSERT → ok, graph_edges SELECT → no parent, songs embedding SELECT → None
        # We use a call-counting cursor that returns different rows per execute().
        call_count = [0]
        rows_sequence = [
            [{"id": 1}],   # songs SELECT — track found
            [],            # feedback INSERT
            [],            # graph_nodes INSERT
            [],            # graph_edges SELECT (no parent)
            [],            # songs embedding SELECT
        ]

        @contextmanager
        def multi_cursor():
            class _Multi:
                def execute(self, sql, params=None):
                    pass
                def fetchone(self):
                    idx = call_count[0]
                    call_count[0] += 1
                    if idx < len(rows_sequence):
                        r = rows_sequence[idx]
                        return r[0] if r else None
                    return None
                def fetchall(self):
                    return []
            yield _Multi()

        monkeypatch.setattr("app.routers.feedback.get_cursor", multi_cursor)
        # Patch steering and ann_search so they don't try to hit the DB
        monkeypatch.setattr(
            "app.routers.feedback.steering.apply_steering",
            lambda emb, tid: emb,
        )
        monkeypatch.setattr(
            "app.routers.feedback.embeddings.ann_search",
            lambda *a, **kw: [],
        )

        response = client.post(
            "/feedback",
            json={"track_id": "sometrackid12345678", "action": "accept"},
        )
        # Should NOT be 400 (the guard passes)
        assert response.status_code != 400

    def test_valid_action_reject_not_rejected_by_guard(self, monkeypatch, client):
        """A parent-scoped 'reject' passes the action guard."""
        from contextlib import contextmanager

        call_count = [0]
        rows_sequence = [
            [{"id": 1}],  # rejected song SELECT
            [{"id": 2}],  # source song SELECT
        ]

        @contextmanager
        def multi_cursor():
            class _Multi:
                def execute(self, sql, params=None):
                    pass
                def fetchone(self):
                    idx = call_count[0]
                    call_count[0] += 1
                    if idx < len(rows_sequence):
                        r = rows_sequence[idx]
                        return r[0] if r else None
                    return None
                def fetchall(self):
                    return []
            yield _Multi()

        monkeypatch.setattr("app.routers.feedback.get_cursor", multi_cursor)

        response = client.post(
            "/feedback",
            json={
                "track_id": "sometrackid12345678",
                "source_track_id": "source-track-id",
                "action": "reject",
            },
        )
        assert response.status_code != 400

    def test_reject_requires_source_track_id(self, client):
        response = client.post(
            "/feedback",
            json={"track_id": "sometrackid12345678", "action": "reject"},
        )
        assert response.status_code == 400
        assert "source_track_id" in response.json()["detail"]

    def test_missing_fields_returns_422(self, client):
        """Pydantic validation: empty body → 422."""
        response = client.post("/feedback", json={})
        assert response.status_code == 422
