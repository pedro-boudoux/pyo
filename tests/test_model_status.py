from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from jobs import model_status


def _cursor_with(rows):
    queue = list(rows)

    @contextmanager
    def cursor():
        class Cursor:
            current = None

            def execute(self, sql, params=None):
                self.current = queue.pop(0)

            def fetchone(self):
                return self.current

        yield Cursor()

    return cursor


def test_status_summarizes_active_run_and_candidate_coverage(monkeypatch):
    now = datetime.now(timezone.utc)
    active = {
        "id": 7,
        "model": "weighted_deepwalk_skipgram",
        "published_at": now - timedelta(hours=12),
        "age_hours": 12,
        "node_count": 20000,
        "edge_count": 80000,
        "edge_cutoff": now - timedelta(hours=13),
        "song_count": 100,
        "candidate_count": 100,
        "fallback_count": 4,
    }
    latest = {
        "id": 7,
        "status": "active",
        "started_at": now - timedelta(hours=13),
        "finished_at": now - timedelta(hours=12),
        "failure_details": None,
    }
    monkeypatch.setattr(model_status, "get_cursor", _cursor_with([active, latest, None]))

    result = model_status.model_status(stale_after_hours=192)

    assert result["healthy"] is True
    assert result["active"]["run_id"] == 7
    assert result["active"]["candidate"]["coverage"] == 1.0
    assert result["active"]["candidate"]["fallback_rate"] == 0.04


def test_status_alerts_on_stale_model_and_latest_failure(monkeypatch):
    now = datetime.now(timezone.utc)
    active = {
        "id": 7,
        "model": "model",
        "published_at": now - timedelta(hours=250),
        "age_hours": 250,
        "node_count": 1,
        "edge_count": 1,
        "edge_cutoff": now,
        "song_count": 1,
        "candidate_count": 1,
        "fallback_count": 0,
    }
    failure = {
        "id": 8,
        "status": "failed",
        "started_at": now - timedelta(hours=1),
        "finished_at": now - timedelta(minutes=30),
        "failure_details": "DATABASE_URL=postgres://user:pass@private/db token=abc",
    }
    monkeypatch.setattr(
        model_status, "get_cursor", _cursor_with([active, failure, failure])
    )

    result = model_status.model_status(stale_after_hours=192)

    assert result["healthy"] is False
    assert result["alerts"] == ["active_model_stale", "latest_run_failed"]
    assert "user:pass" not in result["latest_attempt"]["failure"]
    assert "abc" not in result["latest_attempt"]["failure"]


def test_status_alerts_when_no_active_model(monkeypatch):
    monkeypatch.setattr(model_status, "get_cursor", _cursor_with([None, None, None]))
    result = model_status.model_status()
    assert result["healthy"] is False
    assert result["alerts"] == ["no_active_model"]


def test_latest_attempt_is_ordered_by_completion_time(monkeypatch):
    statements = []

    @contextmanager
    def cursor():
        class Cursor:
            def execute(self, sql, params=None):
                statements.append(" ".join(sql.split()))

            def fetchone(self):
                return None

        yield Cursor()

    monkeypatch.setattr(model_status, "get_cursor", cursor)
    model_status.model_status()

    assert "ORDER BY COALESCE(finished_at, started_at) DESC" in statements[1]
