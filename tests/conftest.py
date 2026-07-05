"""
Shared fixtures and helpers for the Discover test suite.

Tier 2 agents can reuse:
  - FakeCursor          — dict-row cursor (execute/fetchone/fetchall)
  - FakeCtxManager      — context-manager wrapper around FakeCursor
  - make_fake_get_cursor(rows) -> callable
      Returns a zero-arg callable that, when used as `with get_cursor() as cur`,
      yields a FakeCursor pre-loaded with `rows` (list[dict]).
  - fake_get_cursor fixture
      A pytest fixture that injects make_fake_get_cursor into tests via
      monkeypatching. Usage in a test:

          def test_something(monkeypatch, fake_get_cursor):
              monkeypatch.setattr("app.services.embeddings.get_cursor",
                                  fake_get_cursor([{"id": 1, "tag": "rock"}]))
"""

import os

# Turn the slowapi rate limiter off for the whole suite before app.config is
# imported, so router tests that fire many requests aren't throttled (issue #20).
# A dedicated test (test_ratelimit.py) builds its own enabled limiter to verify
# the 429 path.
os.environ["RATE_LIMIT_ENABLED"] = "false"

from contextlib import contextmanager
import pytest


class FakeCursor:
    """
    Minimal cursor that behaves like psycopg2 RealDictCursor.

    Rows are plain dicts, accessed by key: row["id"], row["tag"], etc.

    Args:
        rows: list[dict] returned by fetchall(); the first element is
              also returned by fetchone().
    """

    def __init__(self, rows: list[dict] | None = None):
        self._rows: list[dict] = rows or []
        self._executed_sql: list[str] = []
        self._executed_params: list = []

    def execute(self, sql: str, params=None):
        self._executed_sql.append(sql)
        self._executed_params.append(params)

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return list(self._rows)


class FakeCtxManager:
    """
    A context-manager that yields a FakeCursor.

    Usage:
        ctx = FakeCtxManager(rows=[{"id": 1, "tag": "rock"}])
        with ctx as cursor:
            cursor.fetchall()   # -> [{"id": 1, "tag": "rock"}]
    """

    def __init__(self, rows: list[dict] | None = None):
        self.cursor = FakeCursor(rows)

    def __enter__(self) -> FakeCursor:
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def make_fake_get_cursor(rows: list[dict] | None = None):
    """
    Returns a callable that mimics ``get_cursor`` as a context manager.

    Pass the result directly to monkeypatch.setattr:

        monkeypatch.setattr(
            "app.services.embeddings.get_cursor",
            make_fake_get_cursor([{"id": 0, "tag": "ambient"}])
        )

    Each call to the returned callable creates a fresh FakeCtxManager backed
    by the same ``rows`` list, so multiple ``with get_cursor()`` blocks in one
    code path all see the same data.
    """
    @contextmanager
    def _fake():
        yield FakeCursor(rows)

    return _fake


@pytest.fixture
def fake_get_cursor():
    """
    Fixture that exposes ``make_fake_get_cursor`` directly inside tests.

    Example usage in a test function:

        def test_build(monkeypatch, fake_get_cursor):
            vocab = [{"id": 0, "tag": "rock"}, {"id": 1, "tag": "jazz"}]
            monkeypatch.setattr(
                "app.services.embeddings.get_cursor",
                fake_get_cursor(vocab)
            )
            result = build_tag_vector({"rock": 100, "jazz": 50})
            assert result[0] == pytest.approx(1.0)
    """
    return make_fake_get_cursor
