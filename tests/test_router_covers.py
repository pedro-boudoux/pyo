"""
Tier-2 router tests for GET /songs/{track_id}/cover.

Exercises the DB-cache logic end-to-end through the FastAPI handler:
  - unknown track                → 404
  - real image already stored    → stored url, checked=true, providers NOT called
  - definitive miss cached       → url=null, checked=true, providers NOT called
  - cache miss + found           → resolves, persists image + cover_checked_at
  - cache miss + no cover        → persists a definitive miss (cover_checked_at)
  - cache miss + full outage     → no persist, checked=false (retried later)

All DB seams and the cover service are monkeypatched; init_db is silenced.
"""

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.services.covers import CoversUnavailable, LASTFM_PLACEHOLDER_HASH


@pytest.fixture(autouse=True)
def no_init_db(monkeypatch):
    monkeypatch.setattr("app.db.init_db", lambda: None)


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def patch_cursor(monkeypatch, rows):
    """Patch songs.get_cursor with a recording cursor; returns the SQL log.

    Every `with get_cursor()` block in the handler shares `rows` (for the SELECT)
    and appends each execute() to the returned log (so UPDATEs can be asserted).
    """
    log: list[tuple] = []

    @contextmanager
    def _cursor():
        class _Cur:
            def execute(self, sql, params=None):
                log.append((sql, params))

            def fetchone(self):
                return rows[0] if rows else None

            def fetchall(self):
                return list(rows)

        yield _Cur()

    monkeypatch.setattr("app.routers.songs.get_cursor", _cursor)
    return log


def _persisted_image(log) -> bool:
    return any("UPDATE songs SET image" in sql for sql, _ in log)


def _persisted_miss(log) -> bool:
    return any(
        "UPDATE songs SET cover_checked_at" in sql and "image" not in sql
        for sql, _ in log
    )


def _never_resolves(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("cover providers must not be called on a cache hit")

    monkeypatch.setattr("app.routers.songs.covers.resolve_cover", boom)


class TestCoverRouter:
    def test_unknown_track_returns_404(self, monkeypatch, client):
        patch_cursor(monkeypatch, [])
        resp = client.get("/songs/nope/cover")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_real_image_served_without_calling_providers(self, monkeypatch, client):
        patch_cursor(monkeypatch, [{
            "name": "Song", "artist": "Artist",
            "image": "https://cover.example.com/real.jpg",
            "cover_checked_at": None,
        }])
        _never_resolves(monkeypatch)

        resp = client.get("/songs/abc/cover")
        assert resp.status_code == 200
        assert resp.json() == {
            "url": "https://cover.example.com/real.jpg",
            "checked": True,
        }

    def test_cached_definitive_miss_returns_null_checked(self, monkeypatch, client):
        """image NULL + cover_checked_at set = 'looked, no cover anywhere'."""
        patch_cursor(monkeypatch, [{
            "name": "Song", "artist": "Artist",
            "image": None,
            "cover_checked_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }])
        _never_resolves(monkeypatch)
        resp = client.get("/songs/abc/cover")
        assert resp.json() == {"url": None, "checked": True}

    def test_cached_placeholder_with_checked_at_is_a_definitive_miss(self, monkeypatch, client):
        """A leftover Last.fm placeholder counts as no cover once checked."""
        patch_cursor(monkeypatch, [{
            "name": "Song", "artist": "Artist",
            "image": f"https://lastfm.example.com/{LASTFM_PLACEHOLDER_HASH}.png",
            "cover_checked_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }])
        _never_resolves(monkeypatch)
        resp = client.get("/songs/abc/cover")
        assert resp.json() == {"url": None, "checked": True}

    def test_cache_miss_resolves_and_persists(self, monkeypatch, client):
        log = patch_cursor(monkeypatch, [{
            "name": "Roygbiv", "artist": "Boards of Canada",
            "image": None, "cover_checked_at": None,
        }])
        monkeypatch.setattr(
            "app.routers.songs.covers.resolve_cover",
            lambda artist, name: "https://cover.example.com/found.jpg",
        )
        resp = client.get("/songs/abc/cover")
        assert resp.json() == {
            "url": "https://cover.example.com/found.jpg",
            "checked": True,
        }
        assert _persisted_image(log), "a cache miss should persist the resolved cover"

    def test_cache_miss_no_cover_persists_definitive_miss(self, monkeypatch, client):
        log = patch_cursor(monkeypatch, [{
            "name": "Obscure", "artist": "Nobody",
            "image": None, "cover_checked_at": None,
        }])
        monkeypatch.setattr(
            "app.routers.songs.covers.resolve_cover",
            lambda artist, name: None,
        )
        resp = client.get("/songs/abc/cover")
        assert resp.json() == {"url": None, "checked": True}
        # A definitive miss is persisted so the track is never re-looked-up.
        assert _persisted_miss(log)

    def test_cache_miss_outage_not_persisted(self, monkeypatch, client):
        log = patch_cursor(monkeypatch, [{
            "name": "Song", "artist": "Artist",
            "image": None, "cover_checked_at": None,
        }])

        def unavailable(artist, name):
            raise CoversUnavailable("all providers down")

        monkeypatch.setattr("app.routers.songs.covers.resolve_cover", unavailable)

        resp = client.get("/songs/abc/cover")
        assert resp.json() == {"url": None, "checked": False}
        # Not a definitive answer → must NOT be persisted, so it retries later.
        assert not _persisted_image(log)
        assert not _persisted_miss(log)
