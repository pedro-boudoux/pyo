"""
Tier-2 tests for app/services/ingest.py::embed_and_store_track.

All DB and network seams are monkeypatched — no Postgres or Last.fm required.

Note: the listener cap was removed in 21dc30b ("Remove underground listener
cap") — all tracks are now eligible regardless of listener count, so these
tests assert that popular tracks are returned/embedded rather than filtered.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

from app.services.embeddings import make_track_id
from tests.conftest import make_fake_get_cursor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vector(val=0.5, dim=300):
    """Return a list[float] embedding of length `dim`."""
    return [val] * dim


def _make_two_step_cursor(first_rows, second_rows=None):
    """
    Returns a get_cursor replacement whose successive calls yield different
    FakeCursor instances.

    embed_and_store_track opens get_cursor twice:
      1st call  — SELECT (cache lookup)
      2nd call  — INSERT ... ON CONFLICT (only on cache miss)

    Pass first_rows for the SELECT result; second_rows for the INSERT (usually
    empty since INSERT doesn't return rows in this path).
    """
    call_count = [0]

    @contextmanager
    def _fake():
        call_count[0] += 1
        if call_count[0] == 1:
            from tests.conftest import FakeCursor
            yield FakeCursor(first_rows)
        else:
            from tests.conftest import FakeCursor
            yield FakeCursor(second_rows or [])

    return _fake


# ---------------------------------------------------------------------------
# Cache-hit: embedding present → return stored row, zero Last.fm calls.
# ---------------------------------------------------------------------------

class TestEmbedAndStoreTrackCacheHit:
    """
    When the songs table already has an embedding for this track, the function
    must return the stored data immediately — no Last.fm calls whatsoever.
    """

    def _patch_all_lastfm(self, monkeypatch):
        """Patch every Last.fm function used by embed_and_store_track."""
        for fn in (
            "app.services.ingest.lastfm.get_track_info",
            "app.services.ingest.lastfm.get_artist_top_tags",
            "app.services.ingest.lastfm.get_track_top_tags",
            "app.services.ingest.lastfm.get_similar_artists",
            "app.services.ingest.lastfm.blend_tags",
        ):
            mock = MagicMock(side_effect=AssertionError(f"{fn} must NOT be called on cache hit"))
            monkeypatch.setattr(fn, mock)
        return monkeypatch

    def test_cache_hit_returns_stored_row(self, monkeypatch):
        artist, name = "Burial", "Archangel"
        tid = make_track_id(artist, name)
        stored_vec = make_vector(0.7)

        cached_row = {
            "track_id": tid,
            "name": name,
            "artist": artist,
            "listeners": 10_000,
            "image": "https://example.com/cover.jpg",
            "embedding": stored_vec,
            "tags": {"dubstep": 100},
        }

        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            make_fake_get_cursor([cached_row]),
        )
        self._patch_all_lastfm(monkeypatch)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["track_id"] == tid
        assert result["name"] == name
        assert result["artist"] == artist
        assert result["listeners"] == 10_000
        assert result["image"] == "https://example.com/cover.jpg"
        assert len(result["embedding"]) == 300

    def test_cache_hit_embedding_values_are_floats(self, monkeypatch):
        """embedding list items must be plain Python floats (not numpy scalars)."""
        artist, name = "Boards of Canada", "Roygbiv"
        tid = make_track_id(artist, name)
        stored_vec = make_vector(0.3)

        cached_row = {
            "track_id": tid, "name": name, "artist": artist,
            "listeners": 5_000, "image": None, "embedding": stored_vec, "tags": {"idm": 80},
        }
        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            make_fake_get_cursor([cached_row]),
        )
        self._patch_all_lastfm(monkeypatch)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert all(isinstance(v, float) for v in result["embedding"])

    def test_cache_hit_high_listeners_still_returns_row(self, monkeypatch):
        """
        No listener cap anymore: a very popular cached track is returned, not
        filtered. (Previously listeners >= 500k returned None.)
        """
        artist, name = "Ed Sheeran", "Shape of You"
        tid = make_track_id(artist, name)
        stored_vec = make_vector(0.5)

        cached_row = {
            "track_id": tid, "name": name, "artist": artist,
            "listeners": 9_999_999,
            "image": None, "embedding": stored_vec, "tags": {"pop": 100},
        }
        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            make_fake_get_cursor([cached_row]),
        )
        self._patch_all_lastfm(monkeypatch)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["track_id"] == tid
        assert result["listeners"] == 9_999_999

    def test_cache_hit_null_listeners_returns_row(self, monkeypatch):
        """listeners=None is returned as-is (no filtering)."""
        artist, name = "Actress", "Hubble"
        tid = make_track_id(artist, name)
        stored_vec = make_vector(0.4)

        cached_row = {
            "track_id": tid, "name": name, "artist": artist,
            "listeners": None,
            "image": None, "embedding": stored_vec, "tags": {"techno": 50},
        }
        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            make_fake_get_cursor([cached_row]),
        )
        self._patch_all_lastfm(monkeypatch)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["listeners"] is None


# ---------------------------------------------------------------------------
# Cache miss: SELECT → None, run the full pipeline regardless of listener count.
# ---------------------------------------------------------------------------

class TestEmbedAndStoreTrackCacheMiss:
    """
    No stored embedding → runs the Last.fm pipeline, stores the result,
    returns a dict with the expected keys.
    """

    def _setup_new_track(self, monkeypatch, *, listeners=10_000):
        """Patch all seams for a successful cache-miss path."""
        artist, name = "Autechre", "Gantz Graf"
        tid = make_track_id(artist, name)
        fake_vec = make_vector(0.6)

        # Two get_cursor calls: SELECT → None, INSERT → no rows back
        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            _make_two_step_cursor(first_rows=[], second_rows=[]),
        )

        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_track_info",
            MagicMock(return_value={"listeners": listeners, "playcount": 1000, "tags": ["idm"]}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_artist_top_tags",
            MagicMock(return_value={"idm": 80, "electronic": 60}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_track_top_tags",
            MagicMock(return_value={"idm": 100, "experimental": 40}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_similar_artists",
            MagicMock(return_value=[{"artist": "Aphex Twin", "match": 0.9}]),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.blend_tags",
            MagicMock(return_value={"idm": 100, "electronic": 40}),
        )
        monkeypatch.setattr(
            "app.services.ingest.embeddings.build_tag_vector",
            MagicMock(return_value=fake_vec),
        )
        monkeypatch.setattr(
            "app.services.ingest.get_cover_url",
            MagicMock(return_value="https://cdn.example.com/autechre.jpg"),
        )

        return artist, name, tid, fake_vec

    def test_returns_dict_with_expected_keys(self, monkeypatch):
        artist, name, tid, fake_vec = self._setup_new_track(monkeypatch)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["track_id"] == tid
        assert result["name"] == name
        assert result["artist"] == artist
        assert result["listeners"] == 10_000
        assert result["image"] == "https://cdn.example.com/autechre.jpg"
        assert result["embedding"] == fake_vec

    def test_lastfm_pipeline_called(self, monkeypatch):
        """Verify the full blended-tag pipeline is invoked on a cache miss."""
        artist, name, _, _ = self._setup_new_track(monkeypatch)

        import app.services.ingest as ingest_mod
        from app.services.ingest import embed_and_store_track
        embed_and_store_track(artist, name)

        ingest_mod.lastfm.get_track_info.assert_called_once_with(artist, name)
        ingest_mod.lastfm.get_artist_top_tags.assert_called()
        ingest_mod.lastfm.get_track_top_tags.assert_called_once_with(artist, name)
        ingest_mod.lastfm.get_similar_artists.assert_called_once_with(artist)
        ingest_mod.lastfm.blend_tags.assert_called_once()
        ingest_mod.embeddings.build_tag_vector.assert_called_once()
        ingest_mod.get_cover_url.assert_called_once_with(artist, name)

    def test_cache_miss_high_listeners_still_runs_pipeline(self, monkeypatch):
        """
        No listener cap anymore: even a mainstream track (millions of listeners)
        runs the full pipeline and is embedded/stored. (Previously listeners >=
        500k short-circuited to None.)
        """
        artist, name, tid, fake_vec = self._setup_new_track(monkeypatch, listeners=5_000_000)

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["track_id"] == tid
        assert result["listeners"] == 5_000_000
        assert result["embedding"] == fake_vec

    def test_cache_miss_row_exists_but_embedding_none_reruns_pipeline(self, monkeypatch):
        """
        Row in DB but embedding IS NULL → treated like a cache miss (the embedding
        guard `if row and row["embedding"] is not None` is False).
        """
        artist, name = "Flying Lotus", "Zodiac Shit"
        tid = make_track_id(artist, name)
        fake_vec = make_vector(0.5)

        # Row exists but embedding is None
        partial_row = {
            "track_id": tid, "name": name, "artist": artist,
            "listeners": 20_000, "image": None, "embedding": None,
        }
        monkeypatch.setattr(
            "app.services.ingest.get_cursor",
            _make_two_step_cursor(first_rows=[partial_row], second_rows=[]),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_track_info",
            MagicMock(return_value={"listeners": 20_000, "playcount": 0, "tags": []}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_artist_top_tags",
            MagicMock(return_value={"jazz": 50}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_track_top_tags",
            MagicMock(return_value={"jazz": 80}),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.get_similar_artists",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "app.services.ingest.lastfm.blend_tags",
            MagicMock(return_value={"jazz": 80}),
        )
        monkeypatch.setattr(
            "app.services.ingest.embeddings.build_tag_vector",
            MagicMock(return_value=fake_vec),
        )
        monkeypatch.setattr(
            "app.services.ingest.get_cover_url",
            MagicMock(return_value=None),
        )

        from app.services.ingest import embed_and_store_track
        result = embed_and_store_track(artist, name)

        assert result is not None
        assert result["track_id"] == tid
        assert result["embedding"] == fake_vec
