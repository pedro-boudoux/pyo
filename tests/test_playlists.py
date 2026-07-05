"""
Tests for playlist neighbor selection — specifically that the artist blacklist is
enforced in Tree/Linear playlists, not just MMR recommendations (issue #23).

`find_neighbors` is the shared selection helper both /playlists/linear and
/playlists/tree call, so filtering there covers both modes.
"""
import pytest

from app.routers import playlists
from app.services import blacklist


def _row(track_id, artist, listeners=1000, similarity=0.9):
    return {
        "track_id": track_id,
        "name": f"song-{track_id}",
        "artist": artist,
        "listeners": listeners,
        "similarity": similarity,
        "embedding": [0.0] * 384,
    }


@pytest.fixture
def no_mandatory(monkeypatch):
    monkeypatch.setattr(blacklist, "MANDATORY", set())


class TestFindNeighborsBlacklist:
    def test_per_request_blocked_artist_dropped(self, monkeypatch, no_mandatory):
        pool = [_row("a", "Drake"), _row("b", "Boards of Canada"), _row("c", "Drake")]
        monkeypatch.setattr(playlists.emb_service, "ann_search", lambda *a, **k: list(pool))

        out = playlists.find_neighbors(
            cursor=None, embedding=[0.0] * 384, exclude_ids=set(), k=5, niche=False,
            blocked_artists=blacklist.normalize(["Drake"]),
        )
        assert [r["track_id"] for r in out] == ["b"]

    def test_mandatory_blacklist_applied_with_no_request_artists(self, monkeypatch):
        # The original bug: Tree/Linear never filtered, so the mandatory env set
        # leaked through even with an empty per-request list.
        monkeypatch.setattr(blacklist, "MANDATORY", {"drake"})
        pool = [_row("a", "Drake"), _row("b", "Aphex Twin")]
        monkeypatch.setattr(playlists.emb_service, "ann_search", lambda *a, **k: list(pool))

        out = playlists.find_neighbors(
            cursor=None, embedding=[0.0] * 384, exclude_ids=set(), k=5, niche=False,
        )
        assert [r["track_id"] for r in out] == ["b"]

    def test_credit_aware_in_playlists(self, monkeypatch, no_mandatory):
        pool = [_row("a", "Rihanna feat. Drake"), _row("b", "Burial")]
        monkeypatch.setattr(playlists.emb_service, "ann_search", lambda *a, **k: list(pool))

        out = playlists.find_neighbors(
            cursor=None, embedding=[0.0] * 384, exclude_ids=set(), k=5, niche=False,
            blocked_artists=blacklist.normalize(["Drake"]),
        )
        assert [r["track_id"] for r in out] == ["b"]

    def test_niche_path_filters_blocked(self, monkeypatch, no_mandatory):
        pool = [_row("a", "Drake", listeners=50), _row("b", "Four Tet", listeners=80)]
        # ann_search is called once per niche threshold; return the same pool each
        # time but excluded ids are tracked by the caller so nothing repeats.
        monkeypatch.setattr(
            playlists.emb_service, "ann_search",
            lambda *a, **k: [r for r in pool if r["track_id"] not in k.get("exclude_ids", set())],
        )

        out = playlists.find_neighbors(
            cursor=None, embedding=[0.0] * 384, exclude_ids=set(), k=5, niche=True,
            blocked_artists=blacklist.normalize(["Drake"]),
        )
        assert [r["track_id"] for r in out] == ["b"]

    def test_nothing_blocked_when_no_list(self, monkeypatch, no_mandatory):
        pool = [_row("a", "Drake"), _row("b", "Four Tet")]
        monkeypatch.setattr(playlists.emb_service, "ann_search", lambda *a, **k: list(pool))

        out = playlists.find_neighbors(
            cursor=None, embedding=[0.0] * 384, exclude_ids=set(), k=5, niche=False,
        )
        assert {r["track_id"] for r in out} == {"a", "b"}