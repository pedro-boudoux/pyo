"""
Tier-2 tests for app/routers/recommendations.py.

Covers:
  1. topup_from_lastfm — orchestration logic, fully mocked.
  2. GET /recommendations/{track_id} via TestClient — 404 on unknown id.
"""

import os
import pytest
from unittest.mock import MagicMock
from contextlib import contextmanager

from app.services.embeddings import make_track_id
from tests.conftest import make_fake_get_cursor, FakeCursor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vector(val=0.5, dim=300):
    return [val] * dim


def _seed_cursor(seed_row):
    """get_cursor replacement that returns seed_row on the first (only) call."""
    return make_fake_get_cursor([seed_row] if seed_row else [])


# ---------------------------------------------------------------------------
# topup_from_lastfm
# ---------------------------------------------------------------------------

class TestTopupFromLastfm:
    """
    topup_from_lastfm(seed_track_id, query_embedding, exclude_ids, needed)

    Seams to mock (all in app.routers.recommendations namespace):
      - get_cursor            → seed lookup
      - lastfm.get_similar_tracks
      - ingest.embed_and_store_track
      - embeddings.cosine_similarity (via embeddings.make_track_id is a pure fn)
    """

    @pytest.fixture(autouse=True)
    def _no_artist_fallback(self, monkeypatch):
        """
        Isolate the primary (track.getSimilar) path by default: the similar-artist
        cold-start fallback is a no-op unless a test explicitly enables it. Keeps
        these tests off the network when `added < needed`.
        """
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_artists",
            MagicMock(return_value=[]),
        )

    def _patch(self, monkeypatch, *, seed_row, similar_tracks, embed_results, cos_sims=None):
        """
        Wire up all seams for a topup_from_lastfm call.

        embed_results: list aligned with similar_tracks (None → embed returns None).
        cos_sims: list of floats aligned with non-None embed_results; defaults to 0.9 each.
        """
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row] if seed_row else []),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=similar_tracks),
        )

        embed_call_count = [0]
        embed_side = []
        for r in embed_results:
            embed_side.append(r)
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            MagicMock(side_effect=embed_side),
        )

        sim_iter = iter(cos_sims if cos_sims else [0.9] * len(embed_results))
        monkeypatch.setattr(
            "app.routers.recommendations.embeddings.cosine_similarity",
            MagicMock(side_effect=lambda a, b: next(sim_iter)),
        )

    # ---- seed not found ---------------------------------------------------

    def test_seed_not_found_returns_empty(self, monkeypatch):
        """Unknown seed_track_id → empty list immediately."""
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([]),
        )
        # These must not be reached
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(side_effect=AssertionError("must not be called")),
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm("nonexistent_id", make_vector(), set(), needed=3)

        assert result == []

    # ---- stops at `needed` ------------------------------------------------

    def test_stops_when_needed_satisfied(self, monkeypatch):
        """Returns exactly `needed` items even if more similar tracks are available."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [
            {"artist": "Andy Stott", "name": "Numb"},
            {"artist": "Demdike Stare", "name": "Maze Thought"},
            {"artist": "Raime", "name": "Ketu"},
            {"artist": "Emptyset", "name": "Dusk"},
        ]
        # Build song dicts for all 4
        songs = [
            {"track_id": make_track_id(s["artist"], s["name"]),
             "name": s["name"], "artist": s["artist"],
             "listeners": 5000, "image": None, "embedding": make_vector(0.5)}
            for s in similar
        ]

        self._patch(
            monkeypatch,
            seed_row=seed_row,
            similar_tracks=similar,
            embed_results=songs,
            cos_sims=[0.9, 0.85, 0.8, 0.75],
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=2,
        )

        assert len(result) == 2

    # ---- skips excluded ids -----------------------------------------------

    def test_skips_already_excluded_ids(self, monkeypatch):
        """Tracks whose track_id is in exclude_ids are silently skipped."""
        seed_row = {"name": "Shell of Light", "artist": "Burial"}
        similar = [
            {"artist": "Shackleton", "name": "Blood on My Hands"},
            {"artist": "Kode9", "name": "9 Samurai"},
        ]
        shackleton_id = make_track_id("Shackleton", "Blood on My Hands")
        kode9_id = make_track_id("Kode9", "9 Samurai")

        kode9_song = {
            "track_id": kode9_id, "name": "9 Samurai", "artist": "Kode9",
            "listeners": 7_000, "image": None, "embedding": make_vector(0.6),
        }

        # Shackleton is excluded → embed is only called for Kode9
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row]),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=similar),
        )
        embed_mock = MagicMock(return_value=kode9_song)
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            embed_mock,
        )
        monkeypatch.setattr(
            "app.routers.recommendations.embeddings.cosine_similarity",
            MagicMock(return_value=0.88),
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Shell of Light"),
            make_vector(),
            {shackleton_id},  # Shackleton excluded
            needed=2,
        )

        # Only Kode9 passes the exclusion filter
        assert len(result) == 1
        assert result[0]["track_id"] == kode9_id
        # embed was only called once (for Kode9)
        embed_mock.assert_called_once()

    # ---- skips when embed returns None ------------------------------------

    def test_skips_when_embed_returns_none(self, monkeypatch):
        """
        When embed_and_store_track returns None (too popular / can't fetch),
        the track is skipped and does not count toward `needed`.
        """
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [
            {"artist": "Too Popular", "name": "Big Hit"},
            {"artist": "Actress", "name": "Hubble"},
        ]
        actress_id = make_track_id("Actress", "Hubble")
        actress_song = {
            "track_id": actress_id, "name": "Hubble", "artist": "Actress",
            "listeners": 2_000, "image": None, "embedding": make_vector(0.55),
        }

        self._patch(
            monkeypatch,
            seed_row=seed_row,
            similar_tracks=similar,
            embed_results=[None, actress_song],
            cos_sims=[0.85],
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=1,
        )

        assert len(result) == 1
        assert result[0]["track_id"] == actress_id

    # ---- similarity is rounded -------------------------------------------

    def test_similarity_is_rounded_to_3dp(self, monkeypatch):
        """Returned similarity values are rounded to 3 decimal places."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [{"artist": "Actress", "name": "Hubble"}]
        actress_id = make_track_id("Actress", "Hubble")
        actress_song = {
            "track_id": actress_id, "name": "Hubble", "artist": "Actress",
            "listeners": 2_000, "image": None, "embedding": make_vector(0.5),
        }

        self._patch(
            monkeypatch,
            seed_row=seed_row,
            similar_tracks=similar,
            embed_results=[actress_song],
            cos_sims=[0.876543],  # should be rounded to 0.877
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=1,
        )

        assert result[0]["similarity"] == 0.877

    # ---- result dict shape ------------------------------------------------

    def test_result_dict_has_expected_keys(self, monkeypatch):
        """Each returned item contains the keys expected by the caller."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [{"artist": "Actress", "name": "Hubble"}]
        actress_id = make_track_id("Actress", "Hubble")
        actress_song = {
            "track_id": actress_id, "name": "Hubble", "artist": "Actress",
            "listeners": 3_000, "image": "https://cover.example.com/a.jpg",
            "embedding": make_vector(0.5),
        }

        self._patch(
            monkeypatch,
            seed_row=seed_row,
            similar_tracks=similar,
            embed_results=[actress_song],
            cos_sims=[0.91],
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=1,
        )

        assert len(result) == 1
        item = result[0]
        assert "track_id" in item
        assert "name" in item
        assert "artist" in item
        assert "listeners" in item
        assert "image" in item
        assert "similarity" in item

    # ---- null listeners coerced to 0 -------------------------------------

    def test_null_listeners_coerced_to_zero(self, monkeypatch):
        """listeners=None in song dict → 0 in result (explicit `or 0` in source)."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [{"artist": "Actress", "name": "Hubble"}]
        actress_id = make_track_id("Actress", "Hubble")
        actress_song = {
            "track_id": actress_id, "name": "Hubble", "artist": "Actress",
            "listeners": None, "image": None, "embedding": make_vector(0.5),
        }

        self._patch(
            monkeypatch,
            seed_row=seed_row,
            similar_tracks=similar,
            embed_results=[actress_song],
            cos_sims=[0.9],
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=1,
        )

        assert result[0]["listeners"] == 0

    # ---- no similar tracks → empty ----------------------------------------

    def test_no_similar_tracks_returns_empty(self, monkeypatch):
        seed_row = {"name": "Archangel", "artist": "Burial"}

        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row]),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            MagicMock(side_effect=AssertionError("must not be called")),
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=3,
        )

        assert result == []

    # ---- needed=0 → empty immediately ------------------------------------

    def test_needed_zero_returns_empty(self, monkeypatch):
        """needed=0 means the caller is already satisfied — skip everything."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [{"artist": "Actress", "name": "Hubble"}]

        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row]),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=similar),
        )
        embed_mock = MagicMock(return_value={
            "track_id": make_track_id("Actress", "Hubble"),
            "name": "Hubble", "artist": "Actress",
            "listeners": 2_000, "image": None, "embedding": make_vector(),
        })
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            embed_mock,
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=0,
        )

        assert result == []
        embed_mock.assert_not_called()

    # ---- deduplication within the loop -----------------------------------

    def test_does_not_yield_same_track_twice(self, monkeypatch):
        """
        The same sim_id from different similar_tracks entries is deduplicated
        via the internal `excluded` set.
        """
        seed_row = {"name": "Archangel", "artist": "Burial"}
        # Two entries that resolve to the same track_id (same artist/name, different case)
        similar = [
            {"artist": "Actress", "name": "Hubble"},
            {"artist": "actress", "name": "hubble"},  # same id after normalization
        ]
        actress_id = make_track_id("Actress", "Hubble")
        actress_song = {
            "track_id": actress_id, "name": "Hubble", "artist": "Actress",
            "listeners": 2_000, "image": None, "embedding": make_vector(0.5),
        }

        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row]),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=similar),
        )
        embed_mock = MagicMock(return_value=actress_song)
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            embed_mock,
        )
        monkeypatch.setattr(
            "app.routers.recommendations.embeddings.cosine_similarity",
            MagicMock(return_value=0.88),
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"),
            make_vector(),
            set(),
            needed=5,
        )

        # Should only appear once despite two entries in similar
        track_ids = [r["track_id"] for r in result]
        assert track_ids.count(actress_id) == 1

    # ---- cold-start fallback to similar-artist top tracks -----------------

    def test_falls_back_to_similar_artist_top_tracks(self, monkeypatch):
        """
        When the seed has no track.getSimilar, mine the seed's similar artists'
        top tracks (artist.getSimilar → artist.getTopTracks) instead.
        """
        seed_row = {"name": "Mia & Sebastian's Theme", "artist": "Justin Hurwitz"}
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([seed_row]),
        )
        # primary path is empty
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_tracks",
            MagicMock(return_value=[]),
        )
        # fallback: one similar artist with one top track (overrides autouse stub)
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_artists",
            MagicMock(return_value=[{"artist": "Tim Simonec", "match": 1.0}]),
        )
        top_tracks_mock = MagicMock(return_value=[{"artist": "Tim Simonec", "name": "Too Hip To Retire"}])
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_artist_top_tracks",
            top_tracks_mock,
        )
        song = {
            "track_id": make_track_id("Tim Simonec", "Too Hip To Retire"),
            "name": "Too Hip To Retire", "artist": "Tim Simonec",
            "listeners": 85_803, "image": None, "embedding": make_vector(0.5),
        }
        monkeypatch.setattr(
            "app.routers.recommendations.ingest.embed_and_store_track",
            MagicMock(return_value=song),
        )
        monkeypatch.setattr(
            "app.routers.recommendations.embeddings.cosine_similarity",
            MagicMock(return_value=0.79),
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Justin Hurwitz", "Mia & Sebastian's Theme"),
            make_vector(),
            set(),
            needed=3,
        )

        assert len(result) == 1
        assert result[0]["track_id"] == song["track_id"]
        top_tracks_mock.assert_called_once()

    def test_fallback_skipped_when_primary_satisfies_needed(self, monkeypatch):
        """If track.getSimilar already fills `needed`, the artist fallback is never consulted."""
        seed_row = {"name": "Archangel", "artist": "Burial"}
        similar = [{"artist": "Actress", "name": "Hubble"}]
        actress_song = {
            "track_id": make_track_id("Actress", "Hubble"), "name": "Hubble",
            "artist": "Actress", "listeners": 2_000, "image": None, "embedding": make_vector(0.5),
        }
        self._patch(
            monkeypatch, seed_row=seed_row, similar_tracks=similar,
            embed_results=[actress_song], cos_sims=[0.9],
        )
        artist_fallback = MagicMock(side_effect=AssertionError("fallback must not run"))
        monkeypatch.setattr(
            "app.routers.recommendations.lastfm.get_similar_artists", artist_fallback,
        )

        from app.routers.recommendations import topup_from_lastfm
        result = topup_from_lastfm(
            make_track_id("Burial", "Archangel"), make_vector(), set(), needed=1,
        )

        assert len(result) == 1
        artist_fallback.assert_not_called()


# ---------------------------------------------------------------------------
# Router-level: GET /recommendations/{track_id} 404 on unknown id
# ---------------------------------------------------------------------------

class TestRecommendationsRouter:
    """
    TestClient tests for the /recommendations router.
    init_db is patched to a no-op so TestClient startup doesn't hit Postgres.
    """

    @pytest.fixture(autouse=True)
    def no_init_db(self, monkeypatch):
        monkeypatch.setattr("app.db.init_db", lambda: None)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_unknown_track_id_returns_404(self, monkeypatch, client):
        """GET /recommendations/{unknown_id} → 404 when DB returns nothing."""
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([]),
        )
        response = client.get("/recommendations/deadbeefdeadbeefdeadbeef")
        assert response.status_code == 404

    def test_unknown_track_id_error_message(self, monkeypatch, client):
        """404 detail contains a helpful message."""
        monkeypatch.setattr(
            "app.routers.recommendations.get_cursor",
            make_fake_get_cursor([]),
        )
        response = client.get("/recommendations/nope")
        assert "not found" in response.json()["detail"].lower()


class TestBuildRecommendations:
    def test_local_ann_pool_has_no_listener_cap(self, monkeypatch):
        """The main ANN pool must not filter candidates by listener count."""
        from app.routers import recommendations as recs

        calls = iter([
            [{"name": "Seed", "artist": "Artist", "embedding": make_vector(dim=384)}],
            [],
        ])

        @contextmanager
        def fake_cursor():
            yield FakeCursor(next(calls))

        candidate = {
            "track_id": "candidate",
            "name": "Candidate",
            "artist": "Another Artist",
            "listeners": 10_000,
            "image": None,
            "embedding": make_vector(dim=384),
            "similarity": 0.9,
        }
        ann_search = MagicMock(return_value=[candidate])

        monkeypatch.setattr(recs, "get_cursor", fake_cursor)
        monkeypatch.setattr(recs.steering, "apply_steering", lambda embedding, _: embedding)
        monkeypatch.setattr(recs.embeddings, "ann_search", ann_search)
        monkeypatch.setattr(recs, "mmr_rerank", lambda _query, pool, _k, _lambda: pool)

        result = recs.build_recommendations("seed", k=1, include_tags=False)

        assert len(result.recommendations) == 1
        assert "listeners_cap" not in ann_search.call_args.kwargs


# ---------------------------------------------------------------------------
# _fetch_tags — batches songs.tags into the ordered tag list for the response
# (issue #22: tags ship with recommendations so the popover needs no re-fetch).
# ---------------------------------------------------------------------------

class TestFetchTags:
    def test_empty_ids_returns_empty_without_query(self, monkeypatch):
        from app.routers import recommendations as recs
        # get_cursor must not even be called for an empty id list.
        monkeypatch.setattr(recs, "get_cursor", lambda *a, **k: (_ for _ in ()).throw(AssertionError("queried")))
        assert recs._fetch_tags([]) == {}

    def test_orders_tags_by_count_and_handles_null(self, monkeypatch):
        from app.routers import recommendations as recs
        rows = [
            {"track_id": "a", "tags": {"rock": 2, "indie": 9}},
            {"track_id": "b", "tags": None},
        ]
        monkeypatch.setattr(recs, "get_cursor", make_fake_get_cursor(rows))
        out = recs._fetch_tags(["a", "b"])
        assert out == {"a": ["indie", "rock"], "b": []}
