"""Regression tests for the feature-flagged graph seeding flow (issue #35)."""

from contextlib import contextmanager

from app.services import seed_discovery


@contextmanager
def _empty_graph_cursor():
    class Cursor:
        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return []

    yield Cursor()


def _song(artist: str, name: str, similarity_axis: float = 1.0) -> dict:
    return {
        "track_id": f"{artist}:{name}",
        "name": name,
        "artist": artist,
        "listeners": 100,
        "image": None,
        "embedding": [similarity_axis, 1.0 - similarity_axis],
        "tags": {},
    }


def _base_patches(monkeypatch, *, ann=None):
    monkeypatch.setattr(seed_discovery, "get_cursor", _empty_graph_cursor)
    monkeypatch.setattr(seed_discovery.blacklist, "is_blocked", lambda artist: False)
    monkeypatch.setattr(seed_discovery.colisten, "record_edges", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        seed_discovery.embeddings,
        "ann_search",
        lambda vector, **kwargs: list(ann or []),
    )
    monkeypatch.setattr(
        seed_discovery.ingest,
        "embed_and_store_track",
        lambda artist, name: _song(artist, name),
    )


def test_full_ablation_variant_includes_recursive_expansion(monkeypatch):
    _base_patches(monkeypatch)
    calls = []

    def similar(artist, name, limit):
        calls.append((artist, name, limit))
        if name == "seed":
            return [{"artist": "Warm", "name": "direct"}]
        if name == "direct":
            return [{"artist": "Branch", "name": "expanded"}]
        return []

    monkeypatch.setattr(seed_discovery.lastfm, "get_similar_tracks", similar)
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_artists",
        lambda artist: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    result = seed_discovery.discover_seed_candidates(
        track_id="seed-id",
        artist="Seed",
        name="seed",
        vector=[1.0, 0.0],
        options=seed_discovery.SeedDiscoveryOptions(
            recursive_expansion=True,
            artist_fallback=True,
            record_colisten=False,
        ),
    )

    assert [candidate["name"] for candidate in result.candidates] == [
        "direct",
        "expanded",
    ]
    assert result.discovered_by_source == {
        "ann": 0,
        "recursive_expansion": 1,
        "seed_similar": 1,
    }
    assert result.fallback_attempted is False
    assert calls == [
        ("Seed", "seed", seed_discovery.SEED_SIMILAR_LIMIT),
        ("Warm", "direct", seed_discovery.EXPANSION_LIMIT),
    ]


def test_full_ablation_variant_includes_similar_artist_fallback(monkeypatch):
    _base_patches(monkeypatch)
    monkeypatch.setattr(seed_discovery.lastfm, "get_similar_tracks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_artists",
        lambda artist: [{"artist": "Neighbor Artist", "match": 0.8}],
    )
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_artist_top_tracks",
        lambda artist, limit: [{"artist": artist, "name": "fallback"}],
    )

    result = seed_discovery.discover_seed_candidates(
        track_id="cold-id",
        artist="Cold",
        name="no similar",
        vector=[1.0, 0.0],
        options=seed_discovery.SeedDiscoveryOptions(
            recursive_expansion=True,
            artist_fallback=True,
            record_colisten=False,
        ),
    )

    assert [candidate["name"] for candidate in result.candidates] == ["fallback"]
    assert result.selected_by_source == {"artist_fallback": 1}
    assert result.fallback_attempted is True
    assert result.discovery_calls == {
        "artist.getSimilar": 1,
        "artist.getTopTracks": 1,
        "track.getSimilar": 1,
    }


def test_ablation_controls_are_independent(monkeypatch):
    _base_patches(monkeypatch)
    similar_calls = []
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_tracks",
        lambda artist, name, limit: similar_calls.append(name) or [],
    )
    fallback_calls = []
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_artists",
        lambda artist: fallback_calls.append(artist) or [],
    )

    result = seed_discovery.discover_seed_candidates(
        track_id="cold-id",
        artist="Cold",
        name="seed",
        vector=[1.0, 0.0],
        options=seed_discovery.SeedDiscoveryOptions(
            recursive_expansion=False,
            artist_fallback=False,
            record_colisten=False,
        ),
    )

    assert result.candidates == []
    assert result.fallback_attempted is False
    assert similar_calls == ["seed"]
    assert fallback_calls == []


def test_production_defaults_use_direct_seed_flow_only(monkeypatch):
    ann = [
        {
            **_song(f"Artist {index}", f"ANN {index}"),
            "similarity": 1.0 - index / 100,
        }
        for index in range(10)
    ]
    _base_patches(monkeypatch, ann=ann)
    calls = []
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_tracks",
        lambda artist, name, limit: calls.append((name, limit)) or [],
    )
    monkeypatch.setattr(
        seed_discovery.lastfm,
        "get_similar_artists",
        lambda artist: (_ for _ in ()).throw(
            AssertionError("production fallback is disabled")
        ),
    )

    result = seed_discovery.discover_seed_candidates(
        track_id="cold-id",
        artist="Cold",
        name="no similar",
        vector=[1.0, 0.0],
    )

    assert len(result.candidates) == 10
    assert result.selected_by_source == {"ann": 10}
    assert result.fallback_attempted is False
    assert calls == [("no similar", seed_discovery.SEED_SIMILAR_LIMIT)]
    assert seed_discovery.SeedDiscoveryOptions() == seed_discovery.SeedDiscoveryOptions(
        recursive_expansion=False,
        artist_fallback=False,
        record_colisten=True,
    )


def test_seed_ann_search_remains_uncapped(monkeypatch):
    calls = []
    monkeypatch.setattr(seed_discovery, "get_cursor", _empty_graph_cursor)
    monkeypatch.setattr(seed_discovery.blacklist, "is_blocked", lambda artist: False)
    monkeypatch.setattr(
        seed_discovery.embeddings,
        "ann_search",
        lambda vector, **kwargs: calls.append(kwargs) or [],
    )
    monkeypatch.setattr(seed_discovery.lastfm, "get_similar_tracks", lambda *args, **kwargs: [])

    seed_discovery.discover_seed_candidates(
        track_id="seed-id",
        artist="Seed",
        name="seed",
        vector=[1.0, 0.0],
        options=seed_discovery.SeedDiscoveryOptions(
            recursive_expansion=False,
            artist_fallback=False,
            record_colisten=False,
        ),
    )

    assert "listeners_cap" not in calls[0]
