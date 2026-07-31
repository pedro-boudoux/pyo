import pytest

from app.services.embeddings import make_canonical_key, make_track_id
from eval.ground_truth_colisten import build_entries, validate_fixture


def _song(artist, name):
    return {
        "track_id": make_track_id(artist, name),
        "artist": artist,
        "name": name,
        "canonical_key": make_canonical_key(artist, name),
    }


def test_build_entries_uses_playlist_adjacency_and_folds_cosmetic_titles():
    songs = [
        _song("Alpha", "One"),
        _song("Beta", "Two"),
        _song("Gamma", "Three"),
        _song("Delta", "Four"),
    ]
    playlist = {"id": "42", "title": "independent", "queries": ["test"]}
    tracks = [
        {"artist": "Alpha", "name": "One - Remastered 2011"},
        {"artist": "Beta", "name": "Two"},
        {"artist": "Gamma", "name": "Three"},
        {"artist": "Delta", "name": "Four"},
    ]

    entries, playlists = build_entries(
        [(playlist, tracks)],
        songs,
        window=3,
        target_limit=3,
        min_targets=2,
        seed_limit=10,
    )

    assert len(entries) == 4
    assert playlists[0]["matched_tracks"] == 4
    alpha = next(entry for entry in entries if entry["artist"] == "Alpha")
    assert make_track_id("Beta", "Two") in alpha["targets"]
    assert alpha["seed_track_id"] not in alpha["targets"]


def test_build_entries_excludes_same_artist_pairs():
    songs = [_song("Alpha", "One"), _song("Alpha", "Two"), _song("Beta", "Three")]
    tracks = [{"artist": song["artist"], "name": song["name"]} for song in songs]
    entries, _ = build_entries(
        [({"id": "1", "title": "x", "queries": []}, tracks)],
        songs,
        window=2,
        target_limit=5,
        min_targets=1,
        seed_limit=10,
    )
    by_id = {entry["seed_track_id"]: entry for entry in entries}
    assert make_track_id("Alpha", "Two") not in by_id[make_track_id("Alpha", "One")]["targets"]


def test_validate_fixture_rejects_circular_or_tiny_data():
    with pytest.raises(ValueError, match="independent"):
        validate_fixture({"source": "lastfm_getsimilar", "seeds": []}, min_seeds=1)

    with pytest.raises(ValueError, match="at least 2"):
        validate_fixture(
            {"source": "human", "independent_from_lastfm": True, "seeds": []},
            min_seeds=2,
        )


def test_validate_fixture_accepts_independent_unique_pairs():
    validate_fixture(
        {
            "source": "deezer_public_playlist_adjacency",
            "independent_from_lastfm": True,
            "seeds": [
                {"seed_track_id": "a", "targets": ["b", "c"]},
                {"seed_track_id": "b", "targets": ["a"]},
            ],
        },
        min_seeds=2,
    )
