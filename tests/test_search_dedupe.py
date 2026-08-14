"""
Tier-1 test for search-result variant dedupe (issue #11).

The /songs/search merge is a *live* path (local DB + Last.fm), so rather than
hit the network we mock its seams and assert that two cosmetic variants of one
song (an Explicit and a Clean edition) collapse to a single result.
"""

from app.routers import songs


def _patch_seams(monkeypatch, lastfm_tracks, local_tracks):
    # Last.fm + local DB are the two inputs to the merge.
    monkeypatch.setattr(songs.lastfm, "search_tracks", lambda q: list(lastfm_tracks))
    monkeypatch.setattr(songs, "_search_local_songs", lambda q: list(local_tracks))
    # Skip the cover-cache lookup and the DB upsert entirely.
    monkeypatch.setattr(songs, "_fetch_cached_images", lambda ids: {})
    monkeypatch.setattr(songs, "_upsert_songs", lambda tracks: None)


def test_explicit_and_clean_variants_collapse(monkeypatch):
    """Last.fm returns HUMBLE. and HUMBLE. (Explicit) — only one should survive."""
    _patch_seams(
        monkeypatch,
        lastfm_tracks=[
            {"name": "HUMBLE.", "artist": "Kendrick Lamar", "image": "http://img/a.jpg"},
            {"name": "HUMBLE. (Explicit)", "artist": "Kendrick Lamar", "image": "http://img/b.jpg"},
        ],
        local_tracks=[],
    )

    results = songs.search_songs(request=None, response=None, q="humble")

    assert len(results) == 1
    # First (most-popular, Last.fm-ordered) instance wins.
    assert results[0].name == "HUMBLE."


def test_local_variant_does_not_duplicate_lastfm_hit(monkeypatch):
    """A locally-cached Remastered edition must not re-appear under the Last.fm one."""
    _patch_seams(
        monkeypatch,
        lastfm_tracks=[
            {"name": "Idioteque", "artist": "Radiohead", "image": "http://img/a.jpg"},
        ],
        local_tracks=[
            {"track_id": "local000000000000000",
             "name": "Idioteque - Remastered", "artist": "Radiohead", "image": "http://img/c.jpg"},
        ],
    )

    results = songs.search_songs(request=None, response=None, q="idioteque")

    assert len(results) == 1
    assert results[0].name == "Idioteque"


def test_live_version_is_kept_separate(monkeypatch):
    """A live edition is a different recording — it must NOT be folded away."""
    _patch_seams(
        monkeypatch,
        lastfm_tracks=[
            {"name": "Idioteque", "artist": "Radiohead", "image": "http://img/a.jpg"},
            {"name": "Idioteque (Live)", "artist": "Radiohead", "image": "http://img/b.jpg"},
        ],
        local_tracks=[],
    )

    results = songs.search_songs(request=None, response=None, q="idioteque")

    assert len(results) == 2


def test_search_never_resolves_covers_inline(monkeypatch):
    """Covers load lazily via /songs/{id}/cover — search must not block on providers."""
    _patch_seams(
        monkeypatch,
        lastfm_tracks=[{"name": "Idioteque", "artist": "Radiohead", "image": None}],
        local_tracks=[],
    )

    def boom(*a, **kw):
        raise AssertionError("search must not call cover providers")

    monkeypatch.setattr(songs.covers, "resolve_cover", boom)
    monkeypatch.setattr(songs, "get_cover_url", boom)

    results = songs.search_songs(request=None, response=None, q="idioteque")

    # No cached cover and a broken/absent Last.fm image → null, not a fetch.
    assert len(results) == 1
    assert results[0].image is None
