from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, Query, HTTPException, Request, Response
from app.models import SongSearchResult, TrackFeatures
from app.ratelimit import limiter
from app.security import require_maintenance_key
from app.services import lastfm, ingest, embeddings as emb_service, spotify
from app.services.covers import get_cover_url, is_broken_image
from app.services.vector_utils import to_float_list
from app.db import get_cursor
from app.config import RATE_LIMIT_HEAVY, RATE_LIMIT_MAINTENANCE

router = APIRouter(prefix="/songs", tags=["songs"])


SEARCH_LIMIT = 15
LOCAL_SEARCH_LIMIT = 20


def _search_local_songs(q: str, limit: int = LOCAL_SEARCH_LIMIT) -> list[dict]:
    pattern = f"%{q}%"
    prefix = f"{q}%"
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT track_id, name, artist, image
            FROM songs
            WHERE name ILIKE %s OR artist ILIKE %s
            ORDER BY
                CASE
                    WHEN name ILIKE %s THEN 0
                    WHEN artist ILIKE %s THEN 1
                    ELSE 2
                END,
                length(name)
            LIMIT %s
            """,
            (pattern, pattern, prefix, prefix, limit),
        )
        rows = cursor.fetchall()
    return [
        {
            "track_id": r["track_id"],
            "name": r["name"],
            "artist": r["artist"],
            "image": r["image"],
        }
        for r in rows
    ]


def _fetch_cached_images(track_ids: list[str]) -> dict[str, str | None]:
    if not track_ids:
        return {}
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT track_id, image FROM songs WHERE track_id = ANY(%s)",
            (track_ids,),
        )
        return {r["track_id"]: r["image"] for r in cursor.fetchall()}


def _upsert_songs(tracks: list[dict]) -> None:
    if not tracks:
        return
    rows = [
        (
            t["track_id"],
            t["name"],
            t["artist"],
            t.get("image"),
            emb_service.make_canonical_key(t["artist"], t["name"]),
        )
        for t in tracks
    ]
    with get_cursor() as cursor:
        # COALESCE preserves a previously-cached real cover when the latest
        # lookup returns nothing, so we never regress a known image to NULL.
        cursor.executemany(
            """
            INSERT INTO songs (track_id, name, artist, image, canonical_key)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET
                name   = EXCLUDED.name,
                artist = EXCLUDED.artist,
                image  = COALESCE(EXCLUDED.image, songs.image),
                canonical_key = EXCLUDED.canonical_key
            """,
            rows,
        )


"""
    Entry point for the user, takes user's search input and returns a track that matches that query.
    Runs a local DB search and Last.fm search in parallel, merges them, then only fetches
    covers from Deezer / iTunes for tracks we don't already have a real cover for.
    /search?q={USER_INPUT}
"""
@router.get("/search", response_model=list[SongSearchResult])
@limiter.limit(RATE_LIMIT_HEAVY)
def search_songs(
    request: Request,
    response: Response,
    q: str = Query(..., min_length=1),
):

    # Local DB lookup + Last.fm search run concurrently.
    # Last.fm failures (5xx, timeout, network) degrade gracefully to local-only.
    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(_search_local_songs, q)
        lastfm_future = pool.submit(lastfm.search_tracks, q)
        local_tracks = local_future.result()
        try:
            lastfm_tracks = lastfm_future.result()
        except Exception:
            lastfm_tracks = []

    # Assign track_ids to Last.fm results
    for t in lastfm_tracks:
        t["track_id"] = emb_service.make_track_id(t["artist"], t["name"])

    # Merge: Last.fm first (popularity-ranked), then any local-only matches.
    # Dedupe on both the exact track_id and the canonical_key, so cosmetic
    # variants of one song (e.g. an Explicit and a Clean version) collapse to the
    # first/most-popular instance instead of cluttering the dropdown (issue #11).
    seen: set[str] = set()
    seen_keys: set[str] = set()
    merged: list[dict] = []

    def _take(t: dict) -> bool:
        ck = emb_service.make_canonical_key(t["artist"], t["name"])
        if t["track_id"] in seen or ck in seen_keys:
            return False
        seen.add(t["track_id"])
        seen_keys.add(ck)
        return True

    for t in lastfm_tracks:
        if not _take(t):
            continue
        merged.append({
            "track_id": t["track_id"],
            "name": t["name"],
            "artist": t["artist"],
            "image": t.get("image"),
        })
    for t in local_tracks:
        if not _take(t):
            continue
        merged.append(t)

    merged = merged[:SEARCH_LIMIT]

    # Reuse cached covers: only call Deezer/iTunes for tracks without a real one
    cached_images = _fetch_cached_images([t["track_id"] for t in merged])
    need_cover: list[dict] = []
    for t in merged:
        cached = cached_images.get(t["track_id"])
        if cached and not is_broken_image(cached):
            t["image"] = cached
        elif is_broken_image(t.get("image")):
            need_cover.append(t)

    if need_cover:
        with ThreadPoolExecutor(max_workers=8) as pool:
            covers = list(
                pool.map(lambda t: get_cover_url(t["artist"], t["name"]), need_cover)
            )
        for t, cover in zip(need_cover, covers):
            t["image"] = cover  # cover may be None — that's fine

    _upsert_songs(merged)

    return [
        SongSearchResult(
            track_id=t["track_id"],
            name=t["name"],
            artist=t["artist"],
            image=t["image"],
        )
        for t in merged
    ]


"""
    Backfill up to `limit` songs whose semantic embedding is NULL.
    Each song re-fetches its tags from Last.fm and rebuilds the vector against the
    current tag_vocab. Call repeatedly with the same limit until `remaining` hits 0.
    Uses 2 parallel workers — Last.fm free tier allows ~5 req/s per key and each
    song makes ~7 sequential calls, so this stays comfortably within the limit.
"""
@router.post("/backfill-semantic-embeddings")
@limiter.limit(RATE_LIMIT_MAINTENANCE)
def backfill_semantic_embeddings(
    request: Request,
    response: Response,
    limit: int = Query(default=50, ge=1, le=500),
    _: None = Depends(require_maintenance_key),
):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT track_id, name, artist FROM songs WHERE embedding IS NULL LIMIT %s",
            (limit,),
        )
        batch = cursor.fetchall()

    def _reembed(song):
        try:
            result = ingest.embed_and_store_track(song["artist"], song["name"])
            return result is not None
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_reembed, batch))

    embedded = sum(1 for ok in outcomes if ok)
    failed   = sum(1 for ok in outcomes if not ok)

    with get_cursor() as cursor:
        cursor.execute("SELECT count(*) AS cnt FROM songs WHERE embedding IS NULL")
        remaining = cursor.fetchone()["cnt"]

    return {"embedded": embedded, "failed": failed, "remaining": remaining}


"""
    Refresh broken / missing album-cover URLs for songs already in the DB.
    Last.fm's placeholder hash (and any null image) gets replaced with a fresh
    URL from the cover service.
"""
@router.post("/backfill-covers")
@limiter.limit(RATE_LIMIT_MAINTENANCE)
def backfill_covers(
    request: Request,
    response: Response,
    limit: int = Query(default=200, ge=1, le=2000),
    _: None = Depends(require_maintenance_key),
):
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT track_id, name, artist FROM songs
            WHERE image IS NULL
               OR image LIKE %s
            LIMIT %s
        """, (f"%{ '2a96cbd8b46e442fc41c2b86b821562f' }%", limit))
        rows = cursor.fetchall()

    if not rows:
        return {"checked": 0, "updated": 0}

    with ThreadPoolExecutor(max_workers=8) as pool:
        covers = list(pool.map(lambda r: get_cover_url(r["artist"], r["name"]), rows))

    updated = 0
    with get_cursor() as cursor:
        for row, cover in zip(rows, covers):
            if cover:
                cursor.execute(
                    "UPDATE songs SET image = %s WHERE track_id = %s",
                    (cover, row["track_id"]),
                )
                updated += 1
    return {"checked": len(rows), "updated": updated}


"""
    Backfill canonical_key for songs written before the column existed (NULL key).
    Each is a pure string transform — no Last.fm calls — so this is fast; the
    LIMIT just bounds a single request. Call repeatedly until `remaining` is 0.
    Until a row is backfilled its NULL key simply means "no variant folding" for
    that song, so the system degrades gracefully rather than erroring.

    Pass force=true to RE-key every row (not just NULLs) — needed after the
    canonical_title rules change, since existing keys were computed under the old
    rules. With force=true `remaining` always reports 0 (every row is fresh).
"""
@router.post("/backfill-canonical")
@limiter.limit(RATE_LIMIT_MAINTENANCE)
def backfill_canonical(
    request: Request,
    response: Response,
    limit: int = Query(default=1000, ge=1, le=20000),
    force: bool = Query(default=False),
    _: None = Depends(require_maintenance_key),
):
    where = "" if force else "WHERE canonical_key IS NULL"
    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT track_id, name, artist FROM songs {where} LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()

    if not rows:
        return {"checked": 0, "updated": 0, "remaining": 0}

    with get_cursor() as cursor:
        for r in rows:
            cursor.execute(
                "UPDATE songs SET canonical_key = %s WHERE track_id = %s",
                (emb_service.make_canonical_key(r["artist"], r["name"]), r["track_id"]),
            )
        cursor.execute("SELECT count(*) AS cnt FROM songs WHERE canonical_key IS NULL")
        remaining = cursor.fetchone()["cnt"]

    return {"checked": len(rows), "updated": len(rows), "remaining": remaining}


"""
    Lightweight status check — does this song already have an embedding cached?
    Used by the frontend to warn the user about cold seeds (which trigger
    multiple Last.fm calls and take much longer).
"""
@router.get("/{track_id}/status")
def get_song_status(track_id: str):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM songs WHERE track_id = %s",
            (track_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {"exists": False, "cached": False}
        return {"exists": True, "cached": row["embedding"] is not None}


"""
    Resolve a public "listen on Spotify" link for a track via the Spotify
    client-credentials search. Returns { url } where url is the open.spotify.com
    track URL, or null if the song isn't on Spotify (or Spotify isn't configured).
    The frontend only renders the link when url is non-null.
"""
@router.get("/{track_id}/spotify")
@limiter.limit(RATE_LIMIT_HEAVY)
def get_song_spotify_link(request: Request, response: Response, track_id: str):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT name, artist, spotify_url, spotify_checked_at FROM songs WHERE track_id = %s",
            (track_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(404, "Track not found — search for it first")

    # Cache hit: we've resolved this track before (found a link or confirmed it
    # isn't on Spotify). Serve the stored answer without calling Spotify.
    if row["spotify_checked_at"] is not None:
        return {"url": row["spotify_url"], "checked": True}

    # Cache miss: resolve once and persist. A SpotifyUnavailable (creds missing,
    # network/HTTP error) is not a definitive answer, so we leave the row
    # unchecked and report checked=false — clients shouldn't cache it, and it'll
    # be retried next time.
    try:
        url = spotify.find_track_url(row["artist"], row["name"])
    except spotify.SpotifyUnavailable:
        return {"url": None, "checked": False}

    with get_cursor() as cursor:
        cursor.execute(
            "UPDATE songs SET spotify_url = %s, spotify_checked_at = now() WHERE track_id = %s",
            (url, track_id),
        )
    return {"url": url, "checked": True}


"""
    takes a {track_id} (generated in /search), returns the song with embeddings, tags, etc
"""
@router.get("/{track_id}/features", response_model=TrackFeatures)
@limiter.limit(RATE_LIMIT_HEAVY)
def get_song_features(request: Request, response: Response, track_id: str):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT name, artist, listeners, embedding, tags FROM songs WHERE track_id = %s",
            (track_id,)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(404, "Track not found — search for it first")

    if row["embedding"] is not None:
        # cache hit — reuse the stored vector, no API calls
        name, artist = row["name"], row["artist"]
        listeners = row["listeners"] or 0
        embedding = to_float_list(row["embedding"])
        tag_counts = row["tags"] or {}
    else:
        # cache miss — run the shared embedding pipeline. An unbounded cap means
        # features works for any track regardless of popularity.
        song = ingest.embed_and_store_track(row["artist"], row["name"])
        if song is None:
            raise HTTPException(502, "Could not fetch track data from Last.fm")
        name, artist = song["name"], song["artist"]
        listeners = song["listeners"] or 0
        embedding = song["embedding"]
        tag_counts = song.get("tags") or {}

    # The track's tags are the raw blended {tag: count} dict persisted in
    # songs.tags — a dense averaged embedding can't be inverted to discrete tags.
    # Surface them most-applied first (shared helper, reused by the rec/playlist
    # responses too — issue #22).
    tags = emb_service.sorted_tags(tag_counts)

    return TrackFeatures(
        track_id=track_id,
        name=name,
        artist=artist,
        listeners=listeners,
        tags=tags,
        embedding=embedding
    )
