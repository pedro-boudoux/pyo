from fastapi import APIRouter, HTTPException, Request, Response
from app.models import LinearPlaylistRequest, TreePlaylistRequest, PlaylistResponse, PlaylistTrack
from app.db import get_cursor
from app.ratelimit import limiter
from app.services import ingest, embeddings as emb_service, blacklist
from app.config import MAX_LISTENERS, RATE_LIMIT_HEAVY

router = APIRouter(prefix="/playlists", tags=["playlists"])

NICHE_THRESHOLDS = [100, 1_000, 10_000, 100_000, MAX_LISTENERS]


def embed_missing(track_ids: set):
    if not track_ids:
        return
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT artist, name FROM songs WHERE track_id = ANY(%s) AND embedding IS NULL",
            (list(track_ids),)
        )
        unembedded = cursor.fetchall()

    # Neighborhood tracks are already in the graph, so an unbounded cap keeps the
    # shared pipeline from skipping any of them for being too popular.
    for row in unembedded:
        try:
            ingest.embed_and_store_track(row["artist"], row["name"])
        except Exception:
            pass


def get_neighborhood(cursor, track_id: str) -> set:
    cursor.execute(
        "SELECT target_id FROM graph_edges WHERE source_id = %s",
        (track_id,)
    )
    return {row["target_id"] for row in cursor.fetchall()}


def find_neighbors(cursor, embedding, exclude_ids, k, niche, allowed_ids=None, blocked_artists=frozenset()):
    def _allowed(rows):
        # Mandatory env blacklist ∪ this request's blocked artists. Always applied,
        # so blocked artists never appear in Tree/Linear playlists either (issue #23).
        return [r for r in rows if not blacklist.is_blocked(r["artist"], blocked_artists)]

    if not niche:
        # Over-fetch so dropping blocked artists doesn't shrink the result below k.
        rows = emb_service.ann_search(
            embedding, exclude_ids=exclude_ids,
            allowed_ids=allowed_ids, limit=k * 3, cursor=cursor,
        )
        return _allowed(rows)[:k]

    collected = []
    excluded = set(exclude_ids)

    for threshold in NICHE_THRESHOLDS:
        if len(collected) >= k:
            break
        results = emb_service.ann_search(
            embedding, listeners_cap=threshold, exclude_ids=excluded,
            allowed_ids=allowed_ids, limit=(k - len(collected)) * 3, cursor=cursor,
        )
        for r in results:
            if len(collected) >= k:
                break
            excluded.add(r["track_id"])
            if blacklist.is_blocked(r["artist"], blocked_artists):
                continue
            collected.append(r)

    return sorted(collected, key=lambda x: x["listeners"] or 0)


def dedupe_by_canonical(rows: list[dict]) -> list[dict]:
    """
    Drop cosmetic variants (clean/explicit/remastered) that share a canonical
    identity, keeping the first occurrence. A safety net: the seed and rec paths
    already dedupe, so a well-built graph won't hold two variants — but a graph
    built before canonical dedupe existed might, and this keeps those out of the
    exported playlist (issue #11).
    """
    seen_keys = set()
    out = []
    for r in rows:
        ck = emb_service.make_canonical_key(r["artist"], r["name"])
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        out.append(r)
    return out


def _fetch_tags(track_ids: list[str]) -> dict[str, list[str]]:
    """Batch-load the stored {tag: count} dict for each track and return the
    ordered tag list the popover shows (most-applied first). Reads songs.tags only
    — no embedding/API work — so the whole playlist's tags cost one query (issue #22)."""
    if not track_ids:
        return {}
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT track_id, tags FROM songs WHERE track_id = ANY(%s)",
            (track_ids,),
        )
        return {r["track_id"]: emb_service.sorted_tags(r["tags"] or {}) for r in cursor.fetchall()}


def to_playlist_tracks(rows: list[dict]) -> list[PlaylistTrack]:
    """Map ANN rows → PlaylistTrack, attaching each track's stored top tags so the
    popover doesn't need a second /features round-trip (issue #22)."""
    tags_by_id = _fetch_tags([r["track_id"] for r in rows])
    return [
        PlaylistTrack(
            track_id=row["track_id"],
            name=row["name"],
            artist=row["artist"],
            similarity=round(row["similarity"], 3),
            listeners=row["listeners"] or 0,
            image=row.get("image"),
            tags=tags_by_id.get(row["track_id"], []),
        )
        for row in rows
    ]


@router.post("/linear", response_model=PlaylistResponse)
@limiter.limit(RATE_LIMIT_HEAVY)
def linear_playlist(request: Request, response: Response, body: LinearPlaylistRequest):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM songs WHERE track_id = %s",
            (body.track_id,)
        )
        row = cursor.fetchone()
        if not row or row["embedding"] is None:
            raise HTTPException(404, "Track not found or not yet embedded — seed it first")

        seed_embedding = [float(x) for x in row["embedding"]]
        neighborhood = get_neighborhood(cursor, body.track_id)

    embed_missing(neighborhood)
    blocked_artists = blacklist.normalize(body.exclude_artists)

    with get_cursor() as cursor:
        tracks = find_neighbors(
            cursor, seed_embedding,
            {body.track_id, *body.exclude_ids},
            body.n, body.niche,
            neighborhood if neighborhood else None,
            blocked_artists,
        )

    return PlaylistResponse(
        seed_track_id=body.track_id,
        tracks=to_playlist_tracks(dedupe_by_canonical(tracks))
    )


@router.post("/tree", response_model=PlaylistResponse)
@limiter.limit(RATE_LIMIT_HEAVY)
def tree_playlist(request: Request, response: Response, body: TreePlaylistRequest):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM songs WHERE track_id = %s",
            (body.track_id,)
        )
        row = cursor.fetchone()
        if not row or row["embedding"] is None:
            raise HTTPException(404, "Track not found or not yet embedded — seed it first")

        seed_embedding = [float(x) for x in row["embedding"]]
        allowed = get_neighborhood(cursor, body.track_id)

    embed_missing(allowed)
    blocked_artists = blacklist.normalize(body.exclude_artists)

    with get_cursor() as cursor:
        # allowed set starts as the seed's direct neighbors and grows as we visit nodes

        playlist = []
        seen = {body.track_id, *body.exclude_ids}
        queue = [(body.track_id, seed_embedding, 0)]

        while queue and len(playlist) < body.n:
            track_id, embedding, depth = queue.pop(0)
            if depth >= body.max_depth:
                continue

            # expand allowed set with this node's own edges if it has any
            allowed.update(get_neighborhood(cursor, track_id))
            current_allowed = allowed - seen

            neighbors = find_neighbors(
                cursor, embedding, seen, 2, body.niche,
                current_allowed if current_allowed else None,
                blocked_artists,
            )

            for neighbor in neighbors:
                if len(playlist) >= body.n:
                    break
                playlist.append(neighbor)
                seen.add(neighbor["track_id"])
                queue.append((neighbor["track_id"], [float(x) for x in neighbor["embedding"]], depth + 1))

    return PlaylistResponse(
        seed_track_id=body.track_id,
        tracks=to_playlist_tracks(dedupe_by_canonical(playlist))
    )
