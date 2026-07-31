from fastapi import APIRouter, HTTPException, Request, Response
from app.models import (
    GraphResponse, GraphNode, GraphEdge, SeedRequest,
    GraphTagsRequest, DominantTagsResponse,
)
from app.db import get_cursor
from app.ratelimit import limiter
from app.services import lastfm, embeddings, ingest, colisten, blacklist, hybrid
from app.services.vector_utils import to_float_list
from app.config import DEFAULT_K, RATE_LIMIT_HEAVY

SEED_SIMILAR_LIMIT = 25
EXPANSION_DEPTH = 3
EXPANSION_LIMIT = 10
ARTIST_TOPTRACKS_LIMIT = 10   # similar-artist top tracks pulled in the cold-start fallback

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
def get_graph():
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT s.track_id, s.name, s.artist, s.listeners, gn.is_seed
            FROM graph_nodes gn
            JOIN songs s ON gn.track_id = s.track_id
        """)
        nodes = [
            GraphNode(
                track_id=row["track_id"],
                name=row["name"],
                artist=row["artist"],
                is_seed=row["is_seed"],
                listeners=row["listeners"]
            )
            for row in cursor.fetchall()
        ]

        cursor.execute("SELECT source_id, target_id, similarity FROM graph_edges")
        edges = [
            GraphEdge(
                source=row["source_id"],
                target=row["target_id"],
                similarity=row["similarity"]
            )
            for row in cursor.fetchall()
        ]

    return GraphResponse(nodes=nodes, edges=edges)


@router.post("/tags", response_model=DominantTagsResponse)
def graph_dominant_tags(request: GraphTagsRequest):
    """
    Dominant tags across a graph — which genres are taking over (issue #2).

    Pass `track_ids` to scope to a specific node set (e.g. exactly what the UI is
    showing); omit it to aggregate over the whole persisted graph (every song that
    is a node or sits on either end of an edge).
    """
    with get_cursor() as cursor:
        if request.track_ids:
            cursor.execute(
                "SELECT tags FROM songs WHERE track_id = ANY(%s) AND tags IS NOT NULL",
                (request.track_ids,),
            )
        else:
            cursor.execute("""
                SELECT tags FROM songs
                WHERE tags IS NOT NULL
                AND track_id IN (
                    SELECT track_id FROM graph_nodes
                    UNION SELECT source_id FROM graph_edges
                    UNION SELECT target_id FROM graph_edges
                )
            """)
        # tags is jsonb → psycopg2 hands it back as a Python dict {tag: count}.
        tag_dicts = [r["tags"] for r in cursor.fetchall()]

    tags = embeddings.dominant_tags(tag_dicts, request.top_n)
    return DominantTagsResponse(tags=tags)


@router.post("/seed")
@limiter.limit(RATE_LIMIT_HEAVY)
def add_seed(request: Request, response: Response, body: SeedRequest):
    embedding_column = hybrid.active_embedding_column()
    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT name, artist, listeners, {embedding_column} AS embedding FROM songs WHERE track_id = %s",
            (body.track_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Track not found — search for it first")

    name, artist = row["name"], row["artist"]

    if row["embedding"] is not None:
        # already cached — skip all API calls
        vector = to_float_list(row["embedding"])
    else:
        # first time seeing this song — run the shared embedding pipeline. An
        # unbounded cap means the seed itself is never dropped for being too
        # popular (the underground cap only applies to its candidates).
        song = ingest.embed_and_store_track(artist, name)
        if song is None:
            raise HTTPException(502, "Could not fetch track data from Last.fm")
        vector = song["embedding"]

    with get_cursor() as cursor:
        cursor.execute("""
            INSERT INTO graph_nodes (track_id, is_seed)
            VALUES (%s, true)
            ON CONFLICT (track_id) DO UPDATE SET is_seed = true
        """, (body.track_id,))

    candidates = embeddings.ann_search(
        vector,
        exclude_ids=[body.track_id],
        limit=DEFAULT_K,
    )
    # Never let a blocked artist become a graph node / edge.
    candidates = [c for c in candidates if not blacklist.is_blocked(c["artist"])]

    similar = lastfm.get_similar_tracks(artist, name, limit=SEED_SIMILAR_LIMIT)
    colisten.record_edges(artist, name, similar, source="track_similar")
    seen_ids = {c["track_id"] for c in candidates} | {body.track_id}

    # Variant dedupe: a clean/explicit/remastered edition of the seed, of an ANN
    # candidate, or of a song already on the graph shares a canonical_key but not
    # a track_id — keep only one per canonical identity so the pool (and the edges
    # we write from it) can't hold the same song twice (issue #11).
    seen_keys = {embeddings.make_canonical_key(artist, name)}
    seen_keys |= {embeddings.make_canonical_key(c["artist"], c["name"]) for c in candidates}
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT s.canonical_key FROM graph_nodes gn
            JOIN songs s ON gn.track_id = s.track_id
            WHERE s.canonical_key IS NOT NULL
        """)
        seen_keys |= {r["canonical_key"] for r in cursor.fetchall()}

    def process_similar_tracks(similar_list):
        added = 0
        for sim in similar_list:
            if blacklist.is_blocked(sim["artist"]):
                continue
            try:
                sim_id = embeddings.make_track_id(sim["artist"], sim["name"])
                sim_key = embeddings.make_canonical_key(sim["artist"], sim["name"])
                if sim_id in seen_ids or sim_key in seen_keys:
                    continue

                song = ingest.embed_and_store_track(sim["artist"], sim["name"])
                if song is None:
                    continue

                similarity = embeddings.cosine_similarity(vector, song["embedding"])
                candidates.append({"track_id": song["track_id"], "similarity": similarity})
                seen_ids.add(song["track_id"])
                seen_keys.add(sim_key)
                added += 1
            except Exception:
                continue
        return added

    process_similar_tracks(similar)

    # recursive expansion: pull getSimilar from top candidates so the genre-correct pool
    # is thick enough that BFS playlists don't drift into unrelated music when they
    # exhaust the seed's direct edges
    expansion_seeds = sorted(candidates, key=lambda c: c["similarity"], reverse=True)[:EXPANSION_DEPTH]
    for cand in expansion_seeds:
        try:
            with get_cursor() as cursor:
                cursor.execute(
                    "SELECT name, artist FROM songs WHERE track_id = %s",
                    (cand["track_id"],)
                )
                cand_row = cursor.fetchone()
            if not cand_row:
                continue

            cand_similar = lastfm.get_similar_tracks(cand_row["artist"], cand_row["name"], limit=EXPANSION_LIMIT)
            colisten.record_edges(cand_row["artist"], cand_row["name"], cand_similar, source="track_similar")
            process_similar_tracks(cand_similar)
        except Exception:
            continue

    # cold-start fallback: instrumental / soundtrack / very obscure seeds often
    # have NO track.getSimilar at all, which leaves the pool empty and the graph a
    # dead single node. Mine the seed's similar artists' top tracks instead so
    # there's still something to embed, recommend, and branch into.
    if not candidates:
        for sa in lastfm.get_similar_artists(artist):
            sa_tracks = lastfm.get_artist_top_tracks(sa["artist"], limit=ARTIST_TOPTRACKS_LIMIT)
            colisten.record_edges(artist, name, sa_tracks, source="artist_similar", weight=sa["match"])
            process_similar_tracks(sa_tracks)

    # merge ANN + getSimilar + expansion results, keep top DEFAULT_K by similarity
    candidates = sorted(candidates, key=lambda c: c["similarity"], reverse=True)[:DEFAULT_K]

    if candidates:
        with get_cursor() as cursor:
            for c in candidates:
                cursor.execute("""
                    INSERT INTO graph_edges (source_id, target_id, similarity)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_id, target_id) DO UPDATE SET similarity = EXCLUDED.similarity
                """, (body.track_id, c["track_id"], c["similarity"]))

    return {"track_id": body.track_id, "name": name, "artist": artist}
