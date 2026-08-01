from fastapi import APIRouter, HTTPException, Request, Response
from app.models import (
    GraphResponse, GraphNode, GraphEdge, SeedRequest,
    GraphTagsRequest, DominantTagsResponse,
)
from app.db import get_cursor
from app.ratelimit import limiter
from app.services import embeddings, ingest, hybrid
from app.services.seed_discovery import discover_seed_candidates
from app.services.vector_utils import to_float_list
from app.config import RATE_LIMIT_HEAVY

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
        # first time seeing this song — run the shared embedding pipeline.
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

    discovery = discover_seed_candidates(
        track_id=body.track_id,
        artist=artist,
        name=name,
        vector=vector,
    )
    candidates = discovery.candidates

    if candidates:
        with get_cursor() as cursor:
            for c in candidates:
                cursor.execute("""
                    INSERT INTO graph_edges (source_id, target_id, similarity)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_id, target_id) DO UPDATE SET similarity = EXCLUDED.similarity
                """, (body.track_id, c["track_id"], c["similarity"]))

    return {"track_id": body.track_id, "name": name, "artist": artist}
