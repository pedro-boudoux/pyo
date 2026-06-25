from fastapi import APIRouter, HTTPException, Request
from app.models import FeedbackRequest, FeedbackResponse
from app.db import get_cursor
from app.ratelimit import limiter
from app.services import steering, embeddings
from app.config import DEFAULT_K, RATE_LIMIT_HEAVY

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
@limiter.limit(RATE_LIMIT_HEAVY)
def submit_feedback(request: Request, body: FeedbackRequest):
    if body.action not in ("accept", "reject"):
        raise HTTPException(400, "Action must be 'accept' or 'reject'")

    with get_cursor() as cursor:
        cursor.execute(
            "SELECT id FROM songs WHERE track_id = %s",
            (body.track_id,)
        )
        if not cursor.fetchone():
            raise HTTPException(404, "Track not found in database")

        cursor.execute(
            "INSERT INTO feedback (track_id, action) VALUES (%s, %s)",
            (body.track_id, body.action)
        )

        if body.action == "accept":
            cursor.execute("""
                INSERT INTO graph_nodes (track_id, is_seed)
                VALUES (%s, true)
                ON CONFLICT (track_id) DO UPDATE SET is_seed = true
            """, (body.track_id,))

            cursor.execute(
                "SELECT source_id, similarity FROM graph_edges WHERE target_id = %s LIMIT 1",
                (body.track_id,)
            )
            parent = cursor.fetchone()
            if parent:
                cursor.execute("""
                    INSERT INTO graph_edges (source_id, target_id, similarity)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_id, target_id) DO UPDATE SET similarity = EXCLUDED.similarity
                """, (parent["source_id"], body.track_id, parent["similarity"]))

            cursor.execute(
                "SELECT embedding FROM songs WHERE track_id = %s",
                (body.track_id,)
            )
            song_row = cursor.fetchone()
            if song_row and song_row["embedding"] is not None:
                base_embedding = list(song_row["embedding"])
                steered = steering.apply_steering(base_embedding, body.track_id)

                neighbors = embeddings.ann_search(
                    steered,
                    exclude_ids=[body.track_id],
                    limit=DEFAULT_K,
                    cursor=cursor,
                )

                for r in neighbors:
                    cursor.execute("""
                        INSERT INTO graph_edges (source_id, target_id, similarity)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (source_id, target_id) DO UPDATE SET similarity = EXCLUDED.similarity
                    """, (body.track_id, r["track_id"], r["similarity"]))

    return FeedbackResponse(
        success=True,
        message=f"Track {body.action}ed successfully"
    )
