import numpy as np
from app.db import get_cursor
from app.config import STEERING_ALPHA
from app.services.vector_utils import to_float_list
from app.services import hybrid


def get_rejected_embeddings(seed_track_id: str) -> list:
    embedding_column = hybrid.active_embedding_column()
    with get_cursor() as cursor:
        cursor.execute(f"""
            SELECT s.{embedding_column} AS embedding
            FROM feedback f
            JOIN songs s ON f.track_id = s.track_id
            WHERE f.action = 'reject'
            AND EXISTS (
                SELECT 1 FROM graph_edges ge
                WHERE ge.source_id = %s AND ge.target_id = f.track_id
            )
        """, (seed_track_id,))
        results = cursor.fetchall()
        return [to_float_list(row["embedding"]) for row in results if row["embedding"] is not None]


def apply_steering(base_embedding: list, seed_track_id: str) -> list:
    base = np.array(base_embedding)
    rejected = get_rejected_embeddings(seed_track_id)

    if not rejected:
        return base.tolist()

    steering = np.zeros_like(base)
    for rej in rejected:
        rej_vec = np.array(rej)
        steering += STEERING_ALPHA * rej_vec

    result = base - steering
    norm = np.linalg.norm(result)
    result = result / norm if norm > 0 else result

    return result.tolist()
