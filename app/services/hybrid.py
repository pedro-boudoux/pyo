"""Phase 2 hybrid embedding composition and model-column selection."""

import numpy as np

from app.config import (
    COLISTEN_BETA,
    COLISTEN_EMBEDDING_DIM,
    HYBRID_EMBEDDING_DIM,
    RECOMMENDATION_MODEL,
    TAG_EMBEDDING_DIM,
)
from app.services.vector_utils import to_float_list


def active_embedding_column() -> str:
    """SQL identifier for the configured recommendation representation.

    The value comes from a closed enum in config, never request input, so callers
    may safely interpolate it into SQL and alias it back to ``embedding``.
    """
    return "hybrid_embedding" if RECOMMENDATION_MODEL == "hybrid" else "embedding"


def compose(tag_embedding, colisten_embedding=None, beta: float | None = None) -> list[float]:
    """Return normalize(concat(tag_vec, beta * colisten_vec)).

    A missing co-listening vector becomes 128 zeros, making the result a clean
    Stage A fallback. Dimension mismatches fail loudly before reaching pgvector.
    """
    tag = np.asarray(to_float_list(tag_embedding), dtype=float)
    if tag.size != TAG_EMBEDDING_DIM:
        raise ValueError(f"tag embedding must have {TAG_EMBEDDING_DIM} values, got {tag.size}")

    if colisten_embedding is None:
        colisten = np.zeros(COLISTEN_EMBEDDING_DIM, dtype=float)
    else:
        colisten = np.asarray(to_float_list(colisten_embedding), dtype=float)
        if colisten.size != COLISTEN_EMBEDDING_DIM:
            raise ValueError(
                f"co-listening embedding must have {COLISTEN_EMBEDDING_DIM} values, got {colisten.size}"
            )
        # Word2Vec vector magnitudes are not calibrated. Normalize this half so
        # beta controls the intended tag-vs-graph contribution consistently for
        # every track rather than inheriting arbitrary training-vector norms.
        colisten_norm = np.linalg.norm(colisten)
        if colisten_norm:
            colisten = colisten / colisten_norm

    value = np.concatenate((tag, float(COLISTEN_BETA if beta is None else beta) * colisten))
    if value.size != HYBRID_EMBEDDING_DIM:
        raise ValueError(f"hybrid embedding must have {HYBRID_EMBEDDING_DIM} values")
    norm = np.linalg.norm(value)
    return (value / norm).tolist() if norm else value.tolist()


def embedding_from_row(row: dict) -> list[float] | None:
    """Read the configured representation, composing a hybrid fallback in memory."""
    tag = row.get("embedding")
    if tag is None:
        return None
    if RECOMMENDATION_MODEL != "hybrid":
        return to_float_list(tag)
    stored = row.get("hybrid_embedding")
    if stored is not None:
        return to_float_list(stored)
    return compose(tag, row.get("colisten_embedding"))
