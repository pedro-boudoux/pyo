"""
Semantic tag encoder (algorithm 2.0, Phase 1).

Turns Last.fm tag strings into dense 384-dim vectors with all-MiniLM-L6-v2, loaded
via fastembed (ONNX, CPU, no torch). The model is loaded lazily on first use so
importing this module — and running the test suite — never pays the load cost.

CRITICAL caching contract: every embedding is cached by its unique tag string in
`tag_vocab.embedding`. The same handful of tags ("rock", "electronic", ...) recur
across thousands of songs, so each unique tag is encoded **once** and reused. The
backfill is only fast if this cache is solid — never encode per song.
"""
import threading

import numpy as np

from app.config import TAG_ENCODER_MODEL, TAG_EMBEDDING_DIM
from app.db import get_cursor
from app.services.vector_utils import to_float_list

_model = None
# Serialize encode calls: the backfill runs several worker threads, and the ONNX
# session / tokenizer aren't guaranteed thread-safe. Cache hits never reach here,
# so contention is negligible in steady state.
_encode_lock = threading.Lock()


def _get_model():
    """Lazily construct the fastembed model (downloads the ONNX weights once)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=TAG_ENCODER_MODEL)
    return _model


def _encode(tags: list[str]) -> list[np.ndarray]:
    """Encode a batch of tag strings to numpy vectors (no caching)."""
    with _encode_lock:
        return list(_get_model().embed(tags))


def get_tag_embeddings(tags: list[str]) -> dict[str, np.ndarray]:
    """
    Return {tag: 384-dim vector} for the given tags, reading cached vectors from
    tag_vocab.embedding and encoding only the ones not seen before (then caching
    them). Tags are deduped, so a tag repeated within the call is encoded once.
    """
    unique = list(dict.fromkeys(t for t in tags if t))
    if not unique:
        return {}

    with get_cursor() as cursor:
        cursor.execute(
            "SELECT tag, embedding FROM tag_vocab WHERE tag = ANY(%s) AND embedding IS NOT NULL",
            (unique,),
        )
        cached = {r["tag"]: np.asarray(to_float_list(r["embedding"]), dtype=float) for r in cursor.fetchall()}

    missing = [t for t in unique if t not in cached]
    if missing:
        vectors = _encode(missing)
        with get_cursor() as cursor:
            for tag, vec in zip(missing, vectors):
                vec = np.asarray(vec, dtype=float)
                # Upsert the tag row and its cached embedding in one shot. The tag
                # row may already exist (added by an older get_or_create call) with
                # a NULL embedding — DO UPDATE backfills it.
                cursor.execute(
                    """
                    INSERT INTO tag_vocab (tag, embedding) VALUES (%s, %s)
                    ON CONFLICT (tag) DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    (tag, vec.tolist()),
                )
                cached[tag] = vec

    return cached


def embed_tag(tag: str) -> list[float]:
    """Convenience single-tag encode (cached). Returns a zero vector for empties."""
    emb = get_tag_embeddings([tag]).get(tag)
    return emb.tolist() if emb is not None else [0.0] * TAG_EMBEDDING_DIM
