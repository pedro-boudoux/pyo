"""
Tier-2 tests for app/services/tag_encoder.py (algorithm 2.0, Phase 1).

The fastembed model is never loaded — _encode is monkeypatched. These tests pin
the one thing that matters for backfill performance: each unique tag is encoded at
most once, cached hits are served from tag_vocab.embedding without re-encoding, and
misses are written back to the cache.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from app.services import tag_encoder


class _CacheCursor:
    """Fake cursor backed by an in-memory {tag: vector} cache. SELECT returns the
    cached rows for the requested tags; INSERT...ON CONFLICT upserts them."""

    def __init__(self, cache):
        self.cache = cache          # {tag: list[float]}
        self._last_rows = []
        self.inserts = []           # tags written back

    def execute(self, sql, params=()):
        verb = sql.strip().split()[0].upper()
        if verb == "SELECT":
            (tags,) = params
            self._last_rows = [
                {"tag": t, "embedding": self.cache[t]} for t in tags if t in self.cache
            ]
        elif verb == "INSERT":
            tag, vec = params
            self.cache[tag] = vec
            self.inserts.append(tag)

    def fetchall(self):
        return list(self._last_rows)


def _patch(monkeypatch, cache, encode_calls):
    cursor = _CacheCursor(cache)

    @contextmanager
    def fake_get_cursor():
        yield cursor

    monkeypatch.setattr("app.services.tag_encoder.get_cursor", fake_get_cursor)

    def fake_encode(tags):
        encode_calls.append(list(tags))
        # deterministic stand-in vector per tag
        return [np.asarray([float(len(t)), 1.0, 2.0]) for t in tags]

    monkeypatch.setattr("app.services.tag_encoder._encode", fake_encode)
    return cursor


def test_misses_are_encoded_and_cached(monkeypatch):
    cache = {}
    calls = []
    cursor = _patch(monkeypatch, cache, calls)

    out = tag_encoder.get_tag_embeddings(["rock", "jazz"])

    assert set(out) == {"rock", "jazz"}
    assert calls == [["rock", "jazz"]]            # one batch encode of both misses
    assert set(cursor.inserts) == {"rock", "jazz"}  # both written back to the cache


def test_cached_tags_are_not_re_encoded(monkeypatch):
    cache = {"rock": [9.0, 9.0, 9.0]}             # rock already cached
    calls = []
    cursor = _patch(monkeypatch, cache, calls)

    out = tag_encoder.get_tag_embeddings(["rock", "jazz"])

    # rock served from cache, only jazz encoded
    assert calls == [["jazz"]]
    assert cursor.inserts == ["jazz"]
    assert out["rock"].tolist() == [9.0, 9.0, 9.0]


def test_duplicate_tags_encoded_once(monkeypatch):
    cache = {}
    calls = []
    _patch(monkeypatch, cache, calls)

    tag_encoder.get_tag_embeddings(["rock", "rock", "rock"])

    assert calls == [["rock"]]                    # deduped before encoding


def test_empty_and_blank_tags_skipped(monkeypatch):
    cache = {}
    calls = []
    _patch(monkeypatch, cache, calls)

    assert tag_encoder.get_tag_embeddings([]) == {}
    assert tag_encoder.get_tag_embeddings(["", None]) == {}
    assert calls == []                            # nothing encoded