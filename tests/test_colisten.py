"""
Unit tests for co-listening edge collection (algorithm 2.0, Stage B data collection).

These cover the edge-shaping logic in colisten.record_edges — weight resolution,
self-edge skipping, and the best-effort error swallowing — with the DB mocked.
"""
from contextlib import contextmanager

import pytest

from app.services import colisten
from app.services.embeddings import make_track_id


class RecordingCursor:
    """Captures the rows handed to executemany so the test can assert on them."""

    def __init__(self):
        self.executemany_rows = None

    def execute(self, sql, params=None):
        pass

    def executemany(self, sql, rows):
        self.executemany_rows = list(rows)


def fake_get_cursor_factory(cursor):
    @contextmanager
    def _fake():
        yield cursor
    return _fake


def test_track_similar_uses_each_targets_match(monkeypatch):
    cursor = RecordingCursor()
    monkeypatch.setattr(colisten, "get_cursor", fake_get_cursor_factory(cursor))

    targets = [
        {"artist": "Boards of Canada", "name": "Roygbiv", "match": 0.9},
        {"artist": "Aphex Twin", "name": "Xtal", "match": 0.4},
    ]
    n = colisten.record_edges("Autechre", "Gantz Graf", targets, source="track_similar")

    assert n == 2
    src = make_track_id("Autechre", "Gantz Graf")
    rows = cursor.executemany_rows
    assert rows[0] == (src, make_track_id("Boards of Canada", "Roygbiv"), 0.9, "track_similar")
    assert rows[1] == (src, make_track_id("Aphex Twin", "Xtal"), 0.4, "track_similar")


def test_artist_similar_uses_shared_weight(monkeypatch):
    cursor = RecordingCursor()
    monkeypatch.setattr(colisten, "get_cursor", fake_get_cursor_factory(cursor))

    # artist.getTopTracks results carry no per-track match — the artist match is
    # passed as the shared weight.
    targets = [
        {"artist": "Tycho", "name": "A Walk"},
        {"artist": "Tycho", "name": "Awake"},
    ]
    colisten.record_edges("Bonobo", "Kerala", targets, source="artist_similar", weight=0.73)

    rows = cursor.executemany_rows
    assert all(r[2] == 0.73 for r in rows)
    assert all(r[3] == "artist_similar" for r in rows)


def test_self_edge_is_skipped(monkeypatch):
    cursor = RecordingCursor()
    monkeypatch.setattr(colisten, "get_cursor", fake_get_cursor_factory(cursor))

    targets = [
        {"artist": "Burial", "name": "Archangel", "match": 1.0},   # self
        {"artist": "Burial", "name": "Untrue", "match": 0.5},
    ]
    n = colisten.record_edges("Burial", "Archangel", targets, source="track_similar")

    assert n == 1
    assert len(cursor.executemany_rows) == 1
    assert cursor.executemany_rows[0][1] == make_track_id("Burial", "Untrue")


def test_empty_targets_writes_nothing(monkeypatch):
    cursor = RecordingCursor()
    monkeypatch.setattr(colisten, "get_cursor", fake_get_cursor_factory(cursor))

    n = colisten.record_edges("X", "Y", [], source="track_similar")

    assert n == 0
    assert cursor.executemany_rows is None


def test_errors_are_swallowed(monkeypatch):
    # A DB failure here must never break seeding / recommendations.
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(colisten, "get_cursor", boom)

    n = colisten.record_edges(
        "A", "B", [{"artist": "C", "name": "D", "match": 0.5}], source="track_similar"
    )
    assert n == 0
