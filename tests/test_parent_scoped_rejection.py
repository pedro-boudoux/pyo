"""Parent-scoped rejection persistence, steering, and hard exclusions (issue #36)."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from app.services import steering
from tests.conftest import FakeCursor, make_fake_get_cursor


def test_rejected_ids_include_explicit_and_legacy_rows_once(monkeypatch):
    cursor = FakeCursor([
        {"track_id": "explicit-child"},
        {"track_id": "legacy-child"},
        {"track_id": "legacy-child"},
    ])

    @contextmanager
    def fake_cursor():
        yield cursor

    monkeypatch.setattr(steering, "get_cursor", fake_cursor)

    assert steering.get_rejected_track_ids("parent-a") == {
        "explicit-child",
        "legacy-child",
    }
    sql = cursor._executed_sql[0]
    assert "f.source_track_id = %s" in sql
    assert "f.source_track_id IS NULL" in sql
    assert "graph_edges" in sql
    assert cursor._executed_params[0] == ("parent-a", "parent-a")


@pytest.fixture
def feedback_client(monkeypatch):
    monkeypatch.setattr("app.db.init_db", lambda: None)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_reject_submission_is_idempotent_and_removes_only_parent_edge(
    monkeypatch, feedback_client,
):
    cursors: list[FakeCursor] = []

    @contextmanager
    def fake_cursor():
        cursor = FakeCursor([{"id": 1}])
        cursors.append(cursor)
        yield cursor

    monkeypatch.setattr("app.routers.feedback.get_cursor", fake_cursor)
    body = {
        "source_track_id": "parent-a",
        "track_id": "child",
        "action": "reject",
    }

    assert feedback_client.post("/feedback", json=body).status_code == 200
    assert feedback_client.post("/feedback", json=body).status_code == 200

    for cursor in cursors:
        combined = "\n".join(cursor._executed_sql)
        assert "ON CONFLICT (source_track_id, track_id)" in combined
        assert "DELETE FROM graph_edges" in combined
        assert ("parent-a", "child") in cursor._executed_params


def test_recommendations_hard_exclude_only_this_parents_rejections(monkeypatch):
    from app.routers import recommendations as recs

    calls = iter([
        [{"name": "Seed", "artist": "Artist", "embedding": [1.0, 0.0]}],
        [],
    ])

    @contextmanager
    def fake_cursor():
        yield FakeCursor(next(calls))

    ann_search = MagicMock(return_value=[])
    monkeypatch.setattr(recs, "get_cursor", fake_cursor)
    monkeypatch.setattr(recs.steering, "apply_steering", lambda emb, source: [0.8, -0.2])
    monkeypatch.setattr(
        recs.steering,
        "get_rejected_track_ids",
        lambda source: {"rejected-by-a"} if source == "parent-a" else {"rejected-by-b"},
    )
    monkeypatch.setattr(recs.embeddings, "ann_search", ann_search)
    monkeypatch.setattr(recs, "topup_from_lastfm", MagicMock(return_value=[]))

    recs.build_recommendations("parent-a", k=1, include_tags=False)

    assert ann_search.call_args.args[0] == [0.8, -0.2]
    assert set(ann_search.call_args.kwargs["exclude_ids"]) == {
        "parent-a",
        "rejected-by-a",
    }
    assert "rejected-by-b" not in ann_search.call_args.kwargs["exclude_ids"]


def test_linear_and_tree_source_use_parent_feedback(monkeypatch):
    """Both playlist modes steer and hard-exclude for the current parent."""
    from app.routers import playlists
    from app.models import LinearPlaylistRequest, TreePlaylistRequest

    class PlaylistCursor(FakeCursor):
        def fetchone(self):
            return {"embedding": [1.0, 0.0]}

        def fetchall(self):
            return [{"target_id": "allowed-child"}]

    @contextmanager
    def fake_cursor():
        yield PlaylistCursor()

    rejected = MagicMock(return_value={"removed-child"})
    apply = MagicMock(return_value=[0.5, -0.5])
    find_neighbors = MagicMock(return_value=[])
    monkeypatch.setattr(playlists, "get_cursor", fake_cursor)
    monkeypatch.setattr(playlists, "embed_missing", lambda _: None)
    monkeypatch.setattr(playlists, "find_neighbors", find_neighbors)
    monkeypatch.setattr(playlists.steering, "get_rejected_track_ids", rejected)
    monkeypatch.setattr(playlists.steering, "apply_steering", apply)

    playlists.linear_playlist.__wrapped__(
        None, None, LinearPlaylistRequest(track_id="parent", n=2),
    )
    linear_args = find_neighbors.call_args.args
    assert linear_args[1] == [0.5, -0.5]
    assert set(linear_args[2]) == {"parent", "removed-child"}

    find_neighbors.reset_mock()
    rejected.reset_mock()
    apply.reset_mock()
    playlists.tree_playlist.__wrapped__(
        None, None, TreePlaylistRequest(track_id="parent", n=2),
    )
    tree_args = find_neighbors.call_args.args
    assert tree_args[1] == [0.5, -0.5]
    assert "removed-child" in tree_args[2]
    rejected.assert_called_with("parent")
    apply.assert_called_with([1.0, 0.0], "parent")


def test_reseeding_existing_parent_honors_rejections(monkeypatch):
    from app.models import SeedRequest
    from app.routers import graph
    from app.services.embeddings import make_track_id

    removed_id = make_track_id("Removed Artist", "Removed Song")
    rows = iter([
        [{
            "name": "Parent Song",
            "artist": "Parent Artist",
            "listeners": 100,
            "embedding": [1.0, 0.0],
        }],
        [],
        [],
    ])

    @contextmanager
    def fake_cursor():
        yield FakeCursor(next(rows))

    ann_search = MagicMock(return_value=[])
    embed = MagicMock(side_effect=AssertionError("rejected track must not be re-embedded"))
    monkeypatch.setattr(graph, "get_cursor", fake_cursor)
    monkeypatch.setattr(graph.steering, "get_rejected_track_ids", lambda _: {removed_id})
    monkeypatch.setattr(graph.steering, "apply_steering", lambda _emb, _source: [0.7, -0.3])
    monkeypatch.setattr(graph.embeddings, "ann_search", ann_search)
    monkeypatch.setattr(
        graph.lastfm,
        "get_similar_tracks",
        lambda *_args, **_kwargs: [{"artist": "Removed Artist", "name": "Removed Song"}],
    )
    monkeypatch.setattr(graph.lastfm, "get_similar_artists", lambda *_: [])
    monkeypatch.setattr(graph.colisten, "record_edges", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(graph.ingest, "embed_and_store_track", embed)

    graph.add_seed.__wrapped__(None, None, SeedRequest(track_id="parent"))

    assert ann_search.call_args.args[0] == [0.7, -0.3]
    assert set(ann_search.call_args.kwargs["exclude_ids"]) == {"parent", removed_id}
    embed.assert_not_called()


def test_schema_keeps_historical_source_nullable_and_new_rejects_unique():
    from pathlib import Path

    sql = Path("migrations/init.sql").read_text()
    assert "source_track_id TEXT REFERENCES songs(track_id)" in sql
    assert "source_track_id TEXT NOT NULL" not in sql.split("CREATE TABLE IF NOT EXISTS feedback", 1)[1].split(");", 1)[0]
    assert "idx_feedback_parent_reject_unique" in sql
    assert "WHERE action = 'reject' AND source_track_id IS NOT NULL" in sql
