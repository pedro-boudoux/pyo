"""
Tests for the co-listening graph crawl (algorithm 2.0, Phase 2, task 13).

Last.fm, edge persistence and graph_stats are all mocked — these pin the BFS
behavior: breadth-first level advance, visited/dedupe, budget cap, and resume.
"""
import pytest

from jobs import crawl_colisten as cc
from app.services.embeddings import make_track_id


@pytest.fixture
def graph(monkeypatch):
    """Wire an in-memory similarity graph. Returns a recorder so tests can assert
    which tracks were crawled and what edges were written."""
    # adjacency: (artist, name) -> list of similar {artist, name, match}
    sim: dict[tuple, list] = {}
    crawled: list[tuple] = []
    edges: list[tuple] = []

    def fake_get_similar(artist, name, limit=50):
        crawled.append((artist, name))
        return sim.get((artist, name), [])[:limit]

    def fake_record_edges(src_artist, src_name, targets, source, weight=None):
        for t in targets:
            edges.append((src_artist, src_name, t["artist"], t["name"]))
        return len(targets)

    monkeypatch.setattr(cc.lastfm, "get_similar_tracks", fake_get_similar)
    monkeypatch.setattr(cc.colisten, "record_edges", fake_record_edges)
    monkeypatch.setattr(cc.colisten, "record_crawl_states", lambda track_ids: len(track_ids))
    monkeypatch.setattr(cc.colisten, "graph_stats", lambda: {"nodes": 0, "edges": 0, "avg_degree": 0.0})
    # no prior crawl state unless a test overrides it (avoids touching the DB)
    monkeypatch.setattr(cc, "_already_crawled", lambda: set())

    return {"sim": sim, "crawled": crawled, "edges": edges}


def _t(artist, name, match=1.0):
    return {"artist": artist, "name": name, "match": match}


def test_depth_one_only_crawls_seed(graph):
    graph["sim"][("A", "1")] = [_t("B", "2"), _t("C", "3")]
    cc.crawl(seed=[{"artist": "A", "name": "1"}], max_depth=1, delay=0, verbose=False)
    # depth 1 = crawl the seed only; B/C are discovered but not themselves crawled
    assert graph["crawled"] == [("A", "1")]
    assert ("A", "1", "B", "2") in graph["edges"]


def test_bfs_advances_to_next_level(graph):
    graph["sim"][("A", "1")] = [_t("B", "2")]
    graph["sim"][("B", "2")] = [_t("C", "3")]
    cc.crawl(seed=[{"artist": "A", "name": "1"}], max_depth=2, delay=0, verbose=False)
    # depth 2 = seed then its neighbor B get crawled; C only discovered
    assert graph["crawled"] == [("A", "1"), ("B", "2")]


def test_visited_not_recrawled(graph):
    # A and B point at each other → B must not be crawled twice / loop forever
    graph["sim"][("A", "1")] = [_t("B", "2")]
    graph["sim"][("B", "2")] = [_t("A", "1")]
    cc.crawl(seed=[{"artist": "A", "name": "1"}], max_depth=5, delay=0, verbose=False)
    assert graph["crawled"] == [("A", "1"), ("B", "2")]


def test_max_calls_budget_stops_crawl(graph):
    for i in range(10):
        graph["sim"][("A", str(i))] = [_t("A", str(i + 1))]
    seed = [{"artist": "A", "name": str(i)} for i in range(10)]
    out = cc.crawl(seed=seed, max_depth=3, max_calls=3, delay=0, verbose=False)
    assert out["calls"] == 3
    assert len(graph["crawled"]) == 3


def test_per_level_cap_bounds_fanout(graph):
    graph["sim"][("A", "1")] = [_t("X", str(i)) for i in range(100)]
    cc.crawl(seed=[{"artist": "A", "name": "1"}], max_depth=2,
             per_level_cap=5, delay=0, verbose=False)
    # 1 seed crawl + at most 5 carried to the next level
    assert len(graph["crawled"]) == 1 + 5


def test_resume_skips_already_crawled(graph, monkeypatch):
    # pretend A/1 was crawled in a previous run
    monkeypatch.setattr(cc, "_already_crawled", lambda: {make_track_id("A", "1")})
    graph["sim"][("A", "1")] = [_t("B", "2")]
    cc.crawl(seed=[{"artist": "A", "name": "1"}], max_depth=2, delay=0, verbose=False)
    assert graph["crawled"] == []  # nothing to do, already crawled


def test_seed_dedup(graph):
    graph["sim"][("A", "1")] = []
    cc.crawl(
        seed=[{"artist": "A", "name": "1"}, {"artist": "A", "name": "1"}],
        max_depth=1, delay=0, verbose=False,
    )
    assert graph["crawled"] == [("A", "1")]


def test_successful_empty_result_is_persisted_for_resume(graph, monkeypatch):
    completed = []
    monkeypatch.setattr(
        cc.colisten,
        "record_crawl_states",
        lambda track_ids: completed.extend(track_ids) or len(track_ids),
    )

    out = cc.crawl(
        seed=[{"artist": "A", "name": "empty"}],
        max_depth=1,
        delay=0,
        verbose=False,
    )

    assert completed == [make_track_id("A", "empty")]
    assert out["empty_completed"] == 1


def test_failed_call_is_not_marked_complete(graph, monkeypatch):
    completed = []

    def fail(*args, **kwargs):
        raise RuntimeError("temporary upstream error")

    monkeypatch.setattr(cc.lastfm, "get_similar_tracks", fail)
    monkeypatch.setattr(
        cc.colisten,
        "record_crawl_states",
        lambda track_ids: completed.extend(track_ids) or len(track_ids),
    )

    out = cc.crawl(
        seed=[{"artist": "A", "name": "retry me"}],
        max_depth=1,
        delay=0,
        verbose=False,
    )

    assert completed == []
    assert out["errors"] == 1


def test_parallel_workers_batch_edge_writes(graph, monkeypatch):
    graph["sim"][("A", "1")] = [_t("B", "2"), _t("C", "3")]
    graph["sim"][("D", "4")] = [_t("E", "5")]
    batches = []

    def fake_edge_rows(src_artist, src_name, targets, source, weight=None):
        return [(src_artist, src_name, t["artist"], t["name"]) for t in targets]

    def fake_record_edge_rows(rows):
        batches.append(list(rows))
        graph["edges"].extend(rows)
        return len(rows)

    monkeypatch.setattr(cc.colisten, "edge_rows", fake_edge_rows)
    monkeypatch.setattr(cc.colisten, "record_edge_rows", fake_record_edge_rows)

    out = cc.crawl(
        seed=[{"artist": "A", "name": "1"}, {"artist": "D", "name": "4"}],
        max_depth=1,
        workers=2,
        batch_size=2,
        delay=0,
        verbose=False,
    )

    assert out["calls"] == 2
    assert out["edges_written"] == 3
    assert set(graph["crawled"]) == {("A", "1"), ("D", "4")}
    assert sum(len(batch) for batch in batches) == 3
