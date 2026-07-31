import pytest

from jobs import train_colisten_embeddings as trainer


def test_collapse_undirected_edges_uses_strongest_weight():
    adjacency = trainer.collapse_undirected_edges([
        {"source_track_id": "a", "target_track_id": "b", "weight": 0.4},
        {"source_track_id": "b", "target_track_id": "a", "weight": 0.9},
        {"source_track_id": "a", "target_track_id": "a", "weight": 1.0},
    ])
    assert adjacency == {"a": [("b", 0.9)], "b": [("a", 0.9)]}


def test_weighted_walks_are_bounded_and_repeatable():
    adjacency = {"a": [("b", 1.0)], "b": [("a", 1.0)]}
    kwargs = {"walk_length": 4, "walks_per_node": 2, "seed": 7}
    first = list(trainer.WeightedWalks(adjacency, **kwargs))
    second = list(trainer.WeightedWalks(adjacency, **kwargs))
    assert first == second
    assert len(first) == 4
    assert all(len(walk) == 4 for walk in first)


def test_density_gate_stops_before_loading_training_dependency(monkeypatch):
    monkeypatch.setattr(
        trainer.colisten,
        "graph_stats",
        lambda: {"nodes": 100, "edges": 120, "avg_degree": 2.4},
    )
    with pytest.raises(RuntimeError, match="density gate not met"):
        trainer.train()


def test_density_gate_status_exposes_thresholds():
    below = trainer.density_gate_status(
        {"nodes": 30000, "edges": 110000, "avg_degree": 7.33}
    )
    ready = trainer.density_gate_status(
        {"nodes": 30000, "edges": 120000, "avg_degree": 8.0}
    )

    assert below["ready"] is False
    assert ready["ready"] is True
    assert ready["required_nodes"] == trainer.COLISTEN_MIN_NODES
    assert ready["required_avg_degree"] == trainer.COLISTEN_MIN_AVG_DEGREE
