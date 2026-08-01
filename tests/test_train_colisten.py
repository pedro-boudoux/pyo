from contextlib import contextmanager, nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs import train_colisten_embeddings as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_canonical_schema_has_lifecycle_and_run_scoped_candidates():
    schema = (REPO_ROOT / "migrations" / "init.sql").read_text()

    for status in ("running", "candidate", "validated", "active", "failed", "superseded"):
        assert f"'{status}'" in schema
    assert "CREATE TABLE IF NOT EXISTS model_run_vectors" in schema
    assert "PRIMARY KEY (model_run_id, track_id)" in schema
    assert "colisten_embedding vector(128)" in schema
    assert "hybrid_embedding   vector(512) NOT NULL" in schema


def test_startup_backfill_never_mutates_a_real_candidate_snapshot():
    startup = (REPO_ROOT / "app" / "db.py").read_text()

    assert "SELECT 1 FROM model_run_vectors existing" in startup
    assert "existing.model_run_id = run.id" in startup
    assert "AND run.song_count IS NULL" in startup


def _stub_run(monkeypatch, *, stage=None):
    monkeypatch.setattr(trainer, "model_lock", nullcontext)
    monkeypatch.setattr(trainer, "_fail_abandoned_runs", lambda: None)
    monkeypatch.setattr(trainer, "_create_run", lambda params: 41)
    monkeypatch.setattr(
        trainer,
        "_graph_snapshot",
        lambda: {
            "edge_cutoff": "cutoff",
            "nodes": trainer.COLISTEN_MIN_NODES,
            "edges": trainer.COLISTEN_MIN_NODES * 4,
            "avg_degree": trainer.COLISTEN_MIN_AVG_DEGREE,
        },
    )
    monkeypatch.setattr(
        trainer,
        "_load_songs",
        lambda: [{"track_id": "song", "embedding": [1.0] + [0.0] * 383}],
    )
    monkeypatch.setattr(trainer, "_record_snapshot", lambda *args: None)
    monkeypatch.setattr(
        trainer,
        "_load_edges",
        lambda cutoff: [
            {"source_track_id": "song", "target_track_id": "other", "weight": 1.0}
        ],
    )
    model = SimpleNamespace(wv={})
    monkeypatch.setattr(trainer, "_train_model", lambda *args, **kwargs: model)
    if stage is not None:
        monkeypatch.setattr(trainer, "_stage_song_vectors", stage)


def test_density_gate_failure_is_audited_before_training(monkeypatch):
    _stub_run(monkeypatch)
    monkeypatch.setattr(
        trainer,
        "_graph_snapshot",
        lambda: {"edge_cutoff": "cutoff", "nodes": 100, "edges": 120, "avg_degree": 2.4},
    )
    trained = False
    failures = []

    def should_not_train(*args, **kwargs):
        nonlocal trained
        trained = True

    monkeypatch.setattr(trainer, "_train_model", should_not_train)
    monkeypatch.setattr(trainer, "_mark_run_failed", lambda run_id, exc: failures.append((run_id, str(exc))))

    with pytest.raises(RuntimeError, match="density gate not met"):
        trainer.train()

    assert trained is False
    assert len(failures) == 1
    assert failures[0][0] == 41
    assert "density gate not met" in failures[0][1]


def test_interrupted_batch_marks_run_failed_and_never_updates_active_vectors(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("killed during candidate batch")

    _stub_run(monkeypatch, stage=interrupted)
    failures = []
    monkeypatch.setattr(trainer, "_mark_run_failed", lambda run_id, exc: failures.append((run_id, exc)))

    with pytest.raises(KeyboardInterrupt, match="candidate batch"):
        trainer.train()

    assert failures[0][0] == 41
    assert isinstance(failures[0][1], KeyboardInterrupt)
    source = inspect.getsource(trainer.train) + inspect.getsource(
        trainer._stage_song_vectors
    )
    assert "UPDATE songs" not in source
    assert "SET colisten_embedding" not in source


def test_overlapping_invocation_fails_before_loading_training_data(monkeypatch):
    @contextmanager
    def unavailable_lock():
        raise trainer.TrainingLockUnavailable("already running")
        yield

    touched = []
    monkeypatch.setattr(trainer, "model_lock", unavailable_lock)
    monkeypatch.setattr(trainer, "_graph_snapshot", lambda: touched.append("snapshot"))
    monkeypatch.setattr(trainer, "_record_failed_attempt", lambda message: 99)

    with pytest.raises(trainer.TrainingLockUnavailable, match="failed run 99"):
        trainer.train()

    assert touched == []


def test_model_lock_ends_pgvector_setup_transaction_before_autocommit(monkeypatch):
    events = []

    class Cursor:
        def execute(self, sql, params=None):
            events.append(sql)

        def fetchone(self):
            return {"acquired": True}

        def close(self):
            events.append("cursor-close")

    class Connection:
        def __init__(self):
            self._autocommit = False

        def rollback(self):
            events.append("rollback")

        @property
        def autocommit(self):
            return self._autocommit

        @autocommit.setter
        def autocommit(self, value):
            events.append(f"autocommit-{value}")
            self._autocommit = value

        def cursor(self):
            return Cursor()

        def close(self):
            events.append("connection-close")

    monkeypatch.setattr(trainer, "get_connection", Connection)

    with trainer.model_lock():
        events.append("body")

    assert events[0:2] == ["rollback", "autocommit-True"]
    assert "SELECT pg_try_advisory_lock(%s) AS acquired" in events
    assert "SELECT pg_advisory_unlock(%s)" in events


def test_next_lock_holder_marks_hard_killed_runs_failed(monkeypatch):
    statements = []

    class Cursor:
        def execute(self, sql, params=None):
            statements.append((sql, params))

    @contextmanager
    def cursor():
        yield Cursor()

    monkeypatch.setattr(trainer, "get_cursor", cursor)
    trainer._fail_abandoned_runs()

    sql, params = statements[0]
    assert "WHERE status = 'running'" in sql
    assert "status = 'failed'" in sql
    assert "trainer process ended" in sql
    assert params is None


def test_successful_run_finishes_as_candidate_with_complete_metadata(monkeypatch):
    _stub_run(monkeypatch, stage=lambda *args, **kwargs: (1, 1))
    statements = []

    class Cursor:
        def execute(self, sql, params=None):
            statements.append((sql, params))

    @contextmanager
    def cursor():
        yield Cursor()

    monkeypatch.setattr(trainer, "get_cursor", cursor)
    result = trainer.train(seed=7, beta=2.0, batch_size=25)

    assert result["run_id"] == 41
    assert result["status"] == "candidate"
    assert result["songs_updated"] == 1
    assert result["tag_only_fallbacks"] == 1
    assert result["params"]["seed"] == 7
    assert result["params"]["beta"] == 2.0
    assert result["params"]["batch_size"] == 25
    assert any("UPDATE model_runs" in sql and params[0] == "candidate" for sql, params in statements)


def test_staging_writes_only_candidate_table_and_tracks_fallbacks(monkeypatch):
    calls = []

    class Cursor:
        def executemany(self, sql, values):
            calls.append((sql, values))

    @contextmanager
    def cursor():
        yield Cursor()

    monkeypatch.setattr(trainer, "get_cursor", cursor)
    model = SimpleNamespace(wv={"with-graph": [1.0] + [0.0] * 127})
    songs = [
        {"track_id": "with-graph", "embedding": [1.0] + [0.0] * 383},
        {"track_id": "tag-only", "embedding": [1.0] + [0.0] * 383},
    ]

    staged, fallback = trainer._stage_song_vectors(
        12, model, songs, beta=2.0, batch_size=10
    )

    assert (staged, fallback) == (2, 1)
    sql, values = calls[0]
    assert "INSERT INTO model_run_vectors" in sql
    assert "songs" not in sql
    assert values[0][0:2] == (12, "with-graph")
    assert values[0][-1] is False
    assert values[1][-1] is True
