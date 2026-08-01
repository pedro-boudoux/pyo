from contextlib import contextmanager

import pytest

from jobs import train_colisten_embeddings as trainer


class TransactionCursor:
    def __init__(self, *, run, candidate_count, active=None, updated=None):
        self.run = run
        self.candidate_count = candidate_count
        self.active = active
        self.updated = candidate_count if updated is None else updated
        self.rowcount = -1
        self.statements = []
        self._next = None

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if "pg_try_advisory_xact_lock" in sql:
            self._next = {"acquired": True}
        elif "SELECT * FROM model_runs WHERE id" in sql:
            self._next = self.run
        elif "COUNT(*) AS count FROM model_run_vectors" in sql:
            self._next = {"count": self.candidate_count}
        elif "SELECT id FROM model_runs WHERE status = 'active'" in sql:
            self._next = self.active
        elif "SELECT * FROM model_runs WHERE status = 'active'" in sql:
            self._next = self.active
        elif "SELECT run.id FROM model_runs" in sql:
            self._next = None
        elif "UPDATE songs AS song" in sql and "FROM model_run_vectors" in sql:
            self.rowcount = self.updated
            self._next = None
        else:
            self._next = None

    def fetchone(self):
        return self._next

    def close(self):
        pass


class TransactionConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_publish_updates_vectors_and_active_run_in_one_transaction(monkeypatch):
    cursor = TransactionCursor(
        run={"id": 12, "status": "validated", "song_count": 2},
        candidate_count=2,
        active={"id": 9},
    )
    connection = TransactionConnection(cursor)
    monkeypatch.setattr(trainer, "get_connection", lambda: connection)
    monkeypatch.setattr(trainer, "_prune_candidate_vectors", lambda: None)

    result = trainer.publish(12)

    assert result == {
        "run_id": 12,
        "status": "active",
        "songs_updated": 2,
        "previous_active_run_id": 9,
    }
    assert connection.commits == 1
    assert connection.rollbacks == 0
    sql = "\n".join(statement for statement, _ in cursor.statements)
    assert "SET colisten_embedding = NULL, hybrid_embedding = NULL" in sql
    assert "FROM model_run_vectors AS candidate" in sql
    assert "status = 'superseded'" in sql
    assert "status = 'active'" in sql


def test_publish_rolls_back_everything_if_candidate_is_incomplete(monkeypatch):
    cursor = TransactionCursor(
        run={"id": 12, "status": "validated", "song_count": 3},
        candidate_count=2,
        active={"id": 9},
    )
    connection = TransactionConnection(cursor)
    monkeypatch.setattr(trainer, "get_connection", lambda: connection)

    with pytest.raises(RuntimeError, match="changed after validation"):
        trainer.publish(12)

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert not any("UPDATE songs AS song" in sql for sql, _ in cursor.statements)


def test_publish_rolls_back_if_song_update_count_changes(monkeypatch):
    cursor = TransactionCursor(
        run={"id": 12, "status": "validated", "song_count": 2},
        candidate_count=2,
        active={"id": 9},
        updated=1,
    )
    connection = TransactionConnection(cursor)
    monkeypatch.setattr(trainer, "get_connection", lambda: connection)

    with pytest.raises(RuntimeError, match="updated 1 songs"):
        trainer.publish(12)

    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_rollback_republishes_previous_candidate_atomically(monkeypatch):
    target = {"id": 9, "status": "superseded", "song_count": 2}
    cursor = TransactionCursor(
        run=target,
        candidate_count=2,
        active={"id": 12, "previous_active_run_id": 9},
    )
    connection = TransactionConnection(cursor)
    monkeypatch.setattr(trainer, "get_connection", lambda: connection)

    result = trainer.rollback()

    assert result == {
        "run_id": 9,
        "status": "active",
        "rolled_back_from": 12,
        "songs_updated": 2,
    }
    assert connection.commits == 1
    assert connection.rollbacks == 0
    sql = "\n".join(statement for statement, _ in cursor.statements)
    assert "SET colisten_embedding = NULL, hybrid_embedding = NULL" in sql
    assert "FROM model_run_vectors AS candidate" in sql


def _validation_cursor_factory(run, statements):
    calls = 0

    @contextmanager
    def cursor():
        nonlocal calls
        calls += 1

        class Cursor:
            def execute(self, sql, params=None):
                statements.append((sql, params))

            def fetchone(self):
                return run if calls == 1 else None

        yield Cursor()

    return cursor


def _candidate_stats(**overrides):
    stats = {
        "candidate_count": 2,
        "fallback_count": 1,
        "bad_hybrid_dimensions": 0,
        "bad_colisten_dimensions": 0,
        "nonfinite_hybrid_vectors": 0,
        "nonfinite_colisten_vectors": 0,
        "bad_hybrid_norms": 0,
        "bad_colisten_norms": 0,
        "fallback_mismatches": 0,
    }
    stats.update(overrides)
    return stats


def test_validation_failure_marks_candidate_failed_without_publication(monkeypatch):
    run = {
        "id": 12,
        "status": "candidate",
        "song_count": 2,
        "fallback_count": 1,
        "node_count": trainer.COLISTEN_MIN_NODES,
        "edge_count": trainer.COLISTEN_MIN_NODES * 4,
        "edge_cutoff": "cutoff",
    }
    statements = []
    monkeypatch.setattr(
        trainer, "get_cursor", _validation_cursor_factory(run, statements)
    )
    monkeypatch.setattr(
        trainer,
        "_candidate_stats",
        lambda *args: _candidate_stats(bad_hybrid_norms=1),
    )
    monkeypatch.setattr(
        trainer,
        "_independent_eval",
        lambda *args, **kwargs: {
            "coverage": 1.0,
            "recall_at_10": 0.1,
            "mrr": 0.1,
        },
    )
    monkeypatch.setattr(
        trainer,
        "get_connection",
        lambda: pytest.fail("failed validation must not open a publish transaction"),
    )

    with pytest.raises(RuntimeError, match="normalization"):
        trainer.validate(12)

    assert any("SET status = 'failed'" in sql for sql, _ in statements)
    assert not any("UPDATE songs" in sql for sql, _ in statements)


def test_validation_passes_all_required_gates(monkeypatch):
    run = {
        "id": 12,
        "status": "candidate",
        "song_count": 2,
        "fallback_count": 1,
        "node_count": trainer.COLISTEN_MIN_NODES,
        "edge_count": trainer.COLISTEN_MIN_NODES * 4,
        "edge_cutoff": "cutoff",
    }
    statements = []
    monkeypatch.setattr(
        trainer, "get_cursor", _validation_cursor_factory(run, statements)
    )
    monkeypatch.setattr(trainer, "_candidate_stats", lambda *args: _candidate_stats())
    monkeypatch.setattr(
        trainer,
        "_independent_eval",
        lambda *args, **kwargs: {
            "coverage": 1.0,
            "recall_at_10": 0.1,
            "mrr": 0.1,
        },
    )

    report = trainer.validate(12)

    assert report["passed"] is True
    assert all(report["gates"].values())
    assert any("SET status = 'validated'" in sql for sql, _ in statements)
