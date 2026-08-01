from pathlib import Path

import pytest

from jobs import remove_legacy_embedding as cleanup


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(
        self,
        *,
        column_exists=True,
        backup_exists=False,
        scheduled_runs=2,
        rollback_complete=True,
        rows=3,
        uncovered_null_rows=0,
        mismatches=0,
    ):
        self.column_exists = column_exists
        self.backup_exists = backup_exists
        self.scheduled_runs = scheduled_runs
        self.rollback_complete = rollback_complete
        self.rows = rows
        self.uncovered_null_rows = uncovered_null_rows
        self.mismatches = mismatches
        self.statements = []
        self._row = None

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.statements.append((normalized, params))
        if "FROM pg_attribute" in normalized:
            self._row = {"exists": self.column_exists}
        elif "to_regclass" in normalized:
            self._row = {"exists": self.backup_exists}
        elif normalized.startswith(f"CREATE TABLE {cleanup.BACKUP_TABLE}"):
            self.backup_exists = True
        elif "successful_scheduled_runs" in normalized:
            self._row = {"successful_scheduled_runs": self.scheduled_runs}
        elif "AS complete" in normalized:
            self._row = {"complete": self.rollback_complete}
        elif normalized == f"SELECT COUNT(*) AS count FROM {cleanup.BACKUP_TABLE}":
            self._row = {"count": self.rows}
        elif normalized == "SELECT COUNT(*) AS count FROM songs":
            self._row = {"count": self.rows}
        elif "LEFT JOIN" in normalized:
            self._row = {"count": self.uncovered_null_rows}
        elif "FULL JOIN" in normalized:
            self._row = {"count": self.mismatches}
        elif normalized.startswith("ALTER TABLE songs ADD COLUMN"):
            self.column_exists = True
        elif normalized.startswith("ALTER TABLE songs DROP COLUMN"):
            self.column_exists = False

    def fetchone(self):
        assert self._row is not None
        row, self._row = self._row, None
        return row


def sql_text(cursor):
    return "\n".join(sql for sql, _ in cursor.statements)


def test_backup_copies_every_song_and_verifies_exact_values():
    cursor = FakeCursor()

    result = cleanup._backup(cursor)

    assert result == {
        "backup_rows": 3,
        "source_rows": 3,
        "uncovered_null_rows": 0,
        "mismatched_rows": 0,
        "verified": True,
    }
    assert "LOCK TABLE songs IN SHARE MODE" in sql_text(cursor)
    assert "SELECT track_id, embedding_legacy_300 FROM songs" in sql_text(cursor)
    assert "IS DISTINCT FROM backup.embedding" in sql_text(cursor)


def test_backup_refuses_to_overwrite_existing_copy():
    cursor = FakeCursor(backup_exists=True)

    with pytest.raises(RuntimeError, match="already exists"):
        cleanup._backup(cursor)


def test_verify_accepts_new_songs_with_no_obsolete_value():
    cursor = FakeCursor(
        backup_exists=True, rows=4, uncovered_null_rows=1, mismatches=0
    )
    # The backup predates one new song, so it contains three rows.
    cursor.rows = 4

    original_execute = cursor.execute

    def execute(sql, params=None):
        original_execute(sql, params)
        if "COUNT(*) AS count FROM songs_embedding_legacy_300_backup" in " ".join(
            sql.split()
        ):
            cursor._row = {"count": 3}

    cursor.execute = execute

    result = cleanup._verify_backup(cursor)

    assert result["verified"] is True
    assert result["uncovered_null_rows"] == 1


def test_restore_recreates_only_the_legacy_column_and_verifies_it():
    cursor = FakeCursor(column_exists=False, backup_exists=True)

    result = cleanup._restore(cursor)

    assert result["restored"] is True
    statements = sql_text(cursor)
    assert "ADD COLUMN IF NOT EXISTS embedding_legacy_300 vector(300)" in statements
    assert "SET embedding_legacy_300 = backup.embedding" in statements
    assert "DROP COLUMN" not in statements


@pytest.mark.parametrize(
    ("scheduled_runs", "rollback_complete"),
    [(1, True), (2, False)],
)
def test_drop_refuses_until_production_prerequisites_are_recorded(
    scheduled_runs, rollback_complete
):
    cursor = FakeCursor(
        backup_exists=True,
        scheduled_runs=scheduled_runs,
        rollback_complete=rollback_complete,
    )

    with pytest.raises(RuntimeError, match="prerequisites are not met"):
        cleanup._drop(cursor, cleanup.DROP_CONFIRMATION)

    assert "DROP COLUMN" not in sql_text(cursor)


def test_drop_requires_exact_confirmation_and_verified_backup():
    cursor = FakeCursor(backup_exists=True)
    with pytest.raises(RuntimeError, match="--confirm"):
        cleanup._drop(cursor, "yes")

    cursor = FakeCursor(backup_exists=True, mismatches=1)
    with pytest.raises(RuntimeError, match="verification failed"):
        cleanup._drop(cursor, cleanup.DROP_CONFIRMATION)


def test_drop_removes_only_obsolete_sparse_column():
    cursor = FakeCursor(backup_exists=True)

    result = cleanup._drop(cursor, cleanup.DROP_CONFIRMATION)

    assert result["dropped"] is True
    statements = sql_text(cursor)
    assert "ALTER TABLE songs DROP COLUMN embedding_legacy_300" in statements
    assert "DROP COLUMN embedding" not in statements.replace(
        "DROP COLUMN embedding_legacy_300", ""
    )


def test_runtime_schema_and_api_have_no_legacy_sparse_consumer():
    schema = (REPO_ROOT / "migrations/init.sql").read_text()
    startup = (REPO_ROOT / "app/db.py").read_text()
    config = (REPO_ROOT / "app/config.py").read_text()
    songs_router = (REPO_ROOT / "app/routers/songs.py").read_text()

    assert "embedding_legacy_300" not in schema
    assert "embedding_legacy_300" not in startup
    assert "LEGACY_EMBEDDING_DIM" not in config
    assert "/repack-vocab" not in songs_router
    assert 'embedding  vector({EMBEDDING_DIM})' in startup
    assert '@router.post("/backfill-semantic-embeddings")' in songs_router
