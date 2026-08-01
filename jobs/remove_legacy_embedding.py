"""Back up, gate, remove, and restore the obsolete sparse song embedding.

This is deliberately an operator-invoked migration. Application startup must
never drop production data. The semantic ``songs.embedding`` column is not
touched by any command in this module.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.db import get_connection


LEGACY_COLUMN = "embedding_legacy_300"
BACKUP_TABLE = "songs_embedding_legacy_300_backup"
DROP_CONFIRMATION = "DROP embedding_legacy_300"
MIGRATION_LOCK_KEY = 0x50594F40  # "PYO" + issue 40
REQUIRED_SCHEDULED_RUNS = 2


def _column_exists(cursor) -> bool:
    cursor.execute(
        """SELECT EXISTS (
               SELECT 1 FROM pg_attribute attribute
               JOIN pg_class relation ON relation.oid = attribute.attrelid
               WHERE relation.oid = 'songs'::regclass
                 AND attribute.attname = %s
                 AND attribute.attnum > 0
                 AND NOT attribute.attisdropped
           ) AS exists""",
        (LEGACY_COLUMN,),
    )
    return bool(cursor.fetchone()["exists"])


def _backup_exists(cursor) -> bool:
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", (BACKUP_TABLE,))
    return bool(cursor.fetchone()["exists"])


def _prerequisite_status(cursor) -> dict:
    cursor.execute(
        """SELECT COUNT(*) AS successful_scheduled_runs
           FROM model_runs
           WHERE params ->> 'trigger' = 'scheduled'
             AND published_at IS NOT NULL
             AND status IN ('active', 'superseded')
             AND COALESCE((validation ->> 'passed')::boolean, false)"""
    )
    scheduled_runs = int(cursor.fetchone()["successful_scheduled_runs"] or 0)
    cursor.execute(
        """SELECT EXISTS (
               SELECT 1
               FROM model_runs first_run
               JOIN model_runs second_run
                 ON first_run.previous_active_run_id = second_run.id
               WHERE second_run.previous_active_run_id = first_run.id
                 AND first_run.published_at IS NOT NULL
                 AND second_run.published_at IS NOT NULL
           ) AS complete"""
    )
    rollback_drill = bool(cursor.fetchone()["complete"])
    return {
        "successful_scheduled_runs": scheduled_runs,
        "required_scheduled_runs": REQUIRED_SCHEDULED_RUNS,
        "rollback_drill_complete": rollback_drill,
        "ready": scheduled_runs >= REQUIRED_SCHEDULED_RUNS and rollback_drill,
    }


def _backup(cursor) -> dict:
    if not _column_exists(cursor):
        raise RuntimeError(f"songs.{LEGACY_COLUMN} does not exist")
    if _backup_exists(cursor):
        raise RuntimeError(
            f"backup table {BACKUP_TABLE} already exists; verify or restore it"
        )
    cursor.execute("LOCK TABLE songs IN SHARE MODE")
    cursor.execute(
        f"""CREATE TABLE {BACKUP_TABLE} (
                track_id TEXT PRIMARY KEY,
                embedding vector(300),
                backed_up_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"""
    )
    cursor.execute(
        f"""INSERT INTO {BACKUP_TABLE} (track_id, embedding)
            SELECT track_id, {LEGACY_COLUMN} FROM songs"""
    )
    return _verify_backup(cursor)


def _verify_backup(cursor) -> dict:
    if not _backup_exists(cursor):
        raise RuntimeError(f"backup table {BACKUP_TABLE} does not exist")
    cursor.execute(f"SELECT COUNT(*) AS count FROM {BACKUP_TABLE}")
    backup_rows = int(cursor.fetchone()["count"])
    if not _column_exists(cursor):
        return {
            "backup_rows": backup_rows,
            "source_rows": None,
            "mismatched_rows": None,
            "verified": True,
        }
    cursor.execute("SELECT COUNT(*) AS count FROM songs")
    source_rows = int(cursor.fetchone()["count"])
    cursor.execute(
        f"""SELECT COUNT(*) AS count FROM songs song
            LEFT JOIN {BACKUP_TABLE} backup USING (track_id)
            WHERE backup.track_id IS NULL
              AND song.{LEGACY_COLUMN} IS NULL"""
    )
    uncovered_null_rows = int(cursor.fetchone()["count"])
    cursor.execute(
        f"""SELECT COUNT(*) AS count
            FROM songs song
            FULL JOIN {BACKUP_TABLE} backup USING (track_id)
            WHERE song.track_id IS NULL
               OR (backup.track_id IS NULL AND song.{LEGACY_COLUMN} IS NOT NULL)
               OR song.{LEGACY_COLUMN} IS DISTINCT FROM backup.embedding"""
    )
    mismatched_rows = int(cursor.fetchone()["count"])
    return {
        "backup_rows": backup_rows,
        "source_rows": source_rows,
        "uncovered_null_rows": uncovered_null_rows,
        "mismatched_rows": mismatched_rows,
        "verified": backup_rows + uncovered_null_rows == source_rows
        and mismatched_rows == 0,
    }


def _drop(cursor, confirmation: str) -> dict:
    if confirmation != DROP_CONFIRMATION:
        raise RuntimeError(f"pass --confirm '{DROP_CONFIRMATION}' to drop the column")
    prerequisites = _prerequisite_status(cursor)
    if not prerequisites["ready"]:
        raise RuntimeError(
            "legacy cleanup prerequisites are not met: "
            f"{prerequisites['successful_scheduled_runs']}/"
            f"{prerequisites['required_scheduled_runs']} successful scheduled runs; "
            f"rollback_drill_complete={prerequisites['rollback_drill_complete']}"
        )
    if not _column_exists(cursor):
        return {"dropped": False, "reason": "column already absent"}
    backup = _verify_backup(cursor)
    if not backup["verified"] or backup["source_rows"] is None:
        raise RuntimeError(f"legacy backup verification failed: {backup}")
    cursor.execute(f"ALTER TABLE songs DROP COLUMN {LEGACY_COLUMN}")
    return {"dropped": True, "backup": backup, "prerequisites": prerequisites}


def _restore(cursor) -> dict:
    if not _backup_exists(cursor):
        raise RuntimeError(f"backup table {BACKUP_TABLE} does not exist")
    cursor.execute(
        f"ALTER TABLE songs ADD COLUMN IF NOT EXISTS {LEGACY_COLUMN} vector(300)"
    )
    cursor.execute(
        f"""UPDATE songs song
            SET {LEGACY_COLUMN} = backup.embedding
            FROM {BACKUP_TABLE} backup
            WHERE backup.track_id = song.track_id"""
    )
    verification = _verify_backup(cursor)
    if not verification["verified"]:
        raise RuntimeError(f"legacy restore verification failed: {verification}")
    return {"restored": True, "backup": verification}


def execute(command: str, *, confirmation: str = "") -> dict:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
        if command == "status":
            result = {
                "column_present": _column_exists(cursor),
                "backup_present": _backup_exists(cursor),
                "prerequisites": _prerequisite_status(cursor),
            }
        elif command == "backup":
            result = {"backup": _backup(cursor)}
        elif command == "verify-backup":
            result = {"backup": _verify_backup(cursor)}
        elif command == "drop":
            result = _drop(cursor, confirmation)
        elif command == "restore":
            result = _restore(cursor)
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"unknown command: {command}")
        connection.commit()
        return result
    except BaseException:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "backup", "verify-backup", "drop", "restore")
    )
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    print(json.dumps(execute(args.command, confirmation=args.confirm), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Legacy embedding migration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
