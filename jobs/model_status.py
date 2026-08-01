"""Secret-safe model freshness summary for Phase 2 operators."""

import argparse
from datetime import datetime, timezone
import json
import os
import re
import sys

from dotenv import load_dotenv


def _preload_env_file(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--env-file" and index + 1 < len(argv):
            load_dotenv(argv[index + 1], override=True)
            return
        if arg.startswith("--env-file="):
            load_dotenv(arg.split("=", 1)[1], override=True)
            return


_preload_env_file(sys.argv[1:])

from app.db import get_cursor


DEFAULT_STALE_AFTER_HOURS = float(os.getenv("MODEL_STALE_AFTER_HOURS", "192"))
DEFAULT_FAILURE_WINDOW_HOURS = float(
    os.getenv("MODEL_FAILURE_WINDOW_HOURS", "168")
)

_URL_CREDENTIALS = re.compile(r"(?P<scheme>postgres(?:ql)?://)[^@\s]+@", re.I)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key)\s*[=:]\s*[^\s,;]+"
)


def redact_failure(value: str | None) -> str | None:
    if not value:
        return None
    value = _URL_CREDENTIALS.sub(r"\g<scheme><redacted>@", str(value))
    value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", value)
    return value[:500]


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def model_status(
    *,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
    failure_window_hours: float = DEFAULT_FAILURE_WINDOW_HOURS,
) -> dict:
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT run.*,
                      EXTRACT(EPOCH FROM (now() - run.published_at)) / 3600.0 AS age_hours,
                      (SELECT COUNT(*) FROM model_run_vectors vector
                       WHERE vector.model_run_id = run.id) AS candidate_count
               FROM model_runs run
               WHERE run.status = 'active'
               ORDER BY run.published_at DESC NULLS LAST, run.id DESC
               LIMIT 1"""
        )
        active = cursor.fetchone()
        cursor.execute(
            """SELECT id, status, started_at, finished_at, failure_details
               FROM model_runs
               ORDER BY COALESCE(finished_at, started_at) DESC, id DESC LIMIT 1"""
        )
        latest = cursor.fetchone()
        cursor.execute(
            """SELECT id, started_at, finished_at, failure_details
               FROM model_runs WHERE status = 'failed'
               ORDER BY started_at DESC, id DESC LIMIT 1"""
        )
        last_failure = cursor.fetchone()

    alerts = []
    active_summary = None
    if not active:
        alerts.append("no_active_model")
    else:
        age_hours = float(active["age_hours"]) if active.get("age_hours") is not None else None
        candidate_count = int(active.get("candidate_count") or 0)
        song_count = int(active.get("song_count") or 0)
        coverage = candidate_count / song_count if song_count else None
        fallback_count = int(active.get("fallback_count") or 0)
        if age_hours is None or age_hours > stale_after_hours:
            alerts.append("active_model_stale")
        active_summary = {
            "run_id": int(active["id"]),
            "model": active["model"],
            "published_at": _iso(active.get("published_at")),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "stale": age_hours is None or age_hours > stale_after_hours,
            "graph_snapshot": {
                "nodes": int(active.get("node_count") or 0),
                "edges": int(active.get("edge_count") or 0),
                "cutoff": _iso(active.get("edge_cutoff")),
            },
            "candidate": {
                "song_count": song_count,
                "stored_vectors": candidate_count,
                "coverage": round(coverage, 6) if coverage is not None else None,
                "fallback_count": fallback_count,
                "fallback_rate": (
                    round(fallback_count / song_count, 6) if song_count else None
                ),
            },
        }

    latest_summary = None
    if latest:
        latest_summary = {
            "run_id": int(latest["id"]),
            "status": latest["status"],
            "started_at": _iso(latest.get("started_at")),
            "finished_at": _iso(latest.get("finished_at")),
            "failure": redact_failure(latest.get("failure_details")),
        }
        if latest["status"] == "failed":
            alerts.append("latest_run_failed")

    failure_summary = None
    if last_failure:
        finished = last_failure.get("finished_at") or last_failure.get("started_at")
        age = None
        if finished and hasattr(finished, "timestamp"):
            aware = finished
            if aware.tzinfo is None:
                aware = aware.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - aware).total_seconds() / 3600
        failure_summary = {
            "run_id": int(last_failure["id"]),
            "finished_at": _iso(finished),
            "age_hours": round(age, 2) if age is not None else None,
            "recent": age is None or age <= failure_window_hours,
            "failure": redact_failure(last_failure.get("failure_details")),
        }

    return {
        "healthy": not alerts,
        "alerts": alerts,
        "thresholds": {
            "stale_after_hours": stale_after_hours,
            "failure_window_hours": failure_window_hours,
        },
        "active": active_summary,
        "latest_attempt": latest_summary,
        "last_failure": failure_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report active Phase 2 model freshness.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--stale-after-hours", type=float, default=DEFAULT_STALE_AFTER_HOURS)
    parser.add_argument("--failure-window-hours", type=float, default=DEFAULT_FAILURE_WINDOW_HOURS)
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()
    result = model_status(
        stale_after_hours=args.stale_after_hours,
        failure_window_hours=args.failure_window_hours,
    )
    print(json.dumps(result, indent=2, default=str))
    return 2 if args.fail_on_alert and not result["healthy"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
