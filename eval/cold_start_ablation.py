"""Run and compare graph-seeding cold-start ablations (GitHub issue #35).

Each ``run`` invocation evaluates one mechanism combination. Run every variant
from the same freshly restored disposable database snapshot; candidate ingestion
writes cache rows, so running variants sequentially against one database would
contaminate latency, call-count, and ANN measurements.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

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

from app.config import DEFAULT_K
from app.db import get_cursor
from app.services import hybrid, lastfm, tag_encoder
from app.services.seed_discovery import (
    SeedDiscoveryOptions,
    discover_seed_candidates,
)
from app.services.vector_utils import to_float_list
from eval.metrics import mrr, recall_at_k


HERE = Path(__file__).resolve().parent
DEFAULT_INDEPENDENT_FIXTURE = HERE / "ground_truth_colisten.json"
DEFAULT_COLD_FIXTURE = HERE / "cold_start_seeds.json"
LEGACY_LISTENER_CAP = 500_000
VARIANTS = {
    "full": SeedDiscoveryOptions(True, True, False),
    "no_expansion": SeedDiscoveryOptions(False, True, False),
    "no_artist_fallback": SeedDiscoveryOptions(True, False, False),
    "minimal": SeedDiscoveryOptions(False, False, False),
}


def _load_json(path: str | Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _database_snapshot() -> dict:
    """Fingerprint the pre-run state so matrix comparison catches contamination."""
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT COUNT(*)::bigint AS songs,
                      COUNT(*) FILTER (WHERE embedding IS NOT NULL)::bigint AS embedded,
                      COALESCE(MAX(created_at)::text, '') AS newest_song
               FROM songs"""
        )
        songs = cursor.fetchone()
        cursor.execute(
            """SELECT COUNT(*)::bigint AS edges,
                      COALESCE(MAX(created_at)::text, '') AS newest_edge
               FROM colisten_edges"""
        )
        edges = cursor.fetchone()
    return {
        "songs": int(songs["songs"]),
        "embedded_songs": int(songs["embedded"]),
        "newest_song": songs["newest_song"],
        "colisten_edges": int(edges["edges"]),
        "newest_colisten_edge": edges["newest_edge"],
        "recommendation_model": hybrid.active_embedding_column(),
    }


def _load_seed(track_id: str) -> dict | None:
    embedding_column = hybrid.active_embedding_column()
    with get_cursor() as cursor:
        cursor.execute(
            f"""SELECT track_id, name, artist, listeners,
                       {embedding_column} AS embedding
                FROM songs WHERE track_id = %s""",
            (track_id,),
        )
        row = cursor.fetchone()
    if not row or row["embedding"] is None:
        return None
    return {**dict(row), "embedding": to_float_list(row["embedding"])}


@contextmanager
def _count_lastfm_requests():
    """Count every Last.fm HTTP method, including calls made by ingestion."""
    original = lastfm._request
    counts: Counter[str] = Counter()
    timing: defaultdict[str, float] = defaultdict(float)

    def counted(method: str, **params):
        started = perf_counter()
        counts[method] += 1
        try:
            return original(method, **params)
        finally:
            timing[method] += (perf_counter() - started) * 1000

    lastfm._request = counted
    try:
        yield counts, timing
    finally:
        lastfm._request = original


def _entries(
    independent_path: str | Path,
    cold_path: str | Path,
    independent_limit: int,
) -> list[dict]:
    independent = sorted(
        _load_json(independent_path)["seeds"],
        key=lambda entry: entry["seed_track_id"],
    )[:independent_limit]
    independent = [
        {**entry, "cohort": "independent_quality"} for entry in independent
    ]
    cold = _load_json(cold_path)["seeds"]
    return independent + cold


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _aggregate(records: list[dict], errors: list[dict], *, k: int) -> dict:
    quality = [record for record in records if record.get("targets")]
    no_similar = [record for record in records if record["cohort"] == "no_similar"]
    obscure = [
        record
        for record in records
        if record["cohort"] in {"no_similar", "obscure_warm"}
    ]
    all_candidates = [
        candidate
        for record in records
        for candidate in record["candidates"]
    ]
    total_calls: Counter[str] = Counter()
    for record in records:
        total_calls.update(record["lastfm_calls"])

    def coverage(rows: list[dict], threshold: int) -> float:
        if not rows:
            return 0.0
        return round(
            sum(len(row["candidates"]) >= threshold for row in rows) / len(rows),
            4,
        )

    return {
        "seeds_total": len(records) + len(errors),
        "seeds_scored": len(records),
        "errors": errors,
        "seed_coverage": round(
            len(records) / (len(records) + len(errors)), 4
        ) if records or errors else 0.0,
        "independent_quality": {
            "seeds": len(quality),
            f"recall_at_{k}": _mean([record["recall"] for record in quality]),
            "mrr": _mean([record["mrr"] for record in quality]),
        },
        "recommendation_coverage": {
            "any": coverage(records, 1),
            f"at_least_{k}": coverage(records, k),
            "mean_candidates": _mean(
                [len(record["candidates"]) for record in records]
            ),
        },
        "cold_seed_coverage": {
            "seeds": len(no_similar),
            "any": coverage(no_similar, 1),
            f"at_least_{k}": coverage(no_similar, k),
            "mean_candidates": _mean(
                [len(record["candidates"]) for record in no_similar]
            ),
        },
        "obscure_seed_coverage": {
            "seeds": len(obscure),
            "any": coverage(obscure, 1),
            "mean_candidates": _mean(
                [len(record["candidates"]) for record in obscure]
            ),
        },
        "lastfm": {
            "total_calls": sum(total_calls.values()),
            "calls_by_method": dict(sorted(total_calls.items())),
            "mean_calls_per_seed": _mean(
                [sum(record["lastfm_calls"].values()) for record in records]
            ),
        },
        "seed_latency_ms": {
            "mean": _mean([record["latency_ms"] for record in records]),
            "median": round(
                statistics.median([record["latency_ms"] for record in records]),
                4,
            ) if records else 0.0,
        },
        "graph_branching": {
            "mean_degree": _mean(
                [len(record["candidates"]) for record in records]
            ),
            "mean_unique_artists": _mean(
                [record["unique_artists"] for record in records]
            ),
            "mean_expansion_selected": _mean(
                [
                    record["selected_by_source"].get("recursive_expansion", 0)
                    for record in records
                ]
            ),
        },
        "mechanism_activity": {
            "recursive_expansion_selected": sum(
                record["selected_by_source"].get("recursive_expansion", 0)
                for record in records
            ),
            "artist_fallback_attempted_seeds": sum(
                bool(record.get("fallback_attempted")) for record in records
            ),
            "artist_fallback_selected": sum(
                record["selected_by_source"].get("artist_fallback", 0)
                for record in records
            ),
        },
        "listener_policy": {
            "mode": "uncapped",
            "legacy_cap": LEGACY_LISTENER_CAP,
            "returned_above_legacy_cap": sum(
                (candidate.get("listeners") or 0) >= LEGACY_LISTENER_CAP
                for candidate in all_candidates
            ),
            "max_returned_listeners": max(
                (candidate.get("listeners") or 0 for candidate in all_candidates),
                default=0,
            ),
        },
    }


def run_variant(
    variant: str,
    *,
    independent_path: str | Path = DEFAULT_INDEPENDENT_FIXTURE,
    cold_path: str | Path = DEFAULT_COLD_FIXTURE,
    independent_limit: int = 25,
    k: int = DEFAULT_K,
    progress: bool = True,
) -> dict:
    if not lastfm.LASTFM_API_KEY:
        raise RuntimeError("LASTFM_API_KEY is required for the ablation run")
    options = VARIANTS[variant]
    # Production keeps the ONNX session warm. Initialize it before the timed
    # seed loop so a one-time model load/download cannot bias the first variant.
    tag_encoder._get_model()
    snapshot = _database_snapshot()
    entries = _entries(independent_path, cold_path, independent_limit)
    records = []
    errors = []

    for index, entry in enumerate(entries, start=1):
        seed = _load_seed(entry["seed_track_id"])
        if not seed:
            errors.append(
                {
                    "seed_track_id": entry["seed_track_id"],
                    "error": "seed missing or active embedding is null",
                }
            )
            continue
        started = perf_counter()
        try:
            with _count_lastfm_requests() as (calls, upstream_timing):
                discovery = discover_seed_candidates(
                    track_id=seed["track_id"],
                    artist=seed["artist"],
                    name=seed["name"],
                    vector=seed["embedding"],
                    options=options,
                    limit=k,
                )
        except Exception as exc:
            errors.append(
                {
                    "seed_track_id": entry["seed_track_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        candidate_ids = [candidate["track_id"] for candidate in discovery.candidates]
        targets = set(entry.get("targets", []))
        records.append(
            {
                "seed_track_id": seed["track_id"],
                "name": seed["name"],
                "artist": seed["artist"],
                "seed_listeners": seed["listeners"],
                "cohort": entry["cohort"],
                "targets": sorted(targets),
                "recall": recall_at_k(candidate_ids, targets, k) if targets else None,
                "mrr": mrr(candidate_ids, targets) if targets else None,
                "latency_ms": round((perf_counter() - started) * 1000, 3),
                "lastfm_calls": dict(sorted(calls.items())),
                "lastfm_timing_ms": {
                    method: round(value, 3)
                    for method, value in sorted(upstream_timing.items())
                },
                "unique_artists": len(
                    {
                        candidate["artist"].strip().casefold()
                        for candidate in discovery.candidates
                    }
                ),
                "selected_by_source": discovery.selected_by_source,
                "fallback_attempted": discovery.fallback_attempted,
                "candidates": [
                    {
                        key: candidate.get(key)
                        for key in (
                            "track_id",
                            "name",
                            "artist",
                            "listeners",
                            "similarity",
                            "source",
                        )
                    }
                    for candidate in discovery.candidates
                ],
            }
        )
        if progress:
            print(
                f"[{index}/{len(entries)}] {entry['cohort']}: "
                f"{seed['artist']} — {seed['name']}",
                file=sys.stderr,
                flush=True,
            )

    return {
        "issue": 35,
        "variant": variant,
        "options": {
            "recursive_expansion": options.recursive_expansion,
            "artist_fallback": options.artist_fallback,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_snapshot": snapshot,
        "fixtures": {
            "independent": _display_path(independent_path),
            "curated_cold": _display_path(cold_path),
            "independent_limit": independent_limit,
        },
        "k": k,
        "summary": _aggregate(records, errors, k=k),
        "seeds": records,
    }


def compare_results(
    results: list[dict],
    *,
    max_quality_loss: float = 0.02,
    max_coverage_loss: float = 0.05,
) -> dict:
    by_variant = {result["variant"]: result for result in results}
    missing = sorted(set(VARIANTS) - set(by_variant))
    if missing:
        raise ValueError(f"missing variants: {', '.join(missing)}")
    snapshots = [result["database_snapshot"] for result in results]
    if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        raise ValueError(
            "database snapshots differ; restore the same snapshot before every run"
        )

    baseline = by_variant["full"]["summary"]

    def decision(candidate_name: str, mechanism: str) -> dict:
        candidate = by_variant[candidate_name]["summary"]
        k = by_variant[candidate_name]["k"]
        quality_key = f"recall_at_{k}"
        recall_delta = round(
            candidate["independent_quality"][quality_key]
            - baseline["independent_quality"][quality_key],
            4,
        )
        mrr_delta = round(
            candidate["independent_quality"]["mrr"]
            - baseline["independent_quality"]["mrr"],
            4,
        )
        cold_delta = round(
            candidate["cold_seed_coverage"]["any"]
            - baseline["cold_seed_coverage"]["any"],
            4,
        )
        eligible = (
            recall_delta >= -max_quality_loss
            and mrr_delta >= -max_quality_loss
            and cold_delta >= -max_coverage_loss
        )
        activity = baseline.get("mechanism_activity", {})
        control_activity = (
            {
                "selected_candidates": activity.get(
                    "recursive_expansion_selected", 0
                )
            }
            if mechanism == "recursive_expansion"
            else {
                "attempted_seeds": activity.get(
                    "artist_fallback_attempted_seeds", 0
                ),
                "selected_candidates": activity.get(
                    "artist_fallback_selected", 0
                ),
            }
        )
        return {
            "variant": candidate_name,
            "control_activity": control_activity,
            "recall_delta": recall_delta,
            "mrr_delta": mrr_delta,
            "cold_any_coverage_delta": cold_delta,
            "lastfm_calls_delta": candidate["lastfm"]["total_calls"]
            - baseline["lastfm"]["total_calls"],
            "mean_latency_ms_delta": round(
                candidate["seed_latency_ms"]["mean"]
                - baseline["seed_latency_ms"]["mean"],
                3,
            ),
            "eligible_for_removal": eligible,
        }

    return {
        "baseline": "full",
        "thresholds": {
            "max_absolute_quality_loss": max_quality_loss,
            "max_absolute_cold_coverage_loss": max_coverage_loss,
        },
        "recursive_expansion": decision(
            "no_expansion", "recursive_expansion"
        ),
        "artist_fallback": decision(
            "no_artist_fallback", "artist_fallback"
        ),
        "minimal": decision("minimal", "artist_fallback"),
        "listener_policy": baseline["listener_policy"],
    }


def _write_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="measure one ablation variant")
    run.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    run.add_argument("--env-file")
    run.add_argument("--independent-fixture", default=str(DEFAULT_INDEPENDENT_FIXTURE))
    run.add_argument("--cold-fixture", default=str(DEFAULT_COLD_FIXTURE))
    run.add_argument("--independent-limit", type=int, default=25)
    run.add_argument("--k", type=int, default=DEFAULT_K)
    run.add_argument("--out", required=True)
    run.add_argument(
        "--allow-db-writes",
        action="store_true",
        help="required acknowledgement: ingestion writes to a disposable DB copy",
    )

    compare = subparsers.add_parser("compare", help="gate a four-result matrix")
    compare.add_argument("results", nargs=4)
    compare.add_argument("--max-quality-loss", type=float, default=0.02)
    compare.add_argument("--max-coverage-loss", type=float, default=0.05)
    compare.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "run":
        if not args.allow_db_writes:
            parser.error(
                "run requires --allow-db-writes and a disposable database snapshot"
            )
        result = run_variant(
            args.variant,
            independent_path=args.independent_fixture,
            cold_path=args.cold_fixture,
            independent_limit=args.independent_limit,
            k=args.k,
        )
        _write_json(args.out, result)
        print(json.dumps(result["summary"], indent=2))
        return 0 if result["summary"]["seed_coverage"] >= 0.95 else 2

    results = [_load_json(path) for path in args.results]
    comparison = compare_results(
        results,
        max_quality_loss=args.max_quality_loss,
        max_coverage_loss=args.max_coverage_loss,
    )
    _write_json(args.out, comparison)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
