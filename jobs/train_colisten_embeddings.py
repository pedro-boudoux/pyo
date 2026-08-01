"""Train immutable Phase 2 candidates without touching active song vectors.

The graph is treated as undirected because Last.fm similarity is an association,
even when only one direction has been crawled. Duplicate/reverse pairs collapse
to their strongest observed weight. Each attempt is recorded in ``model_runs``
and candidate vectors are stored in ``model_run_vectors``. The active
``songs.colisten_embedding`` and ``songs.hybrid_embedding`` columns are never
updated by this job.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from dotenv import load_dotenv
from psycopg2.extras import Json


def _preload_env_file(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--env-file" and index + 1 < len(argv):
            load_dotenv(argv[index + 1], override=True)
            return
        if arg.startswith("--env-file="):
            load_dotenv(arg.split("=", 1)[1], override=True)
            return


_preload_env_file(sys.argv[1:])

from app.config import (
    COLISTEN_BETA,
    COLISTEN_EMBEDDING_DIM,
    COLISTEN_MIN_AVG_DEGREE,
    COLISTEN_MIN_NODES,
    HYBRID_EMBEDDING_DIM,
)
from app.db import get_connection, get_cursor
from app.services.hybrid import compose


MODEL_NAME = "weighted_deepwalk_skipgram"
MODEL_LOCK_KEY = 0x50594F5F504832  # stable PostgreSQL advisory-lock key: "PYO_PH2"
DEFAULT_FIXTURE = Path("eval/ground_truth_colisten.json")
DEFAULT_VALIDATION_SAMPLE_SIZE = int(os.getenv("MODEL_VALIDATION_SAMPLE_SIZE", "50"))
DEFAULT_MIN_EVAL_COVERAGE = float(os.getenv("MODEL_MIN_EVAL_COVERAGE", "0.5"))
DEFAULT_MIN_RECALL_AT_K = float(os.getenv("MODEL_MIN_RECALL_AT_K", "0"))
DEFAULT_NORM_TOLERANCE = float(os.getenv("MODEL_NORM_TOLERANCE", "0.01"))
DEFAULT_CANDIDATE_RETENTION = max(
    2, int(os.getenv("MODEL_CANDIDATE_RETENTION", "2"))
)


class TrainingLockUnavailable(RuntimeError):
    """Raised when another trainer owns the database advisory lock."""


def density_gate_status(stats: dict) -> dict:
    ready = (
        stats["nodes"] >= COLISTEN_MIN_NODES
        and stats["avg_degree"] >= COLISTEN_MIN_AVG_DEGREE
    )
    return {
        **stats,
        "required_nodes": COLISTEN_MIN_NODES,
        "required_avg_degree": COLISTEN_MIN_AVG_DEGREE,
        "ready": ready,
    }


def collapse_undirected_edges(rows) -> dict[str, list[tuple[str, float]]]:
    """Collapse directed/provenance duplicates and return weighted adjacency."""
    pairs: dict[tuple[str, str], float] = {}
    for row in rows:
        source, target = row["source_track_id"], row["target_track_id"]
        if not source or not target or source == target:
            continue
        pair = (source, target) if source < target else (target, source)
        weight = max(float(row.get("weight") or 0.0), 1e-6)
        pairs[pair] = max(pairs.get(pair, 0.0), weight)

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (left, right), weight in pairs.items():
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    return dict(adjacency)


class WeightedWalks:
    def __init__(self, adjacency, *, walk_length: int, walks_per_node: int, seed: int):
        self.adjacency = adjacency
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.seed = seed

    def __iter__(self):
        nodes = list(self.adjacency)
        for pass_index in range(self.walks_per_node):
            rng = random.Random(self.seed + pass_index)
            rng.shuffle(nodes)
            for start in nodes:
                walk = [start]
                while len(walk) < self.walk_length:
                    choices = self.adjacency.get(walk[-1], ())
                    if not choices:
                        break
                    neighbors, weights = zip(*choices)
                    walk.append(rng.choices(neighbors, weights=weights, k=1)[0])
                yield walk


@contextmanager
def model_lock():
    """Hold one session-level advisory lock for the complete training attempt."""
    connection = get_connection()
    # register_vector() introspects pgvector types and leaves psycopg2 inside a
    # transaction. End that read-only setup transaction before enabling
    # autocommit for a session-level advisory lock.
    connection.rollback()
    connection.autocommit = True
    cursor = connection.cursor()
    acquired = False
    try:
        cursor.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (MODEL_LOCK_KEY,))
        acquired = bool(cursor.fetchone()["acquired"])
        if not acquired:
            raise TrainingLockUnavailable(
                "another Phase 2 trainer already holds the PostgreSQL advisory lock"
            )
        yield
    finally:
        if acquired:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (MODEL_LOCK_KEY,))
        cursor.close()
        connection.close()


def _create_run(params: dict) -> int:
    with get_cursor() as cursor:
        cursor.execute(
            """INSERT INTO model_runs
               (model, status, node_count, edge_count, dimension,
                hybrid_dimension, params, trained_at)
               VALUES (%s, 'running', 0, 0, %s, %s, %s, NULL)
               RETURNING id""",
            (
                MODEL_NAME,
                params["dimension"],
                HYBRID_EMBEDDING_DIM,
                Json(params),
            ),
        )
        return int(cursor.fetchone()["id"])


def _record_failed_attempt(message: str) -> int:
    with get_cursor() as cursor:
        cursor.execute(
            """INSERT INTO model_runs
               (model, status, node_count, edge_count, failure_details, finished_at)
               VALUES (%s, 'failed', 0, 0, %s, now()) RETURNING id""",
            (MODEL_NAME, message[:4000]),
        )
        return int(cursor.fetchone()["id"])


def _mark_run_failed(run_id: int, error: BaseException | str) -> None:
    message = str(error) or type(error).__name__
    with get_cursor() as cursor:
        cursor.execute(
            """UPDATE model_runs
               SET status = 'failed', failure_details = %s, finished_at = now()
               WHERE id = %s AND status = 'running'""",
            (message[:4000], run_id),
        )


def _fail_abandoned_runs() -> None:
    # Holding the advisory lock proves no legitimate trainer is still alive.
    # A row left in `running` therefore belongs to an interrupted process.
    with get_cursor() as cursor:
        cursor.execute(
            """UPDATE model_runs
               SET status = 'failed', finished_at = now(),
                   failure_details = COALESCE(
                       failure_details,
                       'trainer process ended before recording a terminal state'
                   )
               WHERE status = 'running'"""
        )


def _graph_snapshot() -> dict:
    """Capture a stable timestamp cutoff and density metadata for this run."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT COALESCE(MAX(created_at), now()) AS edge_cutoff FROM colisten_edges"
        )
        cutoff = cursor.fetchone()["edge_cutoff"]
        cursor.execute(
            "SELECT COUNT(*) AS edges FROM colisten_edges WHERE created_at <= %s",
            (cutoff,),
        )
        edges = int(cursor.fetchone()["edges"])
        cursor.execute(
            """SELECT COUNT(*) AS nodes FROM (
                   SELECT source_track_id AS track_id
                   FROM colisten_edges WHERE created_at <= %s
                   UNION
                   SELECT target_track_id
                   FROM colisten_edges WHERE created_at <= %s
               ) snapshot_nodes""",
            (cutoff, cutoff),
        )
        nodes = int(cursor.fetchone()["nodes"])
    return {
        "edge_cutoff": cutoff,
        "nodes": nodes,
        "edges": edges,
        "avg_degree": round((2 * edges) / nodes, 2) if nodes else 0.0,
    }


def _load_edges(edge_cutoff):
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT source_track_id, target_track_id, weight
               FROM colisten_edges WHERE created_at <= %s""",
            (edge_cutoff,),
        )
        return cursor.fetchall()


def _load_songs():
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT track_id, embedding FROM songs
               WHERE embedding IS NOT NULL AND vector_norm(embedding) > 0
               ORDER BY id"""
        )
        return cursor.fetchall()


def _record_snapshot(run_id: int, snapshot: dict, song_count: int) -> None:
    with get_cursor() as cursor:
        cursor.execute(
            """UPDATE model_runs
               SET edge_cutoff = %s, node_count = %s, edge_count = %s,
                   song_count = %s
               WHERE id = %s""",
            (
                snapshot["edge_cutoff"],
                snapshot["nodes"],
                snapshot["edges"],
                song_count,
                run_id,
            ),
        )


def _stage_song_vectors(
    run_id: int,
    model,
    songs: list,
    *,
    beta: float,
    batch_size: int,
) -> tuple[int, int]:
    staged = 0
    fallback = 0
    for offset in range(0, len(songs), batch_size):
        values = []
        for song in songs[offset : offset + batch_size]:
            track_id = song["track_id"]
            graph_vector = None
            if track_id in model.wv:
                raw = np.asarray(model.wv[track_id], dtype=float)
                norm = float(np.linalg.norm(raw))
                if norm and np.isfinite(norm):
                    graph_vector = (raw / norm).tolist()
            is_fallback = graph_vector is None
            fallback += int(is_fallback)
            hybrid_vector = compose(song["embedding"], graph_vector, beta)
            values.append(
                (run_id, track_id, graph_vector, hybrid_vector, is_fallback)
            )

        with get_cursor() as cursor:
            cursor.executemany(
                """INSERT INTO model_run_vectors
                   (model_run_id, track_id, colisten_embedding,
                    hybrid_embedding, tag_only_fallback)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (model_run_id, track_id) DO NOTHING""",
                values,
            )
        staged += len(values)
        print(
            f"staged {staged}/{len(songs)} candidate song vectors",
            flush=True,
        )
    return staged, fallback


def _train_model(
    adjacency,
    *,
    dimension: int,
    walk_length: int,
    walks_per_node: int,
    window: int,
    epochs: int,
    workers: int,
    seed: int,
):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("install requirements-jobs.txt before training") from exc

    walks = WeightedWalks(
        adjacency,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        seed=seed,
    )
    return Word2Vec(
        sentences=walks,
        vector_size=dimension,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )


def train(
    *,
    dimension: int = COLISTEN_EMBEDDING_DIM,
    walk_length: int = 40,
    walks_per_node: int = 5,
    window: int = 10,
    epochs: int = 5,
    workers: int = 1,
    seed: int = 42,
    beta: float = COLISTEN_BETA,
    batch_size: int = 1000,
    allow_sparse: bool = False,
    dry_run: bool = False,
    model_out: str | None = None,
) -> dict:
    """Build a candidate run while leaving active recommendation rows intact."""
    params = {
        "algorithm": MODEL_NAME,
        "dimension": dimension,
        "hybrid_dimension": HYBRID_EMBEDDING_DIM,
        "walk_length": walk_length,
        "walks_per_node": walks_per_node,
        "window": window,
        "epochs": epochs,
        "workers": workers,
        "seed": seed,
        "beta": beta,
        "batch_size": batch_size,
        "undirected_pair_weight": "max",
    }

    try:
        with model_lock():
            _fail_abandoned_runs()
            run_id = _create_run(params)
            try:
                snapshot = _graph_snapshot()
                songs = _load_songs()
                _record_snapshot(run_id, snapshot, len(songs))
                gate = density_gate_status(
                    {
                        key: snapshot[key]
                        for key in ("nodes", "edges", "avg_degree")
                    }
                )
                if not allow_sparse and not gate["ready"]:
                    raise RuntimeError(
                        "co-listening density gate not met: "
                        f"nodes={gate['nodes']} (need {COLISTEN_MIN_NODES}), "
                        f"avg_degree={gate['avg_degree']} "
                        f"(need {COLISTEN_MIN_AVG_DEGREE})"
                    )
                if dimension != COLISTEN_EMBEDDING_DIM:
                    raise ValueError(
                        f"candidate dimension must be {COLISTEN_EMBEDDING_DIM}, "
                        f"got {dimension}"
                    )

                adjacency = collapse_undirected_edges(
                    _load_edges(snapshot["edge_cutoff"])
                )
                model = _train_model(
                    adjacency,
                    dimension=dimension,
                    walk_length=walk_length,
                    walks_per_node=walks_per_node,
                    window=window,
                    workers=workers,
                    epochs=epochs,
                    seed=seed,
                )
                if model_out:
                    model.save(model_out)

                staged = fallback = 0
                status = "failed" if dry_run else "candidate"
                failure = "dry run: no candidate vectors were stored" if dry_run else None
                if not dry_run:
                    staged, fallback = _stage_song_vectors(
                        run_id,
                        model,
                        songs,
                        beta=beta,
                        batch_size=batch_size,
                    )
                with get_cursor() as cursor:
                    cursor.execute(
                        """UPDATE model_runs
                           SET status = %s, trained_at = now(), finished_at = now(),
                               songs_updated = %s, fallback_count = %s,
                               failure_details = %s
                           WHERE id = %s""",
                        (status, staged, fallback, failure, run_id),
                    )
                return {
                    **gate,
                    "run_id": run_id,
                    "status": status,
                    "songs_updated": staged,
                    "tag_only_fallbacks": fallback,
                    "params": params,
                }
            except BaseException as exc:
                _mark_run_failed(run_id, exc)
                raise
    except TrainingLockUnavailable as exc:
        failed_run_id = _record_failed_attempt(str(exc))
        raise TrainingLockUnavailable(
            f"{exc} (recorded as failed run {failed_run_id})"
        ) from exc


def _candidate_stats(run_id: int, norm_tolerance: float) -> dict:
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT
                   COUNT(*) AS candidate_count,
                   COUNT(*) FILTER (WHERE tag_only_fallback) AS fallback_count,
                   COUNT(*) FILTER (
                       WHERE vector_dims(hybrid_embedding) <> %s
                   ) AS bad_hybrid_dimensions,
                   COUNT(*) FILTER (
                       WHERE colisten_embedding IS NOT NULL
                         AND vector_dims(colisten_embedding) <> %s
                   ) AS bad_colisten_dimensions,
                   COUNT(*) FILTER (
                       WHERE NOT (
                           vector_norm(hybrid_embedding) >= 0
                           AND vector_norm(hybrid_embedding) < 'Infinity'::float8
                       )
                   ) AS nonfinite_hybrid_vectors,
                   COUNT(*) FILTER (
                       WHERE colisten_embedding IS NOT NULL AND NOT (
                           vector_norm(colisten_embedding) >= 0
                           AND vector_norm(colisten_embedding) < 'Infinity'::float8
                       )
                   ) AS nonfinite_colisten_vectors,
                   COUNT(*) FILTER (
                       WHERE abs(vector_norm(hybrid_embedding) - 1.0) > %s
                   ) AS bad_hybrid_norms,
                   COUNT(*) FILTER (
                       WHERE colisten_embedding IS NOT NULL
                         AND abs(vector_norm(colisten_embedding) - 1.0) > %s
                   ) AS bad_colisten_norms,
                   COUNT(*) FILTER (
                       WHERE tag_only_fallback <> (colisten_embedding IS NULL)
                   ) AS fallback_mismatches
               FROM model_run_vectors WHERE model_run_id = %s""",
            (
                HYBRID_EMBEDDING_DIM,
                COLISTEN_EMBEDDING_DIM,
                norm_tolerance,
                norm_tolerance,
                run_id,
            ),
        )
        return dict(cursor.fetchone())


def _independent_eval(
    run_id: int,
    fixture_path: str | Path,
    *,
    sample_size: int,
    k: int,
) -> dict:
    path = Path(fixture_path)
    if not path.exists():
        raise RuntimeError(f"independent evaluation fixture not found: {path}")
    with path.open() as handle:
        fixture = json.load(handle)
    if not fixture.get("independent_from_lastfm"):
        raise RuntimeError("model validation fixture must be independent from Last.fm")

    seeds = sorted(
        fixture.get("seeds", []), key=lambda item: item["seed_track_id"]
    )[:sample_size]
    recalls = []
    reciprocal_ranks = []
    for seed in seeds:
        seed_id = seed["seed_track_id"]
        with get_cursor() as cursor:
            cursor.execute(
                """SELECT candidate.track_id
                   FROM model_run_vectors candidate
                   WHERE candidate.model_run_id = %s
                     AND candidate.track_id <> %s
                     AND EXISTS (
                         SELECT 1 FROM model_run_vectors seed
                         WHERE seed.model_run_id = %s AND seed.track_id = %s
                     )
                   ORDER BY candidate.hybrid_embedding <=> (
                       SELECT seed.hybrid_embedding FROM model_run_vectors seed
                       WHERE seed.model_run_id = %s AND seed.track_id = %s
                   )
                   LIMIT %s""",
                (run_id, seed_id, run_id, seed_id, run_id, seed_id, k),
            )
            recommendations = [row["track_id"] for row in cursor.fetchall()]
        if not recommendations:
            continue
        targets = set(seed.get("targets", []))
        hit_ranks = [
            rank
            for rank, track_id in enumerate(recommendations, start=1)
            if track_id in targets
        ]
        recalls.append(
            len(hit_ranks) / min(len(targets), k) if targets else 0.0
        )
        reciprocal_ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

    scored = len(recalls)
    return {
        "fixture": str(path),
        "sample_size": len(seeds),
        "seeds_scored": scored,
        "coverage": scored / len(seeds) if seeds else 0.0,
        f"recall_at_{k}": sum(recalls) / scored if scored else 0.0,
        "mrr": sum(reciprocal_ranks) / scored if scored else 0.0,
    }


def validate(
    run_id: int,
    *,
    fixture_path: str | Path = DEFAULT_FIXTURE,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    k: int = 10,
    min_eval_coverage: float = DEFAULT_MIN_EVAL_COVERAGE,
    min_recall_at_k: float = DEFAULT_MIN_RECALL_AT_K,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict:
    """Validate a complete candidate without touching active song vectors."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM model_runs WHERE id = %s", (run_id,))
        run = cursor.fetchone()
    if not run:
        raise RuntimeError(f"model run {run_id} does not exist")
    if run["status"] not in ("candidate", "validated"):
        raise RuntimeError(
            f"model run {run_id} has status {run['status']}; expected candidate"
        )

    try:
        stats = _candidate_stats(run_id, norm_tolerance)
        expected = int(run.get("song_count") or 0)
        candidate_count = int(stats["candidate_count"] or 0)
        candidate_coverage = candidate_count / expected if expected else 0.0
        independent = _independent_eval(
            run_id, fixture_path, sample_size=sample_size, k=k
        )
        recall_key = f"recall_at_{k}"
        node_count = int(run["node_count"] or 0)
        edge_count = int(run["edge_count"] or 0)
        avg_degree = (2 * edge_count / node_count) if node_count else 0.0
        gates = {
            "density": (
                node_count >= COLISTEN_MIN_NODES
                and avg_degree >= COLISTEN_MIN_AVG_DEGREE
            ),
            "complete_candidate": expected > 0 and candidate_count == expected,
            "dimensions": not (
                stats["bad_hybrid_dimensions"]
                or stats["bad_colisten_dimensions"]
            ),
            "finite_values": not (
                stats["nonfinite_hybrid_vectors"]
                or stats["nonfinite_colisten_vectors"]
            ),
            "normalization": not (
                stats["bad_hybrid_norms"] or stats["bad_colisten_norms"]
            ),
            "tag_only_fallbacks": (
                not stats["fallback_mismatches"]
                and int(stats["fallback_count"] or 0)
                == int(run.get("fallback_count") or 0)
            ),
            "independent_eval_coverage": (
                independent["coverage"] >= min_eval_coverage
            ),
            "independent_eval_quality": (
                independent[recall_key] >= min_recall_at_k
            ),
        }
        report = {
            "run_id": run_id,
            "passed": all(gates.values()),
            "gates": gates,
            "candidate": {
                **stats,
                "expected_song_count": expected,
                "coverage": candidate_coverage,
            },
            "graph": {
                "nodes": node_count,
                "edges": edge_count,
                "avg_degree": avg_degree,
                "edge_cutoff": str(run.get("edge_cutoff")),
            },
            "independent_eval": independent,
            "thresholds": {
                "min_eval_coverage": min_eval_coverage,
                "min_recall_at_k": min_recall_at_k,
                "norm_tolerance": norm_tolerance,
            },
        }
        with get_cursor() as cursor:
            if report["passed"]:
                cursor.execute(
                    """UPDATE model_runs
                       SET status = 'validated', validated_at = now(),
                           finished_at = now(), validation = %s,
                           failure_details = NULL
                       WHERE id = %s""",
                    (Json(report), run_id),
                )
            else:
                cursor.execute(
                    """UPDATE model_runs
                       SET status = 'failed', finished_at = now(), validation = %s,
                           failure_details = 'candidate validation failed'
                       WHERE id = %s""",
                    (Json(report), run_id),
                )
        if not report["passed"]:
            failed = [name for name, passed in gates.items() if not passed]
            raise RuntimeError(
                f"candidate validation failed for model run {run_id}: "
                + ", ".join(failed)
            )
        return report
    except BaseException as exc:
        with get_cursor() as cursor:
            cursor.execute(
                """UPDATE model_runs SET status = 'failed', finished_at = now(),
                       failure_details = COALESCE(failure_details, %s)
                   WHERE id = %s AND status <> 'failed'""",
                ((str(exc) or type(exc).__name__)[:4000], run_id),
            )
        raise


def _prune_candidate_vectors(retain: int = DEFAULT_CANDIDATE_RETENTION) -> None:
    with get_cursor() as cursor:
        cursor.execute(
            """DELETE FROM model_run_vectors
               WHERE model_run_id NOT IN (
                   SELECT id FROM model_runs
                   WHERE status IN ('active', 'validated', 'superseded')
                   ORDER BY COALESCE(
                       published_at, validated_at, trained_at, started_at
                   ) DESC, id DESC
                   LIMIT %s
               )""",
            (max(2, retain),),
        )


def _replace_active_vectors(cursor, run_id: int) -> int:
    """Make active song vectors exactly match one immutable candidate snapshot."""
    cursor.execute(
        """UPDATE songs AS song
           SET colisten_embedding = NULL, hybrid_embedding = NULL
           WHERE (song.colisten_embedding IS NOT NULL
                  OR song.hybrid_embedding IS NOT NULL)
             AND NOT EXISTS (
                 SELECT 1 FROM model_run_vectors AS candidate
                 WHERE candidate.model_run_id = %s
                   AND candidate.track_id = song.track_id
             )""",
        (run_id,),
    )
    cursor.execute(
        """UPDATE songs AS song
           SET colisten_embedding = candidate.colisten_embedding,
               hybrid_embedding = candidate.hybrid_embedding
           FROM model_run_vectors AS candidate
           WHERE candidate.model_run_id = %s
             AND candidate.track_id = song.track_id""",
        (run_id,),
    )
    return cursor.rowcount


def publish(run_id: int) -> dict:
    """Atomically replace active vectors with one validated candidate."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (MODEL_LOCK_KEY,)
        )
        if not cursor.fetchone()["acquired"]:
            raise TrainingLockUnavailable(
                "another Phase 2 model operation is already running"
            )
        cursor.execute(
            "SELECT * FROM model_runs WHERE id = %s FOR UPDATE", (run_id,)
        )
        run = cursor.fetchone()
        if not run:
            raise RuntimeError(f"model run {run_id} does not exist")
        if run["status"] == "active":
            connection.commit()
            return {
                "run_id": run_id,
                "status": "active",
                "songs_updated": run["songs_updated"],
            }
        if run["status"] != "validated":
            raise RuntimeError(
                f"model run {run_id} has status {run['status']}; expected validated"
            )

        cursor.execute(
            "SELECT COUNT(*) AS count FROM model_run_vectors WHERE model_run_id = %s",
            (run_id,),
        )
        candidate_count = int(cursor.fetchone()["count"])
        if candidate_count != int(run["song_count"] or 0):
            raise RuntimeError(
                "candidate changed after validation: "
                f"{candidate_count} rows, expected {run['song_count']}"
            )
        cursor.execute("SELECT id FROM model_runs WHERE status = 'active' FOR UPDATE")
        previous = cursor.fetchone()
        previous_id = int(previous["id"]) if previous else None

        updated = _replace_active_vectors(cursor, run_id)
        if updated != candidate_count:
            raise RuntimeError(
                f"atomic publish updated {updated} songs, expected {candidate_count}"
            )
        if previous_id is not None:
            cursor.execute(
                """UPDATE model_runs SET status = 'superseded', finished_at = now()
                   WHERE id = %s""",
                (previous_id,),
            )
        cursor.execute(
            """UPDATE model_runs
               SET status = 'active', published_at = now(), finished_at = now(),
                   songs_updated = %s, previous_active_run_id = %s
               WHERE id = %s""",
            (updated, previous_id, run_id),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    _prune_candidate_vectors()
    return {
        "run_id": run_id,
        "status": "active",
        "songs_updated": updated,
        "previous_active_run_id": previous_id,
    }


def rollback(target_run_id: int | None = None) -> dict:
    """Atomically republish the previous retained successful candidate."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (MODEL_LOCK_KEY,)
        )
        if not cursor.fetchone()["acquired"]:
            raise TrainingLockUnavailable(
                "another Phase 2 model operation is already running"
            )
        cursor.execute("SELECT * FROM model_runs WHERE status = 'active' FOR UPDATE")
        current = cursor.fetchone()
        if not current:
            raise RuntimeError("there is no active model run to roll back")
        current_id = int(current["id"])
        if target_run_id is None:
            target_run_id = current.get("previous_active_run_id")
        if target_run_id is None:
            cursor.execute(
                """SELECT run.id FROM model_runs run
                   WHERE run.status = 'superseded' AND run.id <> %s
                     AND EXISTS (
                         SELECT 1 FROM model_run_vectors vector
                         WHERE vector.model_run_id = run.id
                     )
                   ORDER BY COALESCE(
                       run.published_at, run.validated_at, run.trained_at
                   ) DESC
                   LIMIT 1""",
                (current_id,),
            )
            prior = cursor.fetchone()
            target_run_id = prior["id"] if prior else None
        if target_run_id is None:
            raise RuntimeError(
                "no retained previous candidate is available for rollback"
            )

        cursor.execute(
            "SELECT * FROM model_runs WHERE id = %s FOR UPDATE", (target_run_id,)
        )
        target = cursor.fetchone()
        if not target or target["status"] not in ("superseded", "validated"):
            raise RuntimeError(
                f"model run {target_run_id} is not a rollback candidate"
            )
        cursor.execute(
            "SELECT COUNT(*) AS count FROM model_run_vectors WHERE model_run_id = %s",
            (target_run_id,),
        )
        expected = int(cursor.fetchone()["count"])
        if not expected:
            raise RuntimeError(
                f"model run {target_run_id} has no retained candidate vectors"
            )

        updated = _replace_active_vectors(cursor, target_run_id)
        if updated != expected:
            raise RuntimeError(
                f"rollback updated {updated} songs, expected {expected}"
            )
        cursor.execute(
            """UPDATE model_runs SET status = 'superseded', finished_at = now()
               WHERE id = %s""",
            (current_id,),
        )
        cursor.execute(
            """UPDATE model_runs
               SET status = 'active', published_at = now(), finished_at = now(),
                   previous_active_run_id = %s, songs_updated = %s
               WHERE id = %s""",
            (current_id, updated, target_run_id),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()
    return {
        "run_id": int(target_run_id),
        "status": "active",
        "rolled_back_from": current_id,
        "songs_updated": updated,
    }


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--walk-length", type=int, default=40)
    parser.add_argument("--walks-per-node", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=COLISTEN_BETA)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--allow-sparse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-out", default=None)


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_VALIDATION_SAMPLE_SIZE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-eval-coverage", type=float, default=DEFAULT_MIN_EVAL_COVERAGE)
    parser.add_argument("--min-recall-at-k", type=float, default=DEFAULT_MIN_RECALL_AT_K)
    parser.add_argument("--norm-tolerance", type=float, default=DEFAULT_NORM_TOLERANCE)


def _training_kwargs(args) -> dict:
    return {
        "walk_length": args.walk_length,
        "walks_per_node": args.walks_per_node,
        "window": args.window,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "allow_sparse": args.allow_sparse,
        "dry_run": args.dry_run,
        "model_out": args.model_out,
    }


def _validation_kwargs(args) -> dict:
    return {
        "fixture_path": args.fixture,
        "sample_size": args.sample_size,
        "k": args.k,
        "min_eval_coverage": args.min_eval_coverage,
        "min_recall_at_k": args.min_recall_at_k,
        "norm_tolerance": args.norm_tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage safe Phase 2 model runs.")
    parser.add_argument("--env-file", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-density", help="check the graph density gate")
    train_parser = subparsers.add_parser("train", help="build an immutable candidate")
    _add_training_arguments(train_parser)
    validate_parser = subparsers.add_parser("validate", help="validate a candidate")
    validate_parser.add_argument("run_id", type=int)
    _add_validation_arguments(validate_parser)
    publish_parser = subparsers.add_parser(
        "publish", help="atomically activate a validated candidate"
    )
    publish_parser.add_argument("run_id", type=int)
    rollback_parser = subparsers.add_parser(
        "rollback", help="atomically republish a retained successful run"
    )
    rollback_parser.add_argument("run_id", type=int, nargs="?", default=None)

    args = parser.parse_args()
    if args.command == "check-density":
        snapshot = _graph_snapshot()
        gate = density_gate_status(
            {key: snapshot[key] for key in ("nodes", "edges", "avg_degree")}
        )
        print(json.dumps(gate, indent=2))
        return 0 if gate["ready"] else 2
    if args.command == "train":
        result = train(**_training_kwargs(args))
    elif args.command == "validate":
        result = validate(args.run_id, **_validation_kwargs(args))
    elif args.command == "publish":
        result = publish(args.run_id)
    else:
        result = rollback(args.run_id)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Phase 2 model operation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
