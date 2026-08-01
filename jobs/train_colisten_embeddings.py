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
            "SELECT track_id, embedding FROM songs WHERE embedding IS NOT NULL ORDER BY id"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a staged Phase 2 co-listening candidate."
    )
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--walk-length", type=int, default=40)
    parser.add_argument("--walks-per-node", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=COLISTEN_BETA)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--allow-sparse", action="store_true")
    parser.add_argument(
        "--check-density",
        action="store_true",
        help="print gate status and exit without loading edges or training",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()
    if args.check_density:
        snapshot = _graph_snapshot()
        gate = density_gate_status(
            {key: snapshot[key] for key in ("nodes", "edges", "avg_degree")}
        )
        print(json.dumps(gate, indent=2))
        return 0 if gate["ready"] else 2
    result = train(
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        window=args.window,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        beta=args.beta,
        batch_size=args.batch_size,
        allow_sparse=args.allow_sparse,
        dry_run=args.dry_run,
        model_out=args.model_out,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Phase 2 candidate training failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
