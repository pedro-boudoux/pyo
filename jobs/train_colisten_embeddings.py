"""Train weighted random-walk embeddings from colisten_edges (Phase 2, task 14).

The graph is treated as undirected because Last.fm similarity is an association,
even when only one direction has been crawled. Duplicate/reverse pairs collapse to
their strongest observed weight. Gensim Word2Vec trains skip-gram vectors over
weighted DeepWalk-style walks; this satisfies the node2vec/random-walk work order
without materializing a large NetworkX graph.
"""

import argparse
from collections import defaultdict
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
)
from app.db import get_cursor
from app.services import colisten
from app.services.hybrid import compose


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


def _load_edges():
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT source_track_id, target_track_id, weight FROM colisten_edges"
        )
        return cursor.fetchall()


def _write_song_vectors(model, *, beta: float, batch_size: int = 1000) -> tuple[int, int]:
    with get_cursor() as cursor:
        cursor.execute("SELECT track_id, embedding FROM songs WHERE embedding IS NOT NULL")
        songs = cursor.fetchall()

    updated = 0
    fallback = 0
    for offset in range(0, len(songs), batch_size):
        values = []
        for song in songs[offset : offset + batch_size]:
            track_id = song["track_id"]
            graph_vector = np.asarray(model.wv[track_id], dtype=float) if track_id in model.wv else None
            if graph_vector is None:
                fallback += 1
            hybrid_vector = compose(song["embedding"], graph_vector, beta)
            values.append((graph_vector, hybrid_vector, track_id))
        with get_cursor() as cursor:
            cursor.executemany(
                """UPDATE songs
                   SET colisten_embedding = %s, hybrid_embedding = %s
                   WHERE track_id = %s""",
                values,
            )
        updated += len(values)
        print(f"stored {updated}/{len(songs)} song vectors", flush=True)
    return updated, fallback


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
    allow_sparse: bool = False,
    dry_run: bool = False,
    model_out: str | None = None,
) -> dict:
    gate = density_gate_status(colisten.graph_stats())
    if not allow_sparse and not gate["ready"]:
        raise RuntimeError(
            "co-listening density gate not met: "
            f"nodes={gate['nodes']} (need {COLISTEN_MIN_NODES}), "
            f"avg_degree={gate['avg_degree']} (need {COLISTEN_MIN_AVG_DEGREE})"
        )
    stats = {key: gate[key] for key in ("nodes", "edges", "avg_degree")}

    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("install requirements-jobs.txt before training") from exc

    adjacency = collapse_undirected_edges(_load_edges())
    walks = WeightedWalks(
        adjacency, walk_length=walk_length, walks_per_node=walks_per_node, seed=seed
    )
    model = Word2Vec(
        sentences=walks,
        vector_size=dimension,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    if model_out:
        model.save(model_out)

    params = {
        "algorithm": "weighted_deepwalk_skipgram",
        "dimension": dimension,
        "walk_length": walk_length,
        "walks_per_node": walks_per_node,
        "window": window,
        "epochs": epochs,
        "workers": workers,
        "seed": seed,
        "beta": beta,
        "undirected_pair_weight": "max",
    }
    updated = fallback = 0
    if not dry_run:
        updated, fallback = _write_song_vectors(model, beta=beta)
        with get_cursor() as cursor:
            cursor.execute(
                """INSERT INTO model_runs
                   (model, node_count, edge_count, dimension, songs_updated, params)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    "weighted_deepwalk_skipgram",
                    stats["nodes"],
                    stats["edges"],
                    dimension,
                    updated,
                    Json(params),
                ),
            )
    return {**stats, "songs_updated": updated, "tag_only_fallbacks": fallback, "params": params}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 2 co-listening embeddings.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--walk-length", type=int, default=40)
    parser.add_argument("--walks-per-node", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=COLISTEN_BETA)
    parser.add_argument("--allow-sparse", action="store_true")
    parser.add_argument(
        "--check-density",
        action="store_true",
        help="print the current gate status and exit without loading edges or training",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()
    if args.check_density:
        gate = density_gate_status(colisten.graph_stats())
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
        allow_sparse=args.allow_sparse,
        dry_run=args.dry_run,
        model_out=args.model_out,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
