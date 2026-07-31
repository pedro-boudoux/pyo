"""
Co-listening graph crawl (algorithm 2.0, Phase 2, task 13).

Builds out the `colisten_edges` graph ahead of node2vec training instead of waiting
for it to fill from organic seeding traffic. Starting from the songs we already have,
it does a breadth-first walk via Last.fm `track.getSimilar`: ask "what's similar to
this?", persist those crowd-sourced links as weighted edges, then ask the same of the
newly-discovered tracks, a bounded number of hops deep.

This is an OFFLINE batch job, not a request path:

    python -m jobs.crawl_colisten --max-depth 2 --similar-limit 50 --max-calls 5000
    python -m jobs.crawl_colisten --start-limit 200   # small test run

Properties:
  - Resumable / idempotent. Edge-producing sources are detected in
    colisten_edges; successful zero-result calls are recorded in
    colisten_crawl_state. Both are skipped on later runs, so the crawl advances
    instead of retrying empty tracks forever.
  - Budget-capped. `--max-calls` hard-limits total getSimilar calls (cost/time), and
    `--per-level-cap` bounds the BFS fan-out so a level can't explode.
  - Rate-limited. `--delay` seconds between calls keeps us within Last.fm's free tier.
  - Pure data collection. It does NOT embed songs or touch the `songs` table —
    discovered tracks live only as edge endpoints (colisten_edges has no FK), which is
    exactly what densifies the graph beyond our own catalog.

Check progress against the density gate (~20-30k nodes, avg degree >=8-10) with
`colisten.graph_stats()` (printed at the end of every run).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import time

from dotenv import load_dotenv


def _preload_env_file(argv: list[str]) -> None:
    """Load --env-file before importing app.config-dependent modules."""
    env_file = None
    for idx, arg in enumerate(argv):
        if arg == "--env-file" and idx + 1 < len(argv):
            env_file = argv[idx + 1]
            break
        if arg.startswith("--env-file="):
            env_file = arg.split("=", 1)[1]
            break
    if env_file:
        load_dotenv(env_file, override=True)


_preload_env_file(sys.argv[1:])

from app.db import get_cursor
from app.services import colisten, lastfm
from app.services.embeddings import make_track_id

DEFAULT_MAX_DEPTH = 2
DEFAULT_SIMILAR_LIMIT = 50
DEFAULT_MAX_CALLS = 5000
DEFAULT_PER_LEVEL_CAP = 2000
DEFAULT_DELAY = 0.25  # ~4 req/s, comfortably under Last.fm's free tier
DEFAULT_WORKERS = 1
DEFAULT_BATCH_SIZE = 1000


class _RateLimiter:
    """Shared minimum interval between Last.fm request starts."""

    def __init__(self, delay: float):
        self.interval = max(0.0, float(delay or 0.0))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_at)
            self._next_at = start_at + self.interval
            wait_for = start_at - now
        if wait_for > 0:
            time.sleep(wait_for)


def _load_seed_frontier(start_limit: int | None = None) -> list[dict]:
    """The crawl's depth-0 frontier: useful/warm songs first, then cold rows."""
    sql = """SELECT name, artist FROM songs
             ORDER BY (embedding IS NULL), (listeners IS NULL), id"""
    params = ()
    if start_limit:
        sql += " LIMIT %s"
        params = (start_limit,)
    with get_cursor() as cursor:
        cursor.execute(sql, params)
        return [dict(r) for r in cursor.fetchall()]


def _already_crawled() -> set[str]:
    """Sources with edges plus successful zero-result calls from prior runs."""
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT source_track_id AS track_id FROM colisten_edges
            UNION
            SELECT track_id FROM colisten_crawl_state
        """)
        return {r["track_id"] for r in cursor.fetchall()}


def _tid(artist: str, name: str) -> str | None:
    try:
        return make_track_id(artist, name)
    except Exception:
        return None


def _enqueue_similar(similar, *, visited, next_seen, next_frontier, per_level_cap):
    """Queue newly discovered tracks for the next BFS level."""
    for t in similar:
        ttid = _tid(t.get("artist", ""), t.get("name", ""))
        if (ttid and ttid not in visited and ttid not in next_seen
                and len(next_frontier) < per_level_cap):
            next_frontier.append({"artist": t["artist"], "name": t["name"]})
            next_seen.add(ttid)


def _crawl_one(node: dict, similar_limit: int, limiter: _RateLimiter) -> tuple[dict, list, list, bool]:
    limiter.wait()
    try:
        similar = lastfm.get_similar_tracks(node["artist"], node["name"], limit=similar_limit)
    except Exception:
        return node, [], [], False
    try:
        rows = colisten.edge_rows(node["artist"], node["name"], similar, source="track_similar")
    except Exception:
        rows = []
    return node, similar, rows, True


def crawl(
    *,
    seed: list[dict] | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    similar_limit: int = DEFAULT_SIMILAR_LIMIT,
    max_calls: int = DEFAULT_MAX_CALLS,
    per_level_cap: int = DEFAULT_PER_LEVEL_CAP,
    delay: float = DEFAULT_DELAY,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> dict:
    """BFS over track.getSimilar, persisting edges as it goes. Returns a summary
    dict including the final graph_stats."""
    if seed is None:
        seed = _load_seed_frontier()

    visited = _already_crawled()  # crawled sources, skipped (resume support)

    # Build the depth-0 frontier from the seed, deduped and minus anything already crawled.
    frontier: list[dict] = []
    in_frontier: set[str] = set()
    for s in seed:
        tid = _tid(s["artist"], s["name"])
        if tid and tid not in visited and tid not in in_frontier:
            frontier.append({"artist": s["artist"], "name": s["name"]})
            in_frontier.add(tid)

    calls = 0
    edges_written = 0
    empty_completed = 0
    errors = 0

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log(f"crawl start: depth<={max_depth} similar_limit={similar_limit} "
         f"max_calls={max_calls} workers={workers} delay={delay} "
         f"frontier={len(frontier)} already_crawled={len(visited)}")

    for depth in range(max_depth):
        if not frontier or calls >= max_calls:
            break
        next_frontier: list[dict] = []
        next_seen: set[str] = set()
        remaining_budget = max_calls - calls

        # Select this level's work up front. Mark as visited when scheduled so a
        # duplicate track is not scheduled twice even with concurrent workers.
        level_nodes: list[dict] = []
        for node in frontier:
            if len(level_nodes) >= remaining_budget:
                break
            tid = _tid(node["artist"], node["name"])
            if tid is None or tid in visited:
                continue
            visited.add(tid)
            level_nodes.append(node)

        level_calls = len(level_nodes)
        calls += level_calls

        if workers <= 1:
            pending_empty = []
            for i, node in enumerate(level_nodes, start=1):
                try:
                    similar = lastfm.get_similar_tracks(node["artist"], node["name"], limit=similar_limit)
                except Exception:
                    errors += 1
                    continue
                written = colisten.record_edges(
                    node["artist"], node["name"], similar, source="track_similar"
                )
                edges_written += written
                if not similar:
                    track_id = _tid(node["artist"], node["name"])
                    if track_id:
                        pending_empty.append(track_id)
                    if len(pending_empty) >= batch_size:
                        empty_completed += colisten.record_crawl_states(pending_empty)
                        pending_empty = []
                _enqueue_similar(
                    similar,
                    visited=visited,
                    next_seen=next_seen,
                    next_frontier=next_frontier,
                    per_level_cap=per_level_cap,
                )
                if delay:
                    time.sleep(delay)
                if verbose and i % 50 == 0:
                    _log(f"  depth {depth}: {i}/{level_calls} calls, {len(next_frontier)} discovered, "
                         f"{edges_written} edges, {empty_completed + len(pending_empty)} empty, "
                         f"{errors} errors")
            if pending_empty:
                empty_completed += colisten.record_crawl_states(pending_empty)
        else:
            limiter = _RateLimiter(delay)
            pending_rows = []
            pending_empty = []
            completed = 0
            request_batch = max(50, workers * 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for offset in range(0, len(level_nodes), request_batch):
                    futures = [
                        pool.submit(_crawl_one, node, similar_limit, limiter)
                        for node in level_nodes[offset : offset + request_batch]
                    ]
                    for fut in as_completed(futures):
                        node, similar, rows, succeeded = fut.result()
                        completed += 1
                        if not succeeded:
                            errors += 1
                            continue
                        pending_rows.extend(rows)
                        if not similar:
                            track_id = _tid(node["artist"], node["name"])
                            if track_id:
                                pending_empty.append(track_id)
                        _enqueue_similar(
                            similar,
                            visited=visited,
                            next_seen=next_seen,
                            next_frontier=next_frontier,
                            per_level_cap=per_level_cap,
                        )
                        if len(pending_rows) >= batch_size:
                            edges_written += colisten.record_edge_rows(pending_rows)
                            pending_rows = []
                        if len(pending_empty) >= batch_size:
                            empty_completed += colisten.record_crawl_states(pending_empty)
                            pending_empty = []
                        if verbose and completed % 50 == 0:
                            _log(f"  depth {depth}: {completed}/{level_calls} calls, "
                                 f"{len(next_frontier)} discovered, {edges_written} edges, "
                                 f"{empty_completed + len(pending_empty)} empty, {errors} errors")
            if pending_rows:
                edges_written += colisten.record_edge_rows(pending_rows)
            if pending_empty:
                empty_completed += colisten.record_crawl_states(pending_empty)

        _log(f"depth {depth} done: {level_calls} calls -> {len(next_frontier)} next-level tracks")
        frontier = next_frontier

    stats = colisten.graph_stats()
    summary = {
        "calls": calls,
        "edges_written": edges_written,
        "empty_completed": empty_completed,
        "errors": errors,
        **stats,
    }
    _log(f"DONE calls={calls} edges_written={edges_written} "
         f"empty_completed={empty_completed} errors={errors} "
         f"graph: nodes={stats['nodes']} edges={stats['edges']} avg_degree={stats['avg_degree']}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Crawl the co-listening graph via Last.fm getSimilar.")
    p.add_argument("--env-file", default=None, help="dotenv file to load before app config, e.g. .env.prod")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="BFS hops outward from the seed")
    p.add_argument("--similar-limit", type=int, default=DEFAULT_SIMILAR_LIMIT, help="targets per getSimilar call")
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="hard cap on total getSimilar calls")
    p.add_argument("--per-level-cap", type=int, default=DEFAULT_PER_LEVEL_CAP, help="max tracks carried to the next level")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="seconds between calls (rate limit)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel Last.fm workers; 1 preserves sequential crawl")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="edge rows per batched DB upsert when workers > 1")
    p.add_argument("--start-limit", type=int, default=None, help="only seed from the first N songs (test runs)")
    args = p.parse_args()

    crawl(
        max_depth=args.max_depth,
        similar_limit=args.similar_limit,
        max_calls=args.max_calls,
        per_level_cap=args.per_level_cap,
        delay=args.delay,
        workers=args.workers,
        batch_size=args.batch_size,
        seed=_load_seed_frontier(args.start_limit),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
