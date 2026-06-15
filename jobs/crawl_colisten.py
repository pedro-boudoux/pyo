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
  - Resumable / idempotent. Tracks already crawled (already a *source* in
    colisten_edges) are skipped, so re-running picks up where it left off, and the
    underlying upsert just refreshes weights.
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
import time

from app.db import get_cursor
from app.services import colisten, lastfm
from app.services.embeddings import make_track_id

DEFAULT_MAX_DEPTH = 2
DEFAULT_SIMILAR_LIMIT = 50
DEFAULT_MAX_CALLS = 5000
DEFAULT_PER_LEVEL_CAP = 2000
DEFAULT_DELAY = 0.25  # ~4 req/s, comfortably under Last.fm's free tier


def _load_seed_frontier(start_limit: int | None = None) -> list[dict]:
    """The crawl's depth-0 frontier: the songs we already have."""
    sql = "SELECT name, artist FROM songs ORDER BY id"
    params = ()
    if start_limit:
        sql += " LIMIT %s"
        params = (start_limit,)
    with get_cursor() as cursor:
        cursor.execute(sql, params)
        return [dict(r) for r in cursor.fetchall()]


def _already_crawled() -> set[str]:
    """track_ids we've already asked getSimilar for (a source in colisten_edges)."""
    with get_cursor() as cursor:
        cursor.execute("SELECT DISTINCT source_track_id FROM colisten_edges")
        return {r["source_track_id"] for r in cursor.fetchall()}


def _tid(artist: str, name: str) -> str | None:
    try:
        return make_track_id(artist, name)
    except Exception:
        return None


def crawl(
    *,
    seed: list[dict] | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    similar_limit: int = DEFAULT_SIMILAR_LIMIT,
    max_calls: int = DEFAULT_MAX_CALLS,
    per_level_cap: int = DEFAULT_PER_LEVEL_CAP,
    delay: float = DEFAULT_DELAY,
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

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log(f"crawl start: depth<={max_depth} similar_limit={similar_limit} "
         f"max_calls={max_calls} frontier={len(frontier)} already_crawled={len(visited)}")

    for depth in range(max_depth):
        if not frontier or calls >= max_calls:
            break
        next_frontier: list[dict] = []
        next_seen: set[str] = set()
        level_calls = 0

        for node in frontier:
            if calls >= max_calls:
                break
            tid = _tid(node["artist"], node["name"])
            if tid is None or tid in visited:
                continue
            visited.add(tid)
            calls += 1
            level_calls += 1
            try:
                similar = lastfm.get_similar_tracks(node["artist"], node["name"], limit=similar_limit)
            except Exception:
                similar = []
            edges_written += colisten.record_edges(
                node["artist"], node["name"], similar, source="track_similar"
            )
            # queue newly-discovered tracks for the next level
            for t in similar:
                ttid = _tid(t.get("artist", ""), t.get("name", ""))
                if (ttid and ttid not in visited and ttid not in next_seen
                        and len(next_frontier) < per_level_cap):
                    next_frontier.append({"artist": t["artist"], "name": t["name"]})
                    next_seen.add(ttid)
            if delay:
                time.sleep(delay)
            if verbose and level_calls % 50 == 0:
                _log(f"  depth {depth}: {level_calls} calls, {len(next_frontier)} discovered, "
                     f"{edges_written} edges, {calls} total calls")

        _log(f"depth {depth} done: {level_calls} calls -> {len(next_frontier)} next-level tracks")
        frontier = next_frontier

    stats = colisten.graph_stats()
    summary = {"calls": calls, "edges_written": edges_written, **stats}
    _log(f"DONE calls={calls} edges_written={edges_written} "
         f"graph: nodes={stats['nodes']} edges={stats['edges']} avg_degree={stats['avg_degree']}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Crawl the co-listening graph via Last.fm getSimilar.")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="BFS hops outward from the seed")
    p.add_argument("--similar-limit", type=int, default=DEFAULT_SIMILAR_LIMIT, help="targets per getSimilar call")
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="hard cap on total getSimilar calls")
    p.add_argument("--per-level-cap", type=int, default=DEFAULT_PER_LEVEL_CAP, help="max tracks carried to the next level")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="seconds between calls (rate limit)")
    p.add_argument("--start-limit", type=int, default=None, help="only seed from the first N songs (test runs)")
    args = p.parse_args()

    crawl(
        max_depth=args.max_depth,
        similar_limit=args.similar_limit,
        max_calls=args.max_calls,
        per_level_cap=args.per_level_cap,
        delay=args.delay,
        seed=_load_seed_frontier(args.start_limit),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
