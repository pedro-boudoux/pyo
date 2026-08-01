"""
Stress / load harness for the Discover API (issue #12).

Unlike the pytest suite (which mocks every seam), this drives a *live* server
over HTTP — local or prod — to surface reliability weak points under
concurrency: latency tails, error rates, Last.fm rate-limit fallout, and DB
connection-pool exhaustion (get_cursor opens a fresh psycopg2 connection per
call, so concurrent load is the thing most likely to hit Neon's connection cap).

It is NOT collected by pytest (filename isn't test_*.py) and needs no new deps —
just httpx, already in requirements-dev.txt.

Prod safety
-----------
POST /graph/seed and POST /feedback mutate persistent graph state. The harness
refuses to fire those at a non-localhost host unless --allow-writes is given, so
you can hammer prod without polluting the real graph. The default scenario for a
remote host is "read-only" (search / recommend / graph / tags), which only warms
caches the way real traffic does.

Usage
-----
  # Local, full end-to-end journey (search -> seed -> recs -> feedback -> playlist)
  python -m tests.stress --concurrency 20 --duration 30 --scenario journey

  # Prod, safe read-only load
  python -m tests.stress --base-url https://pyo-backend.pedroboudoux.com \
      --concurrency 10 --duration 20

  # Prod, single endpoint
  python -m tests.stress --base-url https://pyo-backend.pedroboudoux.com \
      --scenario search --concurrency 25 --duration 15

Exit code is 1 if any endpoint's error rate exceeds --fail-error-rate (default
5%), so it can gate CI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover - dev-only dependency
    sys.exit("httpx is required: pip install -r requirements-dev.txt")


# A spread of queries — mix of one-word, artist, and song+artist forms so the
# search path exercises both DB cache hits and cold Last.fm lookups.
SEARCH_QUERIES = [
    "radiohead", "tin man", "boards of canada", "frank ocean", "aphex twin",
    "mac demarco", "sufjan stevens", "burial archangel", "tyler the creator",
    "clairo", "men i trust", "slowdive", "duster", "alex g", "weyes blood",
    "japanese breakfast", "king krule", "yves tumor", "caroline polachek",
    "the microphones", "grouper", "panchiko", "feeble little horse", "wednesday",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Recorder:
    """Per-label latency + outcome collector. Single-threaded asyncio, so plain
    dict mutation is safe (no lock needed)."""
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    outcomes: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    start: float = field(default_factory=time.perf_counter)

    def record(self, label: str, status: int | None, dt: float, *, bucket: str) -> None:
        self.latencies[label].append(dt * 1000.0)  # ms
        self.outcomes[label][bucket] += 1

    def total_requests(self) -> int:
        return sum(sum(c.values()) for c in self.outcomes.values())

    def total_bad(self) -> int:
        return sum(
            c["5xx"] + c["error"] for c in self.outcomes.values()
        )


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Shared run state
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    seed_ids: list[str] = field(default_factory=list)   # existing graph nodes (for recs)
    graph_edges: list[tuple[str, str]] = field(default_factory=list)
    search_hits: list[str] = field(default_factory=list)  # track_ids discovered via search
    allow_writes: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def a_seed(self) -> str | None:
        pool = self.seed_ids or self.search_hits
        return random.choice(pool) if pool else None


# ---------------------------------------------------------------------------
# Actions — each returns after recording exactly one request
# ---------------------------------------------------------------------------

async def _do(client, rec, label, method, path, **kw):
    """Issue one request, classify the outcome, record latency. Returns the
    Response (or None on transport error) so callers can chain on the body."""
    t0 = time.perf_counter()
    try:
        r = await client.request(method, path, **kw)
        dt = time.perf_counter() - t0
        bucket = (
            "2xx" if r.status_code < 300
            else "4xx" if r.status_code < 500
            else "5xx"
        )
        rec.record(label, r.status_code, dt, bucket=bucket)
        return r
    except Exception as exc:  # timeout, connection reset, DNS, etc.
        dt = time.perf_counter() - t0
        rec.record(f"{label} [{type(exc).__name__}]", None, dt, bucket="error")
        return None


async def act_search(client, ctx, rec):
    r = await _do(client, rec, "GET /songs/search", "GET",
                  "/songs/search", params={"q": random.choice(SEARCH_QUERIES)})
    if r is not None and r.status_code == 200:
        data = r.json()
        if data:
            async with ctx.lock:
                ctx.search_hits.append(data[0]["track_id"])
                if len(ctx.search_hits) > 200:
                    ctx.search_hits = ctx.search_hits[-200:]


async def act_recommend(client, ctx, rec):
    tid = ctx.a_seed()
    if not tid:
        return
    await _do(client, rec, "GET /recommendations/{id}", "GET",
              f"/recommendations/{tid}", params={"k": 10})


async def act_graph(client, ctx, rec):
    await _do(client, rec, "GET /graph", "GET", "/graph")


async def act_status(client, ctx, rec):
    tid = ctx.a_seed()
    if not tid:
        return
    await _do(client, rec, "GET /songs/{id}/status", "GET", f"/songs/{tid}/status")


async def act_tags(client, ctx, rec):
    await _do(client, rec, "POST /graph/tags", "POST", "/graph/tags",
              json={"top_n": 15})


async def act_seed(client, ctx, rec):
    """WRITE: promotes a node + runs the heaviest pipeline. Gated by allow_writes."""
    if not ctx.allow_writes:
        return
    tid = ctx.a_seed()
    if not tid:
        return
    await _do(client, rec, "POST /graph/seed", "POST", "/graph/seed",
              json={"track_id": tid})


async def act_feedback(client, ctx, rec):
    """WRITE: accept/reject. Gated by allow_writes."""
    if not ctx.allow_writes:
        return
    if ctx.graph_edges and random.choice([True, False]):
        source_id, tid = random.choice(ctx.graph_edges)
        body = {"source_track_id": source_id, "track_id": tid, "action": "reject"}
    else:
        tid = ctx.a_seed()
        if not tid:
            return
        body = {"track_id": tid, "action": "accept"}
    await _do(client, rec, "POST /feedback", "POST", "/feedback",
              json=body)


async def act_playlist(client, ctx, rec):
    tid = ctx.a_seed()
    if not tid:
        return
    kind = random.choice(["linear", "tree"])
    await _do(client, rec, f"POST /playlists/{kind}", "POST", f"/playlists/{kind}",
              json={"track_id": tid, "n": 10})


# Weighted single-action mixes per scenario.
SCENARIOS = {
    "read-only": [(act_search, 5), (act_recommend, 4), (act_graph, 2),
                  (act_status, 2), (act_tags, 1)],
    "search":    [(act_search, 1)],
    "recommend": [(act_recommend, 1)],
    "playlist":  [(act_playlist, 1)],
    "seed":      [(act_seed, 1)],
    "write":     [(act_search, 3), (act_seed, 3), (act_recommend, 3),
                  (act_feedback, 2), (act_playlist, 2)],
}


async def journey(client, ctx, rec):
    """One realistic end-to-end user flow. Writes (seed/feedback) self-skip when
    allow_writes is False, degrading to a read-only journey on prod."""
    # discover a fresh track_id
    r = await _do(client, rec, "GET /songs/search", "GET",
                  "/songs/search", params={"q": random.choice(SEARCH_QUERIES)})
    tid = None
    if r is not None and r.status_code == 200:
        data = r.json()
        if data:
            tid = data[0]["track_id"]
    if not tid:
        tid = ctx.a_seed()
    if not tid:
        return

    if ctx.allow_writes:
        await _do(client, rec, "POST /graph/seed", "POST", "/graph/seed",
                  json={"track_id": tid})
    await _do(client, rec, "GET /recommendations/{id}", "GET",
              f"/recommendations/{tid}", params={"k": 10})
    if ctx.allow_writes:
        await _do(client, rec, "POST /feedback", "POST", "/feedback",
                  json={"track_id": tid, "action": "accept"})
    await _do(client, rec, "POST /playlists/linear", "POST", "/playlists/linear",
              json={"track_id": tid, "n": 10})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def worker(client, ctx, rec, deadline, scenario):
    if scenario == "journey":
        while time.perf_counter() < deadline:
            await journey(client, ctx, rec)
        return
    actions, weights = zip(*SCENARIOS[scenario])
    while time.perf_counter() < deadline:
        action = random.choices(actions, weights=weights, k=1)[0]
        await action(client, ctx, rec)


async def bootstrap(client, ctx, rec):
    """Seed the id pool from the existing graph so read-only recs have targets."""
    r = await _do(client, rec, "GET /graph", "GET", "/graph")
    if r is not None and r.status_code == 200:
        graph = r.json()
        nodes = graph.get("nodes", [])
        ctx.seed_ids = [n["track_id"] for n in nodes]
        ctx.graph_edges = [(e["source"], e["target"]) for e in graph.get("edges", [])]


async def run(args) -> int:
    host = urlparse(args.base_url).hostname or ""
    is_local = host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local")

    scenario_has_writes = args.scenario in ("journey", "write", "seed")
    # Graph mutations fire only when the target is local, or explicitly permitted.
    enable_writes = is_local or args.allow_writes

    if scenario_has_writes and not enable_writes:
        if args.scenario == "seed":  # pure-write scenario — nothing to do read-only
            sys.exit(f"Refusing '{args.scenario}' against {host} without --allow-writes "
                     f"(it would do nothing read-only).")
        print(f"⚠  '{args.scenario}' includes graph-mutating calls and {host} is remote.\n"
              f"   Running read-only — seed/feedback self-skip. Pass --allow-writes to mutate "
              f"the real graph.")

    ctx = Ctx(allow_writes=enable_writes)
    rec = Recorder()
    limits = httpx.Limits(max_connections=args.concurrency + 10,
                          max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    print(f"→ target   {args.base_url}  ({'local' if is_local else 'REMOTE'})")
    print(f"→ scenario {args.scenario}   concurrency {args.concurrency}   "
          f"duration {args.duration}s   writes {'ON' if ctx.allow_writes else 'off'}")

    async with httpx.AsyncClient(base_url=args.base_url, limits=limits, timeout=timeout) as client:
        await bootstrap(client, ctx, rec)
        rec.start = time.perf_counter()  # don't count bootstrap in throughput
        deadline = time.perf_counter() + args.duration
        await asyncio.gather(*[
            worker(client, ctx, rec, deadline, args.scenario)
            for _ in range(args.concurrency)
        ])
        elapsed = time.perf_counter() - rec.start

    return report(rec, elapsed, args.fail_error_rate)


def report(rec: Recorder, elapsed: float, fail_rate: float) -> int:
    total = rec.total_requests()
    print("\n" + "=" * 92)
    print(f"{'endpoint':<34}{'n':>6}{'ok%':>7}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  notes")
    print("-" * 92)

    worst_rate = 0.0
    for label in sorted(rec.latencies):
        lat = rec.latencies[label]
        out = rec.outcomes[label]
        n = sum(out.values())
        ok = out["2xx"]
        bad = out["5xx"] + out["error"]
        ok_pct = (ok / n * 100.0) if n else 0.0
        err_rate = (bad / n) if n else 0.0
        worst_rate = max(worst_rate, err_rate)

        notes = []
        if out["4xx"]:
            notes.append(f"{out['4xx']}×4xx")
        if out["5xx"]:
            notes.append(f"{out['5xx']}×5xx")
        if out["error"]:
            notes.append(f"{out['error']}×err")
        flag = ""
        if err_rate > 0.02:
            flag = "  ⚠ errors"
        elif _pct(lat, 95) > 2000:
            flag = "  ⚠ slow"

        print(f"{label:<34}{n:>6}{ok_pct:>6.0f}%"
              f"{_pct(lat,50):>8.0f}{_pct(lat,95):>8.0f}"
              f"{_pct(lat,99):>8.0f}{max(lat) if lat else 0:>8.0f}"
              f"  {' '.join(notes)}{flag}")

    print("-" * 92)
    rps = total / elapsed if elapsed else 0.0
    print(f"{total} requests in {elapsed:.1f}s  =  {rps:.1f} req/s   "
          f"|   bad (5xx+transport): {rec.total_bad()}")
    print("latency columns are milliseconds.")

    if worst_rate > fail_rate:
        print(f"\n✗ FAIL — worst endpoint error rate {worst_rate*100:.1f}% "
              f"exceeds threshold {fail_rate*100:.0f}%")
        return 1
    print(f"\n✓ PASS — worst endpoint error rate {worst_rate*100:.1f}% "
          f"within threshold {fail_rate*100:.0f}%")
    return 0


def main():
    p = argparse.ArgumentParser(description="Stress-test the Discover API (issue #12).")
    p.add_argument("--base-url", default=os.getenv("STRESS_BASE_URL", "http://localhost:8000"))
    p.add_argument("--scenario", default=None,
                   choices=["read-only", "journey", "write", "search", "recommend",
                            "playlist", "seed"],
                   help="default: read-only for remote hosts, journey for local")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--duration", type=float, default=20.0, help="seconds")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request seconds")
    p.add_argument("--allow-writes", action="store_true",
                   help="permit graph-mutating calls (seed/feedback) against a remote host")
    p.add_argument("--fail-error-rate", type=float, default=0.05,
                   help="exit 1 if any endpoint's error rate exceeds this (default 0.05)")
    args = p.parse_args()

    if args.scenario is None:
        host = urlparse(args.base_url).hostname or ""
        is_local = host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local")
        args.scenario = "journey" if is_local else "read-only"

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
