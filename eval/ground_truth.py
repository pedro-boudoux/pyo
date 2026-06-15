"""
Build a held-out ground-truth set for offline eval (algorithm 2.0, Phase 0, task 1).

Samples seed tracks from the `songs` table and, for each, treats Last.fm
`track.getSimilar` as the "should be recommended" target set. The result is cached
to eval/ground_truth.json so eval runs are reproducible and don't re-hit Last.fm.

    python -m eval.ground_truth --sample 300

NOTE (from the spec): this Last.fm-derived ground truth grades the *representation*
and is valid for Stage A only. Stage B needs an independent, non-Last.fm set
(eval/ground_truth_colisten.json) — see Phase 2, task 15.
"""
import argparse
import json
import os

from app.db import get_cursor
from app.services import lastfm
from app.services.embeddings import make_track_id

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
SIMILAR_LIMIT = 20


def sample_seeds(n: int) -> list[dict]:
    """Random sample of embedded songs (the rec pipeline needs a usable seed)."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT track_id, name, artist
            FROM songs
            WHERE embedding IS NOT NULL
            ORDER BY random()
            LIMIT %s
            """,
            (n,),
        )
        return [dict(r) for r in cursor.fetchall()]


def build(sample: int, out_path: str = GROUND_TRUTH_PATH) -> dict:
    seeds = sample_seeds(sample)
    print(f"Sampled {len(seeds)} seeds; fetching getSimilar targets from Last.fm...")

    entries = []
    for i, seed in enumerate(seeds, start=1):
        try:
            similar = lastfm.get_similar_tracks(seed["artist"], seed["name"], limit=SIMILAR_LIMIT)
        except Exception as exc:
            print(f"  [{i}/{len(seeds)}] {seed['artist']} — {seed['name']}: skipped ({exc})")
            continue

        targets = sorted({make_track_id(s["artist"], s["name"]) for s in similar})
        if not targets:
            continue  # a seed with no getSimilar can't be graded
        entries.append({
            "seed_track_id": seed["track_id"],
            "name": seed["name"],
            "artist": seed["artist"],
            "targets": targets,
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(seeds)}] collected {len(entries)} usable seeds so far")

    data = {"similar_limit": SIMILAR_LIMIT, "seeds": entries}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(entries)} ground-truth seeds to {out_path}")
    return data


def load(path: str = GROUND_TRUTH_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the eval ground-truth set.")
    parser.add_argument("--sample", type=int, default=300, help="number of seed tracks to sample")
    parser.add_argument("--out", default=GROUND_TRUTH_PATH, help="output JSON path")
    args = parser.parse_args()
    build(args.sample, args.out)