"""
Run the offline eval (algorithm 2.0, Phase 0, task 3).

Runs every ground-truth seed through the live recommendation pipeline, scores the
results with eval/metrics.py, and prints a single four-metric table. Optionally
writes the result to a baseline JSON for cross-run comparison.

    python -m eval.run_eval --model current
    python -m eval.run_eval --model current --out eval/baselines/sparse_tag_baseline.json
    python -m eval.run_eval --env-file .env.prod --model stage_a_prod_recheck

The model is identified only by label (the pipeline always uses whatever vectors
are currently stored). Capture a baseline before a representation change, then
re-run after and diff the committed JSON.

READ-ONLY BY DEFAULT. The eval disables the Last.fm top-up, for three reasons:
  1. Safety — the top-up path WRITES (embeds new songs + records colisten edges).
     Disabling it means the eval never mutates the DB, so it's safe against prod.
  2. Reproducibility — a writing pipeline grows the DB between runs, so numbers
     would drift. Phase 0 requires identical numbers on re-run.
  3. Validity — the top-up pulls from track.getSimilar, which is the very source
     of our ground truth. Including it injects the answers into the recs and
     inflates recall, measuring Last.fm rather than the representation we grade.
Pass --with-topup to run the full pipeline instead (writes to the DB; use only
against a local/throwaway DB).
"""
import argparse
import json
import os
import sys
from collections import Counter
from unittest.mock import patch

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

from app.config import DEFAULT_K, MMR_LAMBDA
from app.db import get_cursor
from app.services.vector_utils import to_float_list
from eval import ground_truth, metrics

# Imported lazily-safe: this is the undecorated recommendation pipeline, not the
# FastAPI route wrapper. We pass k/lambda explicitly because route defaults use
# FastAPI Query() objects, which are not scalar defaults outside HTTP.
from app.routers.recommendations import build_recommendations


def _embeddings_for(track_ids: list[str]) -> dict[str, list]:
    if not track_ids:
        return {}
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT track_id, embedding FROM songs WHERE track_id = ANY(%s) AND embedding IS NOT NULL",
            (track_ids,),
        )
        return {r["track_id"]: to_float_list(r["embedding"]) for r in cursor.fetchall()}


def _scored_loop(seeds, k, progress_every: int = 0):
    records = []
    errors = Counter()
    total = len(seeds)
    for idx, entry in enumerate(seeds, start=1):
        seed_id = entry["seed_track_id"]
        try:
            resp = build_recommendations(
                seed_id,
                k=k,
                lambda_param=MMR_LAMBDA,
                exclude=[],
                include_tags=False,
            )
        except Exception as exc:
            label = type(exc).__name__
            if hasattr(exc, "status_code"):
                label = f"HTTP {exc.status_code}"
            errors[label] += 1
            if progress_every and (idx % progress_every == 0 or idx == total):
                print(f"  scored {len(records)} / {idx} seeds ({idx}/{total} visited)", file=sys.stderr, flush=True)
            continue

        recs = resp.recommendations if hasattr(resp, "recommendations") else resp["recommendations"]
        records.append({
            "target": set(entry["targets"]),
            "rec_ids": [r.track_id for r in recs],
            "listeners": [r.listeners for r in recs],
        })
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"  scored {len(records)} / {idx} seeds ({idx}/{total} visited)", file=sys.stderr, flush=True)

    # Fetch recommendation embeddings once instead of once per seed. Over a remote
    # prod DB this removes hundreds of round trips from the eval.
    all_rec_ids = sorted({tid for record in records for tid in record["rec_ids"]})
    emb_map = _embeddings_for(all_rec_ids)

    per_seed = {"recall": [], "mrr": [], "ild": [], "med_listeners": []}
    for record in records:
        rec_ids = record["rec_ids"]
        vectors = [emb_map.get(tid) for tid in rec_ids]

        per_seed["recall"].append(metrics.recall_at_k(rec_ids, record["target"], k))
        per_seed["mrr"].append(metrics.mrr(rec_ids, record["target"]))
        per_seed["ild"].append(metrics.intra_list_distance(vectors))
        per_seed["med_listeners"].append(metrics.median_listeners(record["listeners"]))

    if errors:
        print(f"  skipped seeds by error: {dict(errors)}", file=sys.stderr, flush=True)
    return per_seed, len(records)


def _result(model, k, scored, total, per_seed, read_only):
    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "model": model,
        "k": k,
        "read_only": read_only,
        "seeds_scored": scored,
        "seeds_total": total,
        "coverage": round(scored / total, 4) if total else 0.0,
        f"recall_at_{k}": avg(per_seed["recall"]),
        "mrr": avg(per_seed["mrr"]),
        "intra_list_distance": avg(per_seed["ild"]),
        # median of the per-seed median listener counts — the typical underground depth
        "median_listeners": avg(per_seed["med_listeners"]),
    }


def evaluate(
    model: str,
    k: int = DEFAULT_K,
    gt_path: str = ground_truth.GROUND_TRUTH_PATH,
    read_only: bool = True,
    progress_every: int = 0,
) -> dict:
    gt = ground_truth.load(gt_path)
    seeds = gt["seeds"]

    if read_only:
        # Neutralize the only write path in the pipeline (the Last.fm top-up that
        # embeds+stores new songs and records colisten edges). This keeps the run
        # non-mutating, reproducible, and free of ground-truth leakage. See module
        # docstring. Patched on the recommendations module so the call inside
        # build_recommendations resolves to the no-op.
        with patch("app.routers.recommendations.topup_from_lastfm", return_value=[]):
            per_seed, scored = _scored_loop(seeds, k, progress_every=progress_every)
    else:
        per_seed, scored = _scored_loop(seeds, k, progress_every=progress_every)

    return _result(model, k, scored, len(seeds), per_seed, read_only)


def _print_table(result: dict) -> None:
    print()
    print(f"  model:           {result['model']}")
    print(f"  mode:            {'read-only (no top-up)' if result.get('read_only', True) else 'full pipeline (writes)'}")
    print(f"  seeds scored:    {result['seeds_scored']} / {result['seeds_total']} ({result['coverage']:.1%})")
    print("  " + "-" * 38)
    k = result["k"]
    print(f"  recall@{k:<10} {result[f'recall_at_{k}']:.4f}")
    print(f"  mrr             {result['mrr']:.4f}")
    print(f"  intra_list_dist {result['intra_list_distance']:.4f}")
    print(f"  median_listeners{result['median_listeners']:>14,.0f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recommendation eval.")
    parser.add_argument("--model", default="current", help="label for this run (e.g. current, stage_a)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--env-file", default=None, help="dotenv file to load before app config, e.g. .env.prod")
    parser.add_argument("--ground-truth", default=ground_truth.GROUND_TRUTH_PATH)
    parser.add_argument("--out", default=None, help="optional path to write the result JSON")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="minimum fraction of ground-truth seeds that must score for the run to pass",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print eval progress every N visited seeds; 0 disables progress output",
    )
    parser.add_argument(
        "--with-topup",
        action="store_true",
        help="run the FULL pipeline incl. the Last.fm top-up — WRITES to the DB and "
             "leaks ground truth into recall; use only against a local/throwaway DB",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"Ground truth not found at {args.ground_truth}. Run: python -m eval.ground_truth --sample 300")
        return 1

    result = evaluate(
        args.model,
        k=args.k,
        gt_path=args.ground_truth,
        read_only=not args.with_topup,
        progress_every=args.progress_every,
    )
    _print_table(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote result to {args.out}")
    if result["coverage"] < args.min_coverage:
        print(
            f"Coverage {result['coverage']:.1%} is below --min-coverage "
            f"{args.min_coverage:.1%}; point DATABASE_URL at the DB used for this ground truth "
            "or lower the threshold for a smoke run."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
