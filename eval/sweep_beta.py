"""Rebuild and evaluate the hybrid model across the Phase 2 beta grid.

This intentionally requires an explicit write acknowledgement: each beta rebuilds
songs.hybrid_embedding. Run it against staging/local (or a deliberate maintenance
window), never as the default read-only production eval.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv


def _preload(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--env-file" and index + 1 < len(argv):
            load_dotenv(argv[index + 1], override=True)
            break
        if arg.startswith("--env-file="):
            load_dotenv(arg.split("=", 1)[1], override=True)
            break
    os.environ["RECOMMENDATION_MODEL"] = "hybrid"


_preload(sys.argv[1:])

from eval.run_eval import evaluate, _print_table
from eval.ground_truth_colisten import validate_fixture
from jobs.rebuild_hybrid_embeddings import rebuild


DEFAULT_BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Phase 2 co-listening beta values.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--ground-truth", default="eval/ground_truth_colisten.json")
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="eval/baselines/stage_b_beta_sweep.json")
    parser.add_argument(
        "--allow-db-writes",
        action="store_true",
        help="required acknowledgement that the sweep rewrites songs.hybrid_embedding",
    )
    args = parser.parse_args()

    if not args.allow_db_writes:
        parser.error("--allow-db-writes is required; use a staging/local database")
    if not os.path.exists(args.ground_truth):
        parser.error(
            "independent Stage B ground truth is missing; do not substitute Last.fm getSimilar"
        )
    with open(args.ground_truth) as handle:
        fixture = json.load(handle)
    try:
        validate_fixture(fixture)
    except ValueError as exc:
        parser.error(str(exc))

    results = []
    for beta in args.betas:
        print(f"\nrebuilding hybrid vectors for beta={beta}", flush=True)
        rebuild(beta=beta)
        result = evaluate(
            f"stage_b_beta_{beta:g}",
            k=args.k,
            gt_path=args.ground_truth,
            read_only=True,
            progress_every=25,
        )
        result["beta"] = beta
        _print_table(result)
        results.append(result)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"ground_truth": args.ground_truth, "results": results}, handle, indent=2)
    print(f"wrote {args.out}; hybrid_embedding remains built with beta={args.betas[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
