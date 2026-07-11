# Phase 2 Today Plan

Goal: get from the current Phase 1 dense tag model to a working Phase 2 co-listening
hybrid implementation, with measurable gates and no accidental prod writes during eval.

## Current Status

- Phase 1 is live: `songs.embedding` is `vector(384)` dense semantic tag vectors.
- Rollback column still exists: `songs.embedding_legacy_300` is `vector(300)`.
- Prod `.env.prod` exists locally and is ignored by git.
- Eval runner now supports `--env-file .env.prod`.
- Eval runner no longer calls the SlowAPI-decorated route directly; it uses
  `build_recommendations(...)`.
- Eval runner has coverage enforcement and progress output.
- Last known prod schema/data check:
  - `songs_total`: 4681
  - `songs_embedded`: 2361
  - `ground_truth_seed_hits`: 294 / 294
  - `ground_truth_embedded_seed_hits`: 294 / 294
  - `songs.embedding`: `vector(384)`
  - `songs.embedding_legacy_300`: `vector(300)`
  - `colisten_edges`: 1422 nodes, 2415 edges, avg degree 3.4
- Local DB is not valid for the saved baseline eval: it only had 33 / 294 seed IDs.

## Step 1 — Re-run Phase 1 Prod Eval Gate

Purpose: confirm the current Stage A model is still green/neutral before building Stage B.

Status: complete.

Command:

```bash
.venv/bin/python -m eval.run_eval --env-file .env.prod --model stage_a_prod_recheck --k 10 --progress-every 25
```

Prod result captured July 2026:

| Metric | Historical Stage A | Prod recheck | Read |
|---|---:|---:|---|
| coverage | 294 / 294 | 294 / 294 | pass |
| recall@10 | 0.2276 | 0.2133 | slightly lower, still above sparse baseline 0.1823 |
| mrr | 0.4545 | 0.4611 | slightly better than historical Stage A |
| intra_list_distance | 0.0443 | 0.0418 | similar |
| median_listeners | 220,006 | 197,079 | more underground |

Verdict: green/neutral. Phase 1 remains good enough to proceed to Phase 2 collection/schema work.

Pass criteria:

- Coverage is at least 95%, ideally 294 / 294.
- Metrics are green/neutral against `eval/baselines/stage_a_tags.json`.
- Do not use `--with-topup` for this gate.

If this fails because coverage is low, it is probably not using `.env.prod`.

## Step 2 — Measure Co-listening Density

Purpose: decide whether the graph is dense enough for node2vec or needs crawling first.

Command:

```bash
.venv/bin/python - <<'PY'
from dotenv import load_dotenv
load_dotenv(".env.prod", override=True)
from app.services import colisten
print(colisten.graph_stats())
PY
```

Target gate from the spec:

- Roughly 20k-30k nodes.
- Average degree around 8-10+.

Current known prod graph is far below this: 1422 nodes, 2415 edges, avg degree 3.4.

## Step 3 — Grow `colisten_edges`

Purpose: collect enough Last.fm `track.getSimilar` edges for meaningful graph embeddings.

Start with a bounded crawl:

```bash
.venv/bin/python - <<'PY'
from dotenv import load_dotenv
load_dotenv(".env.prod", override=True)
from jobs.crawl_colisten import crawl
print(crawl(max_depth=2, similar_limit=50, max_calls=5000, delay=0.25))
PY
```

Notes:

- This writes only to `colisten_edges`.
- It does not embed songs or write to `songs`.
- It is resumable: already-crawled source track IDs are skipped.
- Re-check `colisten.graph_stats()` after each run.

## Step 4 — Add Phase 2 Schema

Purpose: create storage for trained co-listening vectors and model run metadata.

Add to `init_db()` and `migrations/init.sql`:

- `songs.colisten_embedding vector(128)`
- `model_runs`
  - `id`
  - `model`
  - `trained_at`
  - `node_count`
  - `edge_count`
  - optional metadata like params/json

Also fix stale `migrations/init.sql`, which still describes the old sparse schema.

## Step 5 — Build Node2vec Training Job

Purpose: turn `colisten_edges` into 128-dim per-track graph embeddings.

Expected artifact:

- `jobs/train_colisten_embeddings.py`

Implementation outline:

- Read weighted edges from `colisten_edges`.
- Build an undirected or directed weighted graph; choose deliberately and document it.
- Generate node2vec/random walks.
- Train Word2Vec/node2vec to 128 dims.
- Write vectors to `songs.colisten_embedding` for matching `songs.track_id`.
- Record the run in `model_runs`.

Important:

- Do not train if the graph is still tiny unless this is explicitly treated as a smoke run.
- If doing a smoke run today, label it clearly and do not call it the production Stage B model.

## Step 6 — Add Independent Stage B Eval

Purpose: avoid circular grading. Stage B is trained from Last.fm-style similarity edges,
so it should not be judged only against Last.fm `getSimilar`.

Needed:

- `eval/ground_truth_colisten.json`
- Eval command using `--ground-truth eval/ground_truth_colisten.json`

Acceptable sources:

- Hand-curated pairs.
- Friend/user-labeled pairs.
- Playlist co-occurrence from a non-Last.fm source.

Do not use Last.fm `getSimilar` as the only Stage B ground truth.

## Step 7 — Wire Hybrid Vector, Sweep Beta, Commit Result

Purpose: move from collected graph vectors to the live hybrid model.

Target representation:

```text
normalize(concat(tag_vec(384), beta * colisten_vec(128)))
```

Tasks:

- Add config for `COLISTEN_EMBEDDING_DIM = 128` and `COLISTEN_BETA`.
- Widen live ANN vector storage from 384 to 512 only after migration/backfill plan is ready.
- Preserve graceful fallback: missing `colisten_embedding` means use zeros for the 128-dim half.
- Rebuild HNSW index for `vector(512)`.
- Sweep beta values: `0`, `0.25`, `0.5`, `1.0`, `2.0`.
- Compare recall/MRR while watching `median_listeners`.
- Commit the chosen beta and sweep output.

## Today Stop Conditions

- If Step 1 fails, stop and fix eval/model quality before Phase 2.
- If Step 2 shows the graph is far below density, prioritize Step 3 crawling and do not pretend
  node2vec production training is ready.
- If a smoke implementation is built on a sparse graph, label it as a smoke test only.
- Do not run eval with `--with-topup` against prod.
- Do not commit `.env.prod`.
