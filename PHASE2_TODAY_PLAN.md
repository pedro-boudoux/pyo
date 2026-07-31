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

## Step 1 — Completed — Re-run Phase 1 Prod Eval Gate

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

## Step 2 — Completed — Measure Co-listening Density

Purpose: decide whether the graph is dense enough for node2vec or needs crawling first.

Status: complete.

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

Prod result captured July 2026:

- `nodes`: 1422
- `edges`: 2415
- `avg_degree`: 3.4
- `track_similar_edges`: 2400
- `artist_similar_edges`: 15
- `crawled_source_nodes`: 147
- `degree_p50`: 1.0
- `degree_p90`: 10.0
- `degree_max`: 43

Verdict: below the node2vec density gate. Do not train a production Stage B model yet.
Proceed to Step 3 and grow `colisten_edges`.

## Step 3 — Grow `colisten_edges`

Purpose: collect enough Last.fm `track.getSimilar` edges for meaningful graph embeddings.

Status: in progress. Smoke crawl and larger crawls succeeded; graph has enough nodes
but still needs more average degree before production node2vec training.

Smoke crawl result captured July 2026:

- Command shape: `crawl(max_depth=1, similar_limit=50, max_calls=100, delay=0.25)`
- `calls`: 100
- `edges_written`: 4391
- `nodes`: 2963
- `edges`: 6806
- `avg_degree`: 4.59

Verdict: crawler works against prod and only writes `colisten_edges`. More crawling is needed.

Post-crawl density captured July 2026:

- `nodes`: 71505
- `edges`: 222044
- `avg_degree`: 6.21
- `track_similar_edges`: 222029
- `artist_similar_edges`: 15
- `crawled_source_nodes`: 4596
- `degree_p50`: 1.0
- `degree_p90`: 10.0
- `degree_max`: 241

Verdict: node count exceeds the 20k-30k gate, but average degree is still below the
8-10+ target. It is reasonable to proceed with Step 4 schema work in parallel, but
do not treat a production node2vec training run as gated until density improves.

Crawler speed-up added during Step 3:

- `jobs/crawl_colisten.py` now accepts `--workers` for parallel Last.fm requests.
- A shared rate limiter still spaces request starts by `--delay`.
- Parallel mode batches `colisten_edges` upserts to reduce Neon connection/write overhead.
- `--env-file .env.prod` is supported directly by the crawler CLI.

Recommended next command:

```bash
.venv/bin/python -m jobs.crawl_colisten --env-file .env.prod --max-depth 2 --similar-limit 50 --max-calls 5000 --per-level-cap 2000 --delay 0.2 --workers 8 --batch-size 2000
```

If an old sequential crawl is still running, it can be stopped with `Ctrl+C`; rerunning
the faster command resumes because already-crawled source IDs are skipped.

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

Status: deployed through Coolify on July 31, 2026. The additive
`colisten_embedding`, `hybrid_embedding`, `model_runs`, and hybrid HNSW index are
present. `RECOMMENDATION_MODEL` is unset in Coolify, so production remains on the
safe `stage_a` default. Health and a real song search passed after deployment.

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

Status: implemented as a density-gated weighted DeepWalk/skip-gram trainer;
production training waits for the graph gate. A Python 3.12 `.venv-train` is
prepared locally, `Dockerfile.training` defines the separate Coolify worker image,
and `--check-density` verifies the gate without loading edges or training.

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

Status: complete. `eval/ground_truth_colisten.json` contains 100 seeds derived
from nearby cross-artist tracks in 31 independently curated public Deezer
playlists (92 unique seed artists). It does not use Last.fm `getSimilar`.
`eval/sweep_beta.py` validates the fixture provenance before changing vectors.

Stage A reference result against this fixture:

- Coverage: 100 / 100
- Recall@10: 0.0403
- MRR: 0.0336
- Intra-list distance: 0.0273
- Median listeners: 374,160

Recorded in `eval/baselines/stage_a_deezer_fixture.json`.

Purpose: avoid circular grading. Stage B is trained from Last.fm-style similarity edges,
so it should not be judged only against Last.fm `getSimilar`.

Artifacts:

- `eval/ground_truth_colisten.json`
- Eval command using `--ground-truth eval/ground_truth_colisten.json`

Acceptable sources:

- Hand-curated pairs.
- Friend/user-labeled pairs.
- Playlist co-occurrence from a non-Last.fm source.

Do not use Last.fm `getSimilar` as the only Stage B ground truth.

Operational correction: the MacBook's `.env.prod` currently points to the old
Neon database, while the live Coolify API uses its private homelab Postgres
service. At verification time the live graph had 102,090 nodes, 373,217 edges,
and average degree 7.31. Production training must inherit `DATABASE_URL` from
Coolify; do not train the Neon graph by accident.

## Step 7 — Wire Hybrid Vector, Sweep Beta, Commit Result

Status: hybrid storage/composition, zero fallback, HNSW index, model switch, rebuild
job, and sweep command are implemented. Choosing beta and enabling production remain
gated on Step 6 and the production training run.

Periodic retraining belongs in a separate Coolify scheduled job/worker built from
this repo with `requirements.txt` plus `requirements-jobs.txt`. It should inherit
`DATABASE_URL` and the chosen `COLISTEN_BETA` from Coolify. Verify the database
target and complete the first manual production training plus independent beta
evaluation before enabling the schedule.

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
