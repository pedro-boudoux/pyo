# Phase 2 Rollout Record

Phase 2 is complete and serving production. This file is the historical rollout
record; remaining engineering work is tracked in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Production state

As of July 31, 2026:

- The Coolify `pyo prod` app serves the hybrid recommendation model with
  `RECOMMENDATION_MODEL=hybrid` and `COLISTEN_BETA=2`.
- `songs.embedding` remains the 384-dimensional semantic-tag vector and instant
  rollback path.
- `songs.colisten_embedding` stores 128-dimensional weighted-random-walk vectors.
- `songs.hybrid_embedding` stores
  `normalize(concat(tag_vector, 2 × colisten_vector))` in 512 dimensions.
- The live co-listening graph had 161,344 nodes, 741,003 edges, and average
  degree 9.19 at the final rollout check.
- 19,067 songs had a hybrid vector at the final production validation check.

The rollback is config-only: set `RECOMMENDATION_MODEL=stage_a` in Coolify and
redeploy. Do not remove `songs.embedding`; it is both the hybrid tag half and the
rollback model.

## Completed rollout gates

### Stage A control

The production Stage A recheck covered all 294 fixture seeds. Its recall@10 was
0.2133 and MRR was 0.4611, which was green/neutral against the saved semantic-tag
baseline and above the earlier sparse model.

### Graph collection and migration

Every Last.fm `getSimilar` response is harvested into append-only
`colisten_edges`. The resumable crawler also records successful empty responses
in `colisten_crawl_state`.

The earlier Neon crawl contained 489,548 edges. It was merged idempotently into
the private Coolify Postgres database, adding 127,720 edges and refreshing
361,828 overlaps. The MacBook crawler was then corrected to reach that database
through the ignored `.env.prod` SSH-tunnel configuration. No production runtime
depends on Neon now.

### Schema and trainer

The additive Phase 2 schema, hybrid HNSW index, density-gated weighted-walk
trainer, separate Python 3.12 training image, and `model_runs` audit table are
implemented. The production training run processed 19,061 embedded songs:

- 18,959 received learned co-listening vectors.
- 102 used the defined tag-only fallback because they were absent from the
  trained graph.

The live count rose after training as additional songs were ingested and hybrid
vectors were composed.

### Independent beta selection

Stage B was graded against cross-artist adjacency from public Deezer playlists,
not Last.fm `getSimilar`, avoiding circular evaluation. All beta runs covered
100/100 fixture seeds and enforced the 500,000-listener ceiling.

| Beta | Recall@10 | MRR | Intra-list distance | Median listeners |
|---:|---:|---:|---:|---:|
| 0 | 0.0167 | 0.0141 | 0.0300 | 119,261 |
| 0.25 | 0.0365 | 0.0449 | 0.0423 | 203,494 |
| 0.5 | 0.0479 | 0.0518 | 0.0651 | 240,616 |
| 1 | 0.0532 | 0.0792 | 0.0951 | 254,648 |
| **2** | **0.0602** | **0.1008** | **0.1207** | **268,945** |

The selected value and machine-readable sweep are committed in
`eval/baselines/stage_b_beta_sweep.json` (commit `2fcc398`).

### Production cutover validation

After the Coolify config-only cutover:

- `/health` passed.
- Five real recommendation seeds returned valid hybrid recommendations.
- The API output matched the direct hybrid recommendation pipeline.
- Returned recommendations respected the listener ceiling.
- Linear and tree playlist generation passed.
- A production-safe stress run completed 674 requests at roughly 28 requests per
  second with no 5xx responses.

## Deliberately unfinished

There is no scheduled Phase 2 trainer yet. The current trainer writes active
song vectors in batches, so scheduling it as-is could expose a mixture of old
and new vectors during a run. Candidate staging, validation, atomic publication,
and rollback must be implemented first. That work and the rest of Phase 3 are in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).
