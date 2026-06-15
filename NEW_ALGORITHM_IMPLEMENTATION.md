# Hybrid Embedding Refactor — Implementation Spec

Instructions for coding agents implementing the new recommendation algorithm for the
underground music discovery backend (FastAPI + pgvector, see `ARCHITECTURE.md`).

This document is a sequence of **work orders**. Each phase is independently shippable
and has explicit acceptance criteria. **Do not skip Phase 0, and do not start an
expensive phase before the cheaper one ahead of it has a green or neutral eval result.**

---

## Context an agent needs before starting

- Current embeddings are sparse `vector(300)` tag-count vectors. Each slot = one
  Last.fm tag (`tag_vocab.id`), value = normalized blended count. This is the thing
  being replaced.
- Everything downstream of the stored vector — `ann_search`, MMR re-ranking, reject
  steering, playlists — is representation-agnostic and **must not change** in Phases 0–2
  except where explicitly stated.
- Prod `songs` table is small (~3k rows) but growing fast. Backfills get more expensive
  every day. Prefer doing migrations sooner rather than later.
- The end state is a **hybrid** vector: a dense semantic tag embedding concatenated with
  a co-listening graph embedding. The two halves are produced independently and blended.

### Final target representation

```
final_vector = normalize( concat( tag_vec(384),  beta * colisten_vec(128) ) )
```

- `tag_vec` — Stage A (semantic tags). Per-track, works even with sparse tags.
- `colisten_vec` — Stage B (co-listening graph). Relational, needs a dense graph.
- `beta` — config-tunable blend weight. `beta = 0` reduces to the Stage A model.
- Tracks not yet present in the co-listening graph fall back to tag-only (the
  `colisten_vec` half is zeros). This must degrade gracefully, never error.

---

## Phase 0 — Evaluation harness (DO THIS FIRST)

**Goal:** a one-command script that scores any model version on a held-out set, so every
later change can be measured instead of guessed. Capture the current-model baseline now,
while the table is small.

### Tasks

1. **Build a held-out ground-truth set.** Sample ~200–500 seed tracks from `songs`. For
   each, fetch `track.getSimilar` top results from Last.fm and treat them as the
   "should be recommended" target set. Persist this to a fixture file
   (`eval/ground_truth.json`) so runs are reproducible and don't re-hit the API.
   - NOTE: this Last.fm-derived ground truth is valid for Stage A only (it grades the
     *representation*, which differs from raw getSimilar). Stage B needs a separate,
     non-Last.fm ground truth — see Phase 2, task 15.

2. **Implement metrics** in `eval/metrics.py`:
   - `recall_at_k(recommended, target, k)` — fraction of targets retrieved in top k.
   - `mrr(recommended, target)` — mean reciprocal rank of first hit.
   - `intra_list_distance(recommended)` — mean pairwise cosine distance (diversity).
   - `median_listeners(recommended)` — underground health. A model that lifts recall by
     returning popular tracks is a REGRESSION for this product; this metric catches it.

3. **Write `eval/run_eval.py`** — takes a model identifier, runs every ground-truth seed
   through the **full live rec pipeline** (`GET /recommendations/{id}` or the underlying
   service functions), prints the four numbers as a single table. Target ~50–100 lines.

4. **Record the baseline.** Run against the current sparse-tag model. Commit the output
   to `eval/baselines/sparse_tag_baseline.json`. This is the number every later phase
   must beat or match.

5. *(Optional, high value)* A minimal blind A/B page: two rec lists side by side for a
   given seed, friend picks the better one, choice logged to a file. Human preference is
   the real target; offline metrics are the fast proxy.

### Acceptance criteria
- `python eval/run_eval.py --model current` prints all four metrics and exits 0.
- Baseline JSON is committed.
- Re-running produces identical numbers (ground truth is cached, not re-fetched).

---

## Phase 1 — Semantic tag embeddings (Stage A)

**Goal:** replace the sparse `vector(300)` with a dense `vector(384)` semantic tag
embedding. This is the near-term quality win and works regardless of graph density.

**Gate:** none — start immediately after Phase 0 baseline is captured.

### Schema migration

Write a reversible migration. **Keep the old column** under a temporary name for rollback.

- `songs.embedding` → `vector(384)` (rename old to `embedding_legacy_300` first; do not
  drop until Phase 1 is validated).
- `songs.tags jsonb` — stores the raw blended `{tag: count}` dict. Needed because a dense
  averaged vector cannot be inverted back to discrete tags (see `dominant_tags` below).
- `tag_vocab.embedding vector(384)` — per-tag semantic vector cache.

### Code changes

6. **Tag encoder.** Add `sentence-transformers`, load `all-MiniLM-L6-v2` (384-dim, CPU,
   ~80MB). Implement `embed_tag(tag: str) -> list[float]`, **cached in `tag_vocab.embedding`**.
   - CRITICAL: cache by unique tag string. At high ingest, the same tags recur across
     thousands of songs. Encode each unique tag **once**, then reuse. Do NOT re-encode
     per song — verify the cache is solid before the backfill or it will be pathologically
     slow.

7. **Rewrite `build_tag_vector`** (`services/embeddings.py`): instead of
   `count / max_count → slot at vocab.id`, compute the **weighted average of tag
   embeddings**, weights = blended counts, then L2-normalize. At the same time, write the
   raw `{tag: count}` dict into `songs.tags`.

8. **Fix `dominant_tags`** (`embeddings.dominant_tags`, powers `POST /graph/tags`): it
   currently reads tag weights back out of the embedding slots. That no longer works —
   read from `songs.tags` (jsonb) instead. Aggregate counts across the node set, map to
   tags, sort, take `top_n`. **Test this in isolation** — it is the one genuinely fiddly
   migration.

9. **Backfill.** Batch job: re-embed every existing song through the new pipeline. MiniLM
   does thousands/sec on CPU; with the tag cache warm this is fast at 3k rows. Do this
   while the table is small.

10. **Leave untouched:** `ann_search`, `mmr_rerank`, `apply_steering`, both playlist
    strategies. They consume whatever vector is stored. (pgvector dimension on the ANN
    query must match the new 384 — verify the index/operator class is rebuilt.)

### Acceptance criteria
- `python eval/run_eval.py --model stage_a` runs against the new vectors.
- Result is **compared against the Phase 0 baseline** and the comparison is recorded.
  - Clear win → ship it, proceed to Phase 2 with confidence.
  - Marginal → tags are not the ceiling; Stage B is the priority.
  - Regression → debug the weighted-average / normalization before going further.
- `POST /graph/tags` returns correct dominant tags from the jsonb source.
- `embedding_legacy_300` still present (rollback path intact) until sign-off.

---

## Phase 2 — Co-listening graph + embeddings (Stage B)

**Goal:** add a second embedding source grounded in crowd listening behavior, and blend
it with Stage A. This is where the global maximum lives, and where cold-start largely
dissolves.

**Gate:** Phase 1 eval is green or neutral, AND the co-listening graph has reached density
(see task 13 gate). **Do not train node2vec on a sparse graph — it produces garbage.**

### Task 12 — pull this FORWARD, do it during Phase 1

12. **Persist getSimilar edges (data collection — start ASAP).** Add table
    `colisten_edges (source_track_id text, target_track_id text, weight float, source text)`
    where `source` ∈ {`track_similar`, `artist_similar`}. Everywhere the code currently
    calls `track.getSimilar` / `artist.getSimilar` (seeding, recommendation top-up),
    **also write the results as weighted edges** (Last.fm similarity score → weight).
    Append-only, idempotent, zero new API cost. This starts the graph-growth clock —
    the earlier it runs, the better Stage B is when you reach it.

### Remaining Stage B tasks

11. Add `songs.colisten_embedding vector(128)`. Add a `model_runs` table
    (`id`, `model`, `trained_at`, `node_count`, `edge_count`) to track embedding recompute.

13. **Graph-crawl batch job.** Seed from existing `songs`, BFS outward via getSimilar to a
    bounded depth, store edges in `colisten_edges`. Rate-limit; run offline.
    - **Density gate before training:** do not proceed to node2vec until the graph has
      roughly **20–30k nodes with average degree ≳ 8–10**. Expose these counts so the
      gate is checkable. At current growth this may be a few weeks out.

14. **Train node2vec offline.** Use `node2vec` or `gensim` Word2Vec over random walks on
    `colisten_edges` → 128-dim vector per track. Write vectors to
    `songs.colisten_embedding`. Record the run in `model_runs`. Trivial at this graph size.

15. **Second ground-truth set (required for Stage B eval).** Last.fm getSimilar CANNOT
    grade a co-listening model trained on Last.fm getSimilar — that is circular. Build an
    independent set: ~100 hand-curated "these go together" pairs (you + friends) and/or
    playlist co-occurrence from a public source. 100 good pairs is enough to detect a real
    win. Add as `eval/ground_truth_colisten.json` and a `--ground-truth` flag to the eval.

16. **Wire the blend.** Final ANN vector = `normalize(concat(tag_vec, beta * colisten_vec))`.
    `beta` is a config param. Tracks absent from the graph → `colisten_vec` is zeros
    (graceful fallback, never an error). Verify the ANN index dimension matches the new
    concatenated length (384 + 128 = 512).

17. **Sweep `beta`** in eval over `{0, 0.25, 0.5, 1.0, 2.0}` (`beta=0` == Stage A model).
    Pick the value that peaks recall/MRR **without tanking `median_listeners`**. Record the
    chosen value and the full sweep.

### Acceptance criteria
- `colisten_edges` is populating in prod (task 12 shipped early).
- Density gate counts are queryable and were met before training.
- node2vec run recorded in `model_runs`; every song either has a `colisten_embedding` or
  cleanly falls back to tag-only.
- Stage B eval uses the **independent** ground truth, not Last.fm getSimilar.
- Chosen `beta` and the full sweep are committed.

---

## Phase 3 — Consolidate

**Gate:** Stage B shipped and beating the Phase 0 baseline on the independent ground truth.

18. **Prune redundant cold-start machinery.** The escalating listener-cap recursion
    (`500k → 1M → 2M → 10M`), recursive seed expansion, and cold-start fallbacks in
    `POST /graph/seed` exist to compensate for sparse coverage. With co-listening
    embeddings, measure whether they still earn their complexity — likely the seeding flow
    can be simplified substantially. Remove only what the eval confirms is now redundant.

19. **Schedule periodic retraining.** Graph embeddings drift as the graph grows. Add a
    weekly/biweekly node2vec rerun (cron or a Render scheduled job). Update `model_runs`
    each time. Backfilling `colisten_embedding` gets more expensive as the table grows —
    batch it.

20. **Revisit steering last.** Current reject steering is linear subtraction
    (`seed − α·Σ rejected`, α = 0.3) and is crude — one strong reject can drag the query
    somewhere incoherent. Better embeddings make this behave better for free. Only invest
    in fancier steering if the eval says it is still the weak link.

### Acceptance criteria
- Any machinery removed in task 18 is backed by an eval showing no regression.
- A scheduled retraining job exists and has run successfully at least once.

---

## Ordering summary (what runs when)

| When | Tasks |
|------|-------|
| **Immediately, in parallel** | Phase 0 (eval harness + baseline); task 12 (persist getSimilar edges — start the graph clock) |
| **Near-term** | Phase 1 (semantic tags) — the real near-term win; backfill while the table is small |
| **After Phase 1 green + graph dense** | Rest of Phase 2 (crawl, node2vec, blend, beta sweep) |
| **After Stage B beats baseline** | Phase 3 (prune, schedule retrain, revisit steering) |

## Non-negotiables for any agent working this spec
1. Phase 0 before everything. No metric, no merge.
2. Never advance to a more expensive phase without a green/neutral eval on the cheaper one.
3. Stage B is graded on the **independent** ground truth, never on Last.fm getSimilar.
4. Downstream components (ANN/MMR/steering/playlists) stay untouched except where stated.
5. Keep rollback paths (legacy columns, config flags) until each phase is signed off.
6. Tag encoding is cached per unique tag string, never re-encoded per song.