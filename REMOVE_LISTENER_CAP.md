# Instruction: remove the 500k underground listener cap

## Background

This repo is the pyo music-discovery backend (FastAPI + Postgres/pgvector).
For a while the app filtered recommendation candidates to tracks with
`listeners < 500_000` via the `MAX_LISTENERS` constant. That ceiling was
already removed from most of the codebase in commit `21dc30b` ("Remove
underground listener cap"), but it was re-introduced in the recommendations
ANN pool by commit `2fcc398` ("Select Phase 2 beta from capped independent
eval"). Your job is to remove it for good, everywhere, so mainstream music
can appear alongside underground picks without a hard listener ceiling.

Niche playlist mode keeps its "underground-first" ordering. That is a
sorting preference, not a ceiling, and it stays.

## What to change

### 1. `app/config.py`

Delete the constant:

```python
MAX_LISTENERS = 500000
```

Do not rename it. Delete it.

### 2. `app/routers/recommendations.py`

- Remove `MAX_LISTENERS,` from the `from app.config import (...)` block.
- In `build_recommendations`, change this call:

```python
pool = embeddings.ann_search(
    steered_embedding,
    listeners_cap=MAX_LISTENERS,
    exclude_ids=exclude_ids,
    limit=k * MMR_POOL_MULTIPLIER,
)
```

to:

```python
pool = embeddings.ann_search(
    steered_embedding,
    exclude_ids=exclude_ids,
    limit=k * MMR_POOL_MULTIPLIER,
)
```

- The comment above the cold-seed embed ("being too popular — the
  underground cap only applies to candidates") is now stale. Update it to
  say the seed embedding is unbounded because the seed itself must never be
  filtered out.

### 3. `tests/test_recommendations.py`

Delete the test `test_local_ann_pool_enforces_underground_listener_ceiling`
in `TestBuildRecommendations`. Replace it with a test named
`test_local_ann_pool_has_no_listener_cap` that asserts
`listeners_cap` is **not** passed to `ann_search` (i.e.
`ann_search.call_args.kwargs` does not contain `"listeners_cap"`). Keep the
rest of the test setup identical, but drop the `from app.config import
MAX_LISTENERS` import.

### 4. `app/routers/playlists.py`

- Remove `MAX_LISTENERS` from the import line.
- Change:

```python
NICHE_THRESHOLDS = [100, 1_000, 10_000, 100_000, MAX_LISTENERS]
```

to:

```python
NICHE_THRESHOLDS = [100, 1_000, 10_000, 100_000, float("inf")]
```

`ann_search` treats `listeners_cap=float("inf")` as "no cap" (see
`app/services/embeddings.py`, `use_cap = listeners_cap < float("inf")`), so
niche mode still fills its quota underground-first and then takes whatever
remains, with no 500k ceiling.

### 5. Stale comments (small edits)

- `app/services/lastfm.py`: the `get_track_info` docstring says the listener
  count is "our underground filter" and "we want less than 500k listens to
  keep it niche or whatever". Rewrite it: listener counts are collected and
  surfaced, but no longer used to filter tracks.
- `app/routers/graph.py`: the comment near the seed embed mentions "the
  underground cap only applies to its candidates". Graph seeding has had no
  cap since `21dc30b`; remove that clause.

## What NOT to change

- `app/services/embeddings.py`: keep the `listeners_cap` parameter on
  `ann_search`. Niche playlist mode still uses it.
- `eval/baselines/stage_a_deezer_fixture.json` and
  `eval/baselines/stage_b_beta_sweep.json`: their `"listener_cap": 500000`
  fields are historical metadata about past eval runs. Leave them.
- `eval/metrics.py` and `eval/run_eval.py`: `median_listeners` is an eval
  health metric, not a filter. Leave it.
- Frontend copy under `frontend/src/` that uses the word "underground"
  (toast messages, `SeedingStatus.tsx`): product copy, not a filter. Leave
  it.
- Do not add a new config option to re-enable the cap.

## Documentation

Update these files so they stop describing a 500k ceiling:

- `AGENTS.md`:
  - the "underground ceiling is `listeners < 500_000` (`MAX_LISTENERS`)"
    claim;
  - the recommendation step that says candidates are pulled with
    `listeners < 500k`;
  - the seed-time listener-cap escalation (`500k → 1M → 2M → 10M`); note
    the current code does not do this, so remove the claim entirely rather
    than rewording it;
  - the niche playlist description `100 → 1k → 10k → 100k → 500k`;
  - the `/recommendations` response claim `listeners < 500k`;
  - the `config.py` listing and the `MAX_LISTENERS = 500000` line.
- `ARCHITECTURE.md`: the `ann_search ... listeners < 500k` diagram labels,
  the seed escalation label, and the niche thresholds label.
- `NEW_ALGORITHM_IMPLEMENTATION.md` and `PROJECT_PLAN.md`: any listener-cap
  escalation references.
- `README.md`: the sentence "Tracks with fewer than 500,000 listeners get
  in. The rest stay out." will already be handled by a human; leave README.md
  alone unless it still contains a 500k claim after that.

Use the same wording style as the surrounding text when you edit docs. Do not
write a changelog.

## Verify

1. `rg -n "MAX_LISTENERS" .` must return zero hits in `app/`, `tests/`, and
   `jobs/`. Hits in `eval/` baselines and docs should only be the ones you
   deliberately left (eval metadata).
2. Run the test suite: `make test` (or `.venv/bin/pytest -q`). All tests must
   pass, including the new no-cap assertion.
3. `python -c "import app.main"` must succeed.
4. Read `app/routers/recommendations.py` and `app/routers/playlists.py` once
   more and confirm no `MAX_LISTENERS` remains.
