# Evaluation harness (algorithm 2.0, Phase 0)

A one-command way to score the recommendation model on a held-out set, so every
later change to the embedding/representation can be **measured instead of guessed**.

Per the spec (`NEW_ALGORITHM_IMPLEMENTATION.md`): capture the current-model
baseline **before** any representation change, then re-run after each phase and
compare against the committed baseline JSON.

## Requirements

Runs in-process against the live pipeline, so it needs the same environment as the
app: `DATABASE_URL` reachable (with the `songs` data) and `LASTFM_API_KEY` set (for
building the ground truth). No extra dependencies beyond the app's.

> **Read-only by default — safe against prod.** `run_eval` disables the Last.fm
> top-up, which is the only write path in the rec pipeline (it embeds new songs and
> records co-listening edges). Disabling it means the eval **never mutates the DB**,
> so it can be pointed at Neon prod safely. It's also *more correct*: a writing
> pipeline grows the DB between runs (numbers drift, breaking reproducibility), and
> the top-up pulls from `track.getSimilar` — our ground-truth source — which would
> leak the answers into the recommendations and inflate recall. Pass `--with-topup`
> to run the full writing pipeline (local/throwaway DBs only).

To run against Neon, set the connection string for the invocation, e.g.:

```bash
DATABASE_URL="postgresql://...neon..." python -m eval.run_eval --model current --out eval/baselines/sparse_tag_baseline.json
```

## 1. Build the ground truth (once, cached)

```bash
python -m eval.ground_truth --sample 300
```

Samples ~300 embedded seeds from `songs` and stores each one's `track.getSimilar`
result as its target set in `eval/ground_truth.json`. Re-running the eval reuses
this file (no re-fetching), so numbers are reproducible. Re-run this command only
when you deliberately want a fresh sample.

> This Last.fm-derived ground truth grades the **representation** and is valid for
> **Stage A only**. Stage B must be graded on an independent, non-Last.fm set
> (`eval/ground_truth_colisten.json`) — grading a co-listening model trained on
> getSimilar against getSimilar is circular. See Phase 2, task 15.

## 2. Run the eval

```bash
# print the four-metric table for the current model
python -m eval.run_eval --model current

# capture the Phase 0 baseline (commit this file)
python -m eval.run_eval --model current --out eval/baselines/sparse_tag_baseline.json
```

## Metrics (`metrics.py`)

| Metric | Meaning |
|---|---|
| `recall@k` | fraction of the getSimilar target set retrieved in the top k |
| `mrr` | mean reciprocal rank of the first hit |
| `intra_list_distance` | mean pairwise cosine distance — diversity |
| `median_listeners` | underground health; a recall win bought with popular tracks shows up here as a regression |

## Reading the result

- **Clear win** over baseline → ship it.
- **Marginal** → the representation isn't the ceiling; prioritize the next stage.
- **Regression** → debug before going further (and check `median_listeners` didn't
  quietly creep up).