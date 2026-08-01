# Evaluation

Offline scoring for the recommendation model. Runs use the same database and
ranking pipeline as the API.

## Fixtures

- `ground_truth.json`: Last.fm `getSimilar` targets for Stage A comparisons.
  Do not use this to judge the co-listening model; that would be circular.
- `ground_truth_colisten.json`: independent cross-artist adjacency from public
  Deezer playlists for Stage B and hybrid evaluation.
- `baselines/*.json`: committed results used for comparisons.

## Read-only evaluation

`run_eval` disables Last.fm top-up by default, so it does not embed songs or
write co-listening edges. Verify the target `DATABASE_URL` before pointing it at
production.

```bash
python -m eval.run_eval \
  --env-file .env.prod \
  --model hybrid_prod \
  --ground-truth eval/ground_truth_colisten.json
```

Use `--out path.json` to save a result. `--with-topup` enables database writes
and ground-truth leakage; use it only with a local or throwaway database.

## Rebuild the independent fixture

The committed fixture is reproducible eval data. Regenerate it only as an
intentional fixture update:

```bash
make build-colisten-ground-truth \
  COLISTEN_ARGS="--env-file .env.prod"
```

## Beta sweep

`eval.sweep_beta` rewrites `songs.hybrid_embedding` for every beta. Run it only
against staging/local data or during a deliberate maintenance workflow after
atomic model publication is implemented (GitHub issue #33).

```bash
python -m eval.sweep_beta \
  --env-file path/to/staging.env \
  --allow-db-writes
```

Metrics are recall@k, MRR, intra-list distance, median listeners, and fixture
coverage. A run below the default 95% coverage gate fails.
