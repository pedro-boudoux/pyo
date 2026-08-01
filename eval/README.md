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

## Blind human preference sessions

`eval.human_preference` compares two captured recommendation sets without showing
the listener which model produced either side. It supports ties, saves after
every answer, resumes partial ballots, combines multiple listeners, and exports
JSON throughout. Use it to supplement the automated fixtures for beta, language
weighting, cold-start, or steering changes. A human result is never a deployment
gate by itself.

Artifacts under `eval/human_runs/` are gitignored because the private key reveals
model placement and vote metadata may identify evaluators.

### 1. Capture each model

Run the same seed fixture against each exact model state. `--model-version`
should be an immutable code/config identifier; `--run-id` should be the database
`model_runs.id` or an equally unique eval-run identifier. The automated fixture,
result artifact, and status are required metadata.

```bash
.venv/bin/python -m eval.human_preference capture \
  --env-file path/to/control.env \
  --seeds eval/ground_truth_colisten.json \
  --model-id hybrid \
  --model-version git:abc123-beta:1 \
  --run-id model-run:41 \
  --metadata experiment=beta \
  --automated-fixture eval/ground_truth_colisten.json \
  --automated-result eval/baselines/control.json \
  --automated-status passed \
  --out eval/human_runs/control.json

.venv/bin/python -m eval.human_preference capture \
  --env-file path/to/candidate.env \
  --seeds eval/ground_truth_colisten.json \
  --model-id hybrid \
  --model-version git:def456-beta:2 \
  --run-id model-run:42 \
  --metadata experiment=beta \
  --automated-fixture eval/ground_truth_colisten.json \
  --automated-result eval/baselines/candidate.json \
  --automated-status passed \
  --out eval/human_runs/candidate.json
```

Capture is read-only by default: Last.fm top-up is disabled, and a cold seed
fails instead of being embedded. `--with-topup` opts into both behaviors and
must only target local/staging data where writes are intentional.

### 2. Prepare and vote

`prepare` canonicalizes the model order and uses a SHA-256-seeded balanced
randomization. The same captures, study ID, and randomization seed reproduce the
same session ID, pair IDs, lists, and placement key even if `--model-a` and
`--model-b` are swapped. Each capture also records `k`, MMR lambda, read-only
mode, and a digest of the seed set.

```bash
.venv/bin/python -m eval.human_preference prepare \
  --model-a eval/human_runs/control.json \
  --model-b eval/human_runs/candidate.json \
  --study-id beta-1-v-2 \
  --randomization-seed beta-study-2026-08 \
  --session-out eval/human_runs/beta.session.json \
  --key-out eval/human_runs/beta.private-key.json

.venv/bin/python -m eval.human_preference vote \
  --session eval/human_runs/beta.session.json \
  --evaluator-id listener-01 \
  --metadata headphones=wired \
  --metadata room=quiet \
  --out eval/human_runs/listener-01.votes.json
```

Give evaluators only the session file. Keep `beta.private-key.json` away from
them until voting is closed; it contains the model/run/version mapping and the
randomization seed. The blind session and vote files contain no model identity.

### 3. Aggregate, then deblind

Omit `--key` to inspect completion, ties, and per-seed anonymous A/B counts while
the study remains blind. Add the key only after voting closes to attribute wins
to the captured model versions.

```bash
.venv/bin/python -m eval.human_preference aggregate \
  --session eval/human_runs/beta.session.json \
  --votes eval/human_runs/*.votes.json \
  --out eval/human_runs/beta.blind-results.json

.venv/bin/python -m eval.human_preference aggregate \
  --session eval/human_runs/beta.session.json \
  --votes eval/human_runs/*.votes.json \
  --key eval/human_runs/beta.private-key.json \
  --out eval/human_runs/beta.results.json
```

The deblinded result records both model IDs, versions, run IDs, per-model wins,
capture parameters, ties, per-seed counts, ballot session metadata, and the
referenced automated evaluation evidence. It explicitly marks human preference
as supplemental and does not calculate a deployment pass from votes.

## Cold-start ablation (issue #35)

`cold_start_ablation` measures recursive seed expansion and the similar-artist
top-track fallback independently. It reports independent-fixture recall/MRR,
warm and no-similar seed coverage, every Last.fm method call, latency, selected
graph degree/artist breadth, and the uncapped listener policy. The curated cold
fixture is `cold_start_seeds.json`, built from a bounded read-only production
sample and live `track.getSimilar` verification.

Candidate ingestion writes song/tag cache rows. Never point this harness at
production. Restore the same disposable production snapshot before each command;
the comparison rejects mismatched snapshot fingerprints:

```bash
python -m eval.cold_start_ablation run \
  --variant full --env-file /path/to/disposable.env \
  --allow-db-writes --out eval/ablations/full.json
python -m eval.cold_start_ablation run \
  --variant no_expansion --env-file /path/to/disposable.env \
  --allow-db-writes --out eval/ablations/no_expansion.json
python -m eval.cold_start_ablation run \
  --variant no_artist_fallback --env-file /path/to/disposable.env \
  --allow-db-writes --out eval/ablations/no_artist_fallback.json
python -m eval.cold_start_ablation run \
  --variant minimal --env-file /path/to/disposable.env \
  --allow-db-writes --out eval/ablations/minimal.json

python -m eval.cold_start_ablation compare \
  eval/ablations/full.json eval/ablations/no_expansion.json \
  eval/ablations/no_artist_fallback.json eval/ablations/minimal.json \
  --out eval/ablations/comparison.json
```

The default removal gate permits at most a 0.02 absolute recall/MRR loss and a
0.05 absolute no-similar-seed coverage loss. A passing gate is evidence to
review, not an automatic production code deletion.

The committed `ablations/comparison.json` passed both gates on an identical
production snapshot: every variant kept 100% coverage for eight verified
no-similar seeds and six obscure warm controls, with unchanged independent
recall/MRR. Recursive expansion added 794 Last.fm calls across 24 seeds and
roughly 6.7 seconds to mean seed latency. The graph-seeding production defaults
therefore disable both mechanisms. The recommendation exhaustion top-up is a
separate path and retains its similar-artist fallback.
