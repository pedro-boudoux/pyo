# Project Implementation Plan

This is the active engineering plan for pyo after the Phase 2 hybrid-model
cutover. The completed rollout is recorded in
[`PHASE2_TODAY_PLAN.md`](PHASE2_TODAY_PLAN.md).

## Current baseline

- Production serves the 512-dimensional hybrid model at `COLISTEN_BETA=2`.
- The independent Stage B sweep and production validation are green.
- The co-listening graph is above its density gate and continues to grow from
  normal `getSimilar` traffic.
- Stage A (`songs.embedding`) remains intact as the hybrid tag half and instant
  rollback path.
- The trainer has run successfully by hand, but no recurring Coolify job is
  configured.
- GitHub issue #21, reducing country/language dominance in recommendations, is
  the only open product issue at the time this plan was written.

## Execution order

| Priority | Workstream | Depends on | Done when |
|---|---|---|---|
| P0 | Safe candidate training and atomic publication | Current trainer | A failed or interrupted run cannot alter the active model |
| P0 | Coolify retraining schedule | Atomic publication | One scheduled run completes and is visible in `model_runs` |
| P1 | Country/language tag attenuation (#21) | Evaluation fixture | Multilingual quality improves without overall regression |
| P1 | Cold-start ablations and pruning | Stable hybrid baseline | Only machinery proven redundant is removed |
| P1 | Parent-scoped rejection from graph removal | Current graph-removal flow | Removing a child changes later expansions from its parent |
| P1 | Reject-steering evaluation | Parent-scoped rejection | Steering has a measured, bounded effect |
| P2 | Blind human preference evaluation | Candidate quality changes | Results are reproducible and exportable |
| P2 | Observability and runbook | Scheduled training | Model freshness and failures are obvious |
| P3 | Legacy sparse-model removal | Hybrid soak + rollback confidence | Only truly unused 300-dim artifacts are removed safely |

P0 is sequential. The language-weighting, cold-start, and rejection-wiring P1
workstreams can proceed in parallel after the production retraining path is safe;
steering evaluation follows the rejection wiring.

## P0 — Safe recurring Phase 2 training

### 1. Stage candidate vectors without touching production

The current trainer updates `songs.colisten_embedding` and
`songs.hybrid_embedding` every batch. Replace that publication path with a
candidate run:

- Add run lifecycle metadata (`running`, `validated`, `active`, `failed`, and
  `superseded`) to `model_runs` or a companion table.
- Store candidate vectors keyed by `model_run_id` and `track_id`; training must
  not mutate active vectors.
- Take a PostgreSQL advisory lock so two trainers cannot overlap.
- Record the edge snapshot/cutoff, beta, dimensions, random seed, trainer params,
  song count, fallback count, start/end times, and failure details.
- Keep at least the current and previous successful candidates so rollback is
  testable rather than theoretical.

Acceptance criteria:

- Killing the trainer during any batch leaves active recommendation rows
  byte-for-byte unchanged.
- Duplicate or overlapping trainer invocations fail cleanly before doing work.
- Every attempted run has an auditable terminal status.

### 2. Validate and publish atomically

Before activation, validate candidate coverage, dimensions, finite values,
normalization, tag-only fallbacks, the density gate, and a bounded independent
evaluation sample. Then copy the complete candidate into the active song columns
inside one database transaction and mark the run active in that same transaction.
PostgreSQL readers should see either the old complete model or the new complete
model, never a batchwise mixture.

Add explicit commands for:

- `train` — build a candidate only.
- `validate` — report gates without publication.
- `publish` — atomically activate an already validated run.
- `rollback` — republish the previous successful run.

Acceptance criteria:

- A deliberately failed validation cannot alter active vectors.
- Recommendations remain available throughout a candidate run.
- Publish changes the active run atomically and rollback restores the prior run.
- Unit/integration tests cover interruption, validation failure, publication,
  and rollback.

### 3. Add the Coolify scheduled job

Create a private Coolify worker/scheduled task from `Dockerfile.training`. It
inherits the production `DATABASE_URL` and `COLISTEN_BETA`; it must have no public
FQDN. Start weekly, off-hours, with a single-run lock and bounded resources. The
job should train, validate, publish, and exit non-zero on any failed gate.

Also define a graph-growth policy:

- Passive edge harvesting remains always on.
- Run a bounded resumable crawl only when edge/node growth or coverage stalls;
  do not make an unlimited crawler part of the API process.

Acceptance criteria:

- One manual end-to-end candidate/publish run succeeds through the worker image.
- One scheduled run succeeds and appears as the active `model_runs` record.
- A simulated failure leaves the previous model active and produces useful logs.

## P1 — Recommendation quality

### 4. Reduce country and language dominance (#21)

Treat language/geography tags as context, not the main sonic signal:

- Build a conservative, reviewed tag taxonomy for languages, countries,
  demonyms, and broad geographic labels. Do not classify genre-bearing terms
  such as `mpb` merely because they are regional.
- Apply a configurable attenuation factor to those tags before the semantic tag
  weighted average. Keep the raw tag JSON unchanged for inspection.
- Sweep at least `1.0` (control), `0.5`, `0.25`, `0.1`, and `0.0`; prefer
  attenuation over total deletion unless evaluation clearly supports deletion.
- Add a multilingual fixture containing same-language/different-genre and
  different-language/same-genre examples.
- Measure global Stage B recall/MRR/diversity and multilingual cross-language
  retrieval together. Do not optimize only the hand-picked examples.

Acceptance criteria:

- Country/language tags no longer overpower clearly stronger genre evidence.
- Cross-language same-genre retrieval improves or stays neutral.
- The existing independent Stage B fixture has no material regression.
- The selected factor and sweep output are committed and documented.

### 5. Ablate cold-start machinery before pruning it

Evaluate these mechanisms independently:

1. Recursive expansion from the seed's top three candidates.
2. Similar-artists' top-tracks fallback.

Add temporary feature flags or an offline harness so every combination can be
measured without risky production edits. Track recommendation coverage,
independent quality, Last.fm call count, seed latency, graph branching, and a
curated set of genuinely obscure/no-similar seeds.

Acceptance criteria:

- Any removal has a committed ablation showing no meaningful quality or
  cold-seed coverage regression.
- Listener-cap enforcement remains correct for returned recommendations.
- The retained seeding flow has regression tests for warm and cold seeds.

### 6. Connect graph removal to parent-scoped rejection

The current “Remove from graph” action only changes frontend state. It does not
call `POST /feedback`, does not persist which parent produced the removed song,
and therefore does not affect later recommendations. Implement rejection as an
explicit product behavior rather than inferring it from a globally rejected
track ID:

- Extend the feedback contract and storage with a nullable/explicit
  `source_track_id` (the parent being expanded). Require it for new `reject`
  actions while retaining a deliberate compatibility path for historical rows.
- When a non-seed node is removed, submit one rejection for each visible incoming
  parent before finalizing the local removal. If persistence fails, show an error
  and keep or restore the node instead of silently losing the learning signal.
- Do not interpret deleting a seed, pruning a disconnected descendant, blocking
  an artist, or restarting the graph as a rejection. Only the song the user
  deliberately removed receives feedback.
- Scope steering queries to the stored `(source_track_id, rejected_track_id)`
  relationship. A rejection from parent A must not steer parent B unless the user
  also rejected that song from parent B.
- Hard-exclude the exact rejected track from later expansions of that parent in
  addition to steering away from its vector neighborhood, so it cannot
  immediately reappear.
- Apply the parent’s rejection state consistently to recommendation, linear, and
  tree expansion modes. Define and test whether re-seeding an existing parent
  should honor the same state rather than leaving `/graph/seed` accidentally
  inconsistent.
- Make duplicate reject submissions idempotent so repeated clicks/retries do not
  multiply the steering force.

Acceptance criteria:

- Removing a child persists a parent-scoped rejection through the production API.
- Re-expanding that parent does not return the exact removed track and uses the
  steered query for every supported expansion mode.
- Expanding an unrelated parent is unchanged.
- Removing a seed, automatic orphan pruning, artist blocking, and restart create
  no false rejection records.
- API and frontend tests cover multiple incoming parents, request failure/retry,
  idempotency, and historical feedback compatibility.

### 7. Re-evaluate reject steering

Keep this after the rejection wiring, embedding, and cold-start work so it is
judged against the stable model:

- Build repeatable rejection scenarios with one and multiple negatives.
- Measure whether rejected songs and their close neighbors move down while the
  remaining list stays coherent and underground.
- Sweep `STEERING_ALPHA` around the current `0.3` control.
- If subtraction is unstable, compare a bounded/averaged negative centroid
  before considering a more complicated online-learning approach.

Acceptance criteria:

- One reject has a predictable local effect rather than moving the query into an
  unrelated region.
- Multiple rejects are bounded and order-independent.
- The chosen method and alpha beat or match the current control on the steering
  fixture.

## P2 — Evaluation and operations

### 8. Add blind human preference testing

Build a small local evaluation page or CLI that shows anonymized A/B
recommendation lists, randomizes sides, records ties, and exports JSON. Use it
for beta changes, language attenuation, cold-start simplification, and steering.
This supplements automated fixtures; it does not replace them.

Acceptance criteria:

- A session can be reproduced from seed/model identifiers.
- Results include model versions and can be aggregated without knowing which
  side was the candidate during voting.

### 9. Add model freshness observability and an operations runbook

- Expose or script a safe summary of the active run, age, graph snapshot,
  candidate coverage, fallback count, and last failure without leaking secrets.
- Make scheduled-job logs easy to find in Coolify and define a stale/failure
  threshold.
- Document train, validate, publish, rollback, density check, and bounded crawl
  procedures, including how to verify the target database.
- Add a production smoke checklist for health, real recommendations, listener
  caps, playlists, and a short safe load test.

Acceptance criteria:

- An operator can answer “what model is live, how old is it, and did the last
  run fail?” without querying raw vectors.
- A rollback drill succeeds from the written procedure.

## P3 — Legacy cleanup after soak

Do this only after the hybrid model has soaked through at least two successful
scheduled runs and the rollback drill has passed.

- Keep `songs.embedding`. It is not legacy: it is the 384-dimensional tag half
  of every hybrid vector and the Stage A rollback model.
- Back up and then remove `songs.embedding_legacy_300` only after confirming no
  supported environment still needs the sparse-model migration.
- Remove sparse-slot migration code and tests that exist solely for the old
  300-dimensional representation.
- Replace or remove `/songs/repack-vocab`; `tag_vocab.id` is no longer a vector
  slot. Preserve an explicit, correctly named maintenance path if full semantic
  re-embedding is still operationally useful.
- Remove stale compatibility documentation only after the schema change ships.

Acceptance criteria:

- A repository-wide reference audit shows no runtime consumer of removed data.
- A tested backup/restore exists before the destructive migration.
- Stage A rollback and hybrid composition still work after cleanup.

## Definition of finished

The project reaches the planned steady state when recurring training is atomic
and observed in production, issue #21 is evaluated and resolved, cold-start and
steering complexity are justified by measurements, operations are documented,
and only the truly obsolete sparse 300-dimensional model has been removed. No
task is complete solely because code exists; its acceptance criteria and the
relevant production or offline gate must pass.
