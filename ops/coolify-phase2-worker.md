# Coolify Phase 2 training worker

The recurring trainer is a private Coolify application built from
`Dockerfile.training`. It has no FQDN, exposed port, or HTTP health check.

## Resource configuration

- Repository: `pedro-boudoux/pyo`
- Build: Dockerfile, path `Dockerfile.training`
- Runtime start command: `sleep infinity`
- CPU limit: 4
- Memory limit: 4 GiB
- Environment: the same encrypted `DATABASE_URL` and `COLISTEN_BETA` values as
  the production API; never copy their resolved values into source or logs
- Scheduled command:
  `python -m jobs.train_colisten_embeddings run --workers 4`
- Schedule: `0 8 * * 0` (Sunday 08:00 UTC, off-hours in Toronto)
- Overlap guard: PostgreSQL advisory lock `PYO_PH2`

The `run` command holds one lock across candidate training, validation, and
atomic publication. Any failed density, structural, fallback, or independent
evaluation gate exits non-zero and leaves the prior active run unchanged.

## Graph-growth policy

Passive edge harvesting in API `getSimilar` calls remains enabled. Run the
offline crawler only when either graph node/edge growth or usable candidate
coverage stalls for two consecutive weekly observations. Every crawl must set a
finite `--limit`; completion is resumable through `colisten_crawl_state`. Never
run an unlimited crawler inside the API process.

## Initial verification

Before enabling the weekly schedule:

1. Verify the worker's redacted database host matches the production Coolify
   Postgres resource.
2. Run `check-density` inside the worker.
3. Run the scheduled command manually and confirm the new `model_runs` row is
   `active`.
4. Stage a second candidate and validate it with an intentionally impossible
   `--min-recall-at-k 1.1`; confirm non-zero exit and an unchanged active run.
5. Enable the schedule and observe one successful scheduled-task execution.
