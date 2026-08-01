# Phase 2 model operations

This runbook covers the production Phase 2 trainer, validation, publication,
rollback, freshness monitoring, and bounded graph crawling.

## Production resources and secret boundary

- API: Coolify application `pyo prod`, id `1`, UUID
  `y120lq5jmtgobmx27s562roa`, public endpoint
  `https://pyo-backend.pedroboudoux.com`
- Trainer: private Coolify application `pyo-phase2-trainer`, id `3`, UUID
  `vmm9l652yt6rs2gofpyh32be`; no FQDN, port, or health check
- Database: private Coolify Postgres `pyo production db`, id `1`, UUID
  `pnrtpthxrxn49mgnyv0nn7ik`
- Scheduled task: `weekly-phase2-training`, Sunday 08:00 UTC, two-hour timeout

`DATABASE_URL` and `COLISTEN_BETA` are encrypted Coolify environment variables.
Never print or copy resolved values. Verify the database target inside the worker
by parsing and printing only the URL hostname; it must equal the database UUID
above.

## Freshness and failure status

Run inside the private worker:

```bash
python -m jobs.model_status
python -m jobs.model_status --fail-on-alert
```

The JSON reports the active run and age, graph snapshot, candidate coverage,
fallback count/rate, latest attempt, and last failure. Failure strings are
redacted and truncated. Defaults:

- stale model: active publication older than 192 hours (weekly cadence plus a
  one-day grace period);
- failed-run review window: 168 hours;
- alert when no active model, the model is stale, or the latest attempt failed.

Override with `MODEL_STALE_AFTER_HOURS` and `MODEL_FAILURE_WINDOW_HOURS` in
Coolify only when the operating cadence deliberately changes.

## Commands

All commands run inside `pyo-phase2-trainer`:

```bash
python -m jobs.train_colisten_embeddings check-density
python -m jobs.train_colisten_embeddings train --workers 4
python -m jobs.train_colisten_embeddings validate RUN_ID
python -m jobs.train_colisten_embeddings publish RUN_ID
python -m jobs.train_colisten_embeddings rollback
python -m jobs.train_colisten_embeddings rollback RUN_ID
python -m jobs.train_colisten_embeddings run --workers 4
```

`run` is the scheduled path and holds one PostgreSQL advisory lock across every
phase. `train` writes only candidate rows. `publish` and `rollback` change all
eligible active song vectors and the active run marker in one transaction.

## Logs

In Coolify, open `pyo-phase2-trainer` → Scheduled Tasks →
`weekly-phase2-training` → Executions. On the host, locate recent executions
without exposing environment values:

```bash
ssh pedro-homelab "docker exec coolify-db psql -U coolify -d coolify -P pager=off -c \"select id,status,started_at,finished_at,duration,left(coalesce(message,''),300) as output from scheduled_task_executions where scheduled_task_id=1 order by id desc limit 10;\""
```

Use `docker logs --tail=160 WORKER_CONTAINER` for container startup only. The
scheduled execution record is authoritative for job output and exit status.

## Rollback drill

1. Record `python -m jobs.model_status` output and the active run ID.
2. Run `rollback`; verify the retained predecessor becomes active.
3. Repeat the model-status check and production smoke checklist.
4. Run `rollback NEWER_RUN_ID` to restore the intended current run.
5. Repeat status and smoke checks. Record both run IDs and timestamps in the
   issue or change log without vector values or secrets.

## Bounded crawl

Passive edge harvesting stays enabled. Compare graph nodes, edges, and candidate
coverage week over week. A crawl is warranted only after two consecutive weeks
of stalled graph growth or degraded coverage. Always provide a finite limit:

```bash
python -m jobs.crawl_colisten --limit 1000
```

The crawl resumes through `colisten_crawl_state`. Never put an unlimited crawler
in the API process.

## Production smoke checklist

After publish or rollback:

1. `curl -fsS https://pyo-backend.pedroboudoux.com/health`
2. Search for a real song, then request recommendations for its returned ID.
3. Confirm recommendations return listener counts. There is intentionally no
   hard recommendation listener ceiling; verify niche playlist mode still
   prioritizes its progressive listener thresholds.
4. Generate one linear and one tree playlist without persisting feedback.
5. Run the safe read-only load check:
   `make stress STRESS_URL=https://pyo-backend.pedroboudoux.com`
6. Run `python -m jobs.model_status --fail-on-alert` in the worker.

Any failure pauses further publication. Roll back before debugging candidate
quality when the API or real recommendation smoke checks regress.
