# Local development helpers for the Underground Music Discovery app.
# Typical loop (three terminals):
#   make db        # 1. start Postgres + pgvector (once)
#   make api       # 2. backend at http://localhost:8000  (auto-reload)
#   make web       # 3. frontend at http://localhost:5173 (auto-reload)
#
# Other targets:
#   make db-down   # stop the DB (keeps data)
#   make db-reset  # wipe the DB and recreate it from scratch
#   make test      # run the backend test suite
#   make stress    # load-test a running API (STRESS_URL=https://... to target prod)
#   make install   # install backend + frontend dependencies

VENV ?= .venv
TRAIN_VENV ?= .venv-train
PYTHON_312 ?= python3.12
STRESS_URL ?= http://localhost:8000

.PHONY: db db-down db-reset api web test stress install train-install crawl-colisten train-colisten validate-colisten publish-colisten rollback-colisten run-colisten model-status legacy-embedding-cleanup rebuild-hybrid build-colisten-ground-truth

db:
	docker compose up -d
	@echo "Postgres + pgvector is up on localhost:5432"

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d
	@echo "Database wiped and recreated."

api:
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test:
	$(VENV)/bin/pytest -q

# Load test a running server. Targets local by default; override the URL to hit
# prod (read-only there unless you add --allow-writes), e.g.:
#   make stress STRESS_URL=https://pyo-backend.pedroboudoux.com
stress:
	$(VENV)/bin/python -m tests.stress --base-url $(STRESS_URL) $(STRESS_ARGS)

install:
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt
	cd frontend && npm install

train-install:
	$(PYTHON_312) -m venv $(TRAIN_VENV)
	$(TRAIN_VENV)/bin/pip install -r requirements.txt -r requirements-jobs.txt

crawl-colisten:
	$(VENV)/bin/python -m jobs.crawl_colisten $(COLISTEN_ARGS)

train-colisten:
	$(TRAIN_VENV)/bin/python -m jobs.train_colisten_embeddings train $(COLISTEN_ARGS)

validate-colisten:
	$(TRAIN_VENV)/bin/python -m jobs.train_colisten_embeddings validate $(MODEL_RUN_ID) $(COLISTEN_ARGS)

publish-colisten:
	$(TRAIN_VENV)/bin/python -m jobs.train_colisten_embeddings publish $(MODEL_RUN_ID)

rollback-colisten:
	$(TRAIN_VENV)/bin/python -m jobs.train_colisten_embeddings rollback $(MODEL_RUN_ID)

run-colisten:
	$(TRAIN_VENV)/bin/python -m jobs.train_colisten_embeddings run $(COLISTEN_ARGS)

model-status:
	$(VENV)/bin/python -m jobs.model_status $(MODEL_STATUS_ARGS)

legacy-embedding-cleanup:
	$(VENV)/bin/python -m jobs.remove_legacy_embedding $(LEGACY_CLEANUP_ARGS)

rebuild-hybrid:
	$(VENV)/bin/python -m jobs.rebuild_hybrid_embeddings $(COLISTEN_ARGS)

build-colisten-ground-truth:
	$(TRAIN_VENV)/bin/python -m eval.ground_truth_colisten $(COLISTEN_ARGS)
