# PledgeCast - PLAN.md sec.8: make setup | ingest | build | train | api | app | test
#
# The virtualenv lives OUTSIDE the repo (../pledgecast_venv). Keeping ~52,000
# site-packages files out of the project folder stops editors and file watchers
# from indexing them; git ignored them either way. Override with VENV=/some/path.
#
# Windows venvs put binaries in Scripts/, POSIX in bin/ - both are probed.

VENV ?= ../pledgecast_venv

PY := $(shell if   [ -x "$(VENV)/Scripts/python.exe" ]; then echo "$(VENV)/Scripts/python.exe"; \
              elif [ -x "$(VENV)/bin/python" ];        then echo "$(VENV)/bin/python"; \
              elif [ -x .venv/Scripts/python.exe ];    then echo .venv/Scripts/python.exe; \
              elif [ -x .venv/bin/python ];            then echo .venv/bin/python; \
              else echo python; fi)

.DEFAULT_GOAL := help
.PHONY: help setup init-db universe ingest build train evaluate score sensitivity api app test lint fmt clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv (outside the repo) and install exact pins (sec.4.2)
	python -m venv "$(VENV)"
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo "Now: cp .env.example .env"

init-db:  ## Create the SQLite schema, WAL mode (sec.6)
	$(PY) scripts/00_init_db.py

universe:  ## Build the ~300 company universe -> companies + data/universe.csv
	$(PY) scripts/01_build_universe.py

ingest:  ## Download XBRL + Reg 31 + prices. Resumable (sec.10)
	$(PY) scripts/02_ingest_all.py

build:  ## Point-in-time panel: 13 features + forward-drawdown label (sec.9.3)
	$(PY) scripts/03_build_panel.py

train:  ## Walk-forward: 3 models x 10 experiments, intervals, + label-shuffle check (sec.9.4)
	$(PY) scripts/04_train_all.py

evaluate:  ## Quintile backtest vs null + SHAP global/local (sec.9.9, sec.11)
	$(PY) scripts/05_evaluate_and_explain.py

score:  ## Score the latest (embargo) quarter through the inference service
	$(PY) scripts/06_score_latest.py

sensitivity:  ## Window + materiality sweeps and the univariate table (Phase 10b)
	$(PY) scripts/07_sensitivity.py

api:  ## Serve the inference API (sec.13)
	$(PY) -m uvicorn pledgecast.api.main:app --app-dir src --reload --port $${API_PORT:-8000}

app:  ## Launch the Streamlit dashboard (sec.14)
	$(PY) -m streamlit run dashboard/app.py

test:  ## Run the test suite (sec.15)
	$(PY) -m pytest

test-critical:  ## Only the three-star tests: leakage + parser + labels
	$(PY) -m pytest -m critical

lint:  ## ruff check
	$(PY) -m ruff check .

fmt:  ## ruff format + autofix
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

clean:  ## Remove caches (leaves data/, models/, the DB alone)
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: init-db universe ingest build train evaluate  ## Full pipeline, top to bottom
