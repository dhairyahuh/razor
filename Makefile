# Entry points, so that nobody has to reconstruct an invocation from the README.
#
# `make demo` is the one that matters: from a clean clone it installs, runs the full loop on
# the quick profile, and leaves a report, a serving bundle and a populated alert queue behind.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install install-serve test test-fast lint run quick demo serve api-test clean \
        docker docker-demo ui dev demo-payload

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

install: $(BIN)/python  ## Install the offline pipeline and its test dependencies
	$(BIN)/pip install --quiet -r requirements.txt
	$(BIN)/pip install --quiet -e .

install-serve: $(BIN)/python  ## Also install the HTTP surface
	$(BIN)/pip install --quiet -r requirements-serve.txt
	$(BIN)/pip install --quiet -e .

test: install  ## Run the full test suite
	$(BIN)/python -m pytest -q

test-fast: install  ## Run everything except the slow end-to-end tests
	$(BIN)/python -m pytest -q -m "not slow"

api-test: install-serve  ## Run the serving-layer tests only
	$(BIN)/python -m pytest -q tests/test_serving.py

quick: install  ## Small profile: the whole loop in about three minutes
	$(BIN)/python -m redteam run --quick

run: install  ## Full profile: 5,000 customers over 45 days, 25-30 minutes
	$(BIN)/python -m redteam run

demo: install-serve  ## Clean clone to running API with a populated alert queue
	$(BIN)/python -m redteam run --quick --run-name demo
	@echo
	@echo "Report:  artifacts/REPORT.md"
	@echo "Bundle:  artifacts/serving/bundle.pkl"
	@echo "Now run 'make serve' and open http://127.0.0.1:8000/docs"

serve: install-serve  ## Start the API over the last run's bundle
	$(BIN)/python -m redteam serve

ui:  ## Start the console against a local API, without Docker
	@command -v npm >/dev/null || { echo "npm not found. 'docker compose up' needs no Node."; exit 1; }
	cd frontend && npm install --no-audit --no-fund && npm run fonts && npm run dev

dev: install-serve  ## API and console together, for development
	@command -v npm >/dev/null || { echo "npm not found. Use 'docker compose up' instead."; exit 1; }
	cd frontend && npm install --no-audit --no-fund && npm run fonts
	$(BIN)/python -m redteam serve & \
	  cd frontend && npm run dev; \
	  kill %1 2>/dev/null || true

demo-payload: install-serve  ## Bake the committed run into static JSON for offline Demo Mode
	$(BIN)/python scripts/build_demo_payload.py --run default_run

docker:  ## Build the container images
	docker compose build

docker-demo: docker  ## Same as `demo`, in containers
	docker compose run --rm pipeline
	docker compose up -d
	@echo "Console on http://127.0.0.1:5173  ·  API on http://127.0.0.1:8000/docs"

clean:  ## Remove generated artefacts, keeping the committed demo outputs
	rm -rf artifacts/*.parquet artifacts/*.csv artifacts/serving data/*.sqlite* \
	       .pytest_cache **/__pycache__
