.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup data ui api test test-cov eval eval-offline docker docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Done. Add your LLM_API_KEY to .env, then: make data"

data: ## Download the real datasets (UCI + World Bank)
	$(PY) scripts/fetch_real_data.py

data-full: ## Also write the complete 1,067,371-row retail file (~95 MB)
	$(PY) scripts/fetch_real_data.py --full

ui: ## Run the Streamlit app on :8501
	.venv/bin/streamlit run ui/app.py --server.port=8501

api: ## Run the FastAPI service on :8000
	.venv/bin/uvicorn api.main:app --reload --port 8000

test: ## Run the test suite
	$(PY) -m pytest -q -p no:warnings

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=core --cov=api --cov-report=term-missing -q -p no:warnings

eval: ## Run the LLM evaluation harness (needs an API key)
	$(PY) evals/run.py --json evals/results.json

api-smoke: ## End-to-end API check against the real datasets (no API key needed)
	$(PY) scripts/api_smoke.py

eval-offline: ## Run the pandas-vs-DuckDB cross-check (no API key needed)
	$(PY) -m pytest tests/test_evals.py -q -p no:warnings

truth: ## Print the independently computed ground truth
	$(PY) evals/ground_truth.py

docker: ## Build the image
	docker compose build

docker-up: ## Start the UI (:8501) and API (:8000)
	docker compose up -d
	@echo "UI  http://localhost:8501"
	@echo "API http://localhost:8000/docs"

docker-down: ## Stop the containers
	docker compose down

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov evals/results.json
