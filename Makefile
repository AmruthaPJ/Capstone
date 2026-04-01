# Judicial AI Data Pipeline — Makefile
# Provides convenience commands for setup, running, and testing

.PHONY: help install setup docker-up docker-down init-db run-pipeline run-api test clean

# Default Python interpreter
PYTHON := python3
PIP := pip3

# ── Help ─────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Judicial AI Data Preprocessing Pipeline"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make install       Install Python dependencies"
	@echo "  make setup         Full setup: install + spacy model + docker"
	@echo "  make docker-up     Start PostgreSQL (Docker)"
	@echo "  make docker-down   Stop PostgreSQL (Docker)"
	@echo "  make init-db       Initialize database schema"
	@echo "  make run-pipeline  Run the full preprocessing pipeline"
	@echo "  make test-run      Quick test: process first 10 PDFs"
	@echo "  make run-api       Start the FastAPI server"
	@echo "  make test          Run unit tests"
	@echo "  make clean         Remove generated data files"
	@echo ""

# ── Installation ─────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-spacy:
	$(PYTHON) -m spacy download en_core_web_sm

setup: install install-spacy docker-up
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	$(PYTHON) scripts/init_db.py
	@echo "✓ Setup complete. Place PDFs in data/raw/ and run: make run-pipeline"

# ── Docker ───────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo "✓ PostgreSQL started. pgAdmin available at http://localhost:5050"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f postgres

# ── Database ─────────────────────────────────────────────────────
init-db:
	$(PYTHON) scripts/init_db.py

reset-db:
	@echo "⚠️  This will DROP all tables. Press Ctrl+C to cancel (5s)..."
	@sleep 5
	$(PYTHON) scripts/init_db.py --drop-all

# ── Pipeline ─────────────────────────────────────────────────────
run-pipeline:
	$(PYTHON) scripts/run_pipeline.py

# Quick test run — processes only first 10 PDFs
test-run:
	$(PYTHON) scripts/run_pipeline.py --limit 10

# Retry previously failed files
retry-failed:
	$(PYTHON) scripts/run_pipeline.py --retry-failed

# ── API ───────────────────────────────────────────────────────────
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Testing ───────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x -k "not integration"

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	@echo "Removing generated data..."
	rm -rf data/processed/ data/faiss_index/ data/checkpoint.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned."

clean-all: clean docker-down
	docker volume rm $$(docker volume ls -q --filter name=capstone) 2>/dev/null || true
	@echo "✓ Full cleanup done."
