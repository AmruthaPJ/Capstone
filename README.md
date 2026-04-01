# Judicial AI — Agentic Decision Support System
## Data Preprocessing Pipeline

> **Bias-aware case prioritization and legal document intelligence for ~26,000 Supreme Court of India judgments (1950–2024)**

---

## 🏗️ Architecture

```
PDFs (data/raw/)
      │
      ▼
[1] PDF Extraction     ← PyMuPDF → pdfplumber → pytesseract (cascade)
      │
      ▼
[2] Text Cleaning      ← Unicode normalization, boilerplate removal
      │
      ▼
[3] Metadata Extraction ← RegEx: case number, judges, acts, citations, outcome
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
[4] Chunking (512 tok)            [5] PostgreSQL (metadata + chunks)
      │
      ▼
[6] Embedding (all-MiniLM-L6-v2)
      │
      ▼
[7] FAISS Vector Index
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Docker Desktop running
- ~5GB disk space (models + data)

### 2. Setup (one command)
```bash
cd /path/to/Capstone
cp .env.example .env
make setup
```

This will:
- Install all Python dependencies
- Download the spaCy English model
- Start PostgreSQL via Docker
- Initialize the database schema

### 3. Add Your PDFs
```bash
# Place all ~26,000 PDF files here:
cp /path/to/sc_judgments/*.pdf data/raw/
```

### 4. Run the Pipeline
```bash
# Full run (all PDFs, resumable):
make run-pipeline

# Quick test (first 10 PDFs only):
make test-run

# Retry previously failed files:
make retry-failed
```

### 5. Start the API
```bash
make run-api
# API docs: http://localhost:8000/docs
```

---

## 📁 Project Structure
```
Capstone/
├── data/
│   ├── raw/                    ← Drop PDFs here
│   ├── processed/              ← Cleaned text cache
│   ├── faiss_index/            ← FAISS index files
│   │   ├── index.faiss
│   │   └── id_map.json
│   └── checkpoint.db           ← SQLite checkpoint (resume support)
│
├── pipeline/
│   ├── extractor.py            ← PDF text extraction
│   ├── cleaner.py              ← Text normalization
│   ├── metadata.py             ← Metadata extraction
│   ├── chunker.py              ← Text chunking for RAG
│   ├── embedder.py             ← Embedding generation
│   ├── vector_store.py         ← FAISS management
│   └── db.py                   ← PostgreSQL ORM layer
│
├── api/
│   ├── main.py                 ← FastAPI application
│   └── routes/
│       ├── ingest.py           ← POST /api/v1/ingest
│       └── search.py           ← POST /api/v1/search
│
├── scripts/
│   ├── run_pipeline.py         ← Standalone batch runner
│   ├── init_db.py              ← DB initializer
│   └── init_schema.sql         ← Raw SQL schema
│
├── tests/
│   └── test_pipeline.py        ← Unit tests
│
├── config.py                   ← Centralized settings
├── docker-compose.yml          ← PostgreSQL + pgAdmin
├── requirements.txt
├── Makefile
└── .env.example
```

---

## 🔌 API Reference

### Trigger Ingestion
```http
POST /api/v1/ingest
Content-Type: application/json

{
  "pdf_dir": "data/raw",
  "limit": null,
  "retry_failed": false,
  "embedding_batch_size": 64
}
```

### Poll Run Status
```http
GET /api/v1/ingest/{run_id}
```

### Semantic Search
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "bail application in murder cases",
  "top_k": 10,
  "filters": {
    "year_from": 2010,
    "year_to": 2024,
    "case_type": "Criminal Appeal",
    "outcome": "Allowed"
  }
}
```

### Get Case Metadata
```http
GET /api/v1/cases/{case_id}
```

### System Stats
```http
GET /api/v1/stats
```

---

## ⚙️ Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embedding batch |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `MAX_WORKERS` | `4` | Parallel PDF extraction workers |
| `USE_IVF` | `false` | Use IVF index for >100K vectors |

---

## 🗄️ Database Schema

### `cases` table
Key columns: `case_id`, `case_number`, `case_type`, `year`, `judges` (JSONB),
`acts_cited` (JSONB), `citations` (JSONB), `outcome`, `extraction_confidence`

### `chunks` table
Key columns: `case_id` (FK), `chunk_index`, `text`, `faiss_id`

### `pipeline_runs` table
Audit log: `run_id`, `status`, `processed_files`, `failed_files`

---

## 🔁 Checkpointing

The pipeline uses SQLite (`data/checkpoint.db`) to track per-file status:

| Status | Meaning |
|---|---|
| `done` | Successfully extracted, embedded, and stored |
| `in_progress` | Currently being processed |
| `failed` | Extraction or processing failed |

Re-running `make run-pipeline` automatically **skips** files with `done` status.
Use `make retry-failed` to reprocess failed files.

---

## 🧪 Running Tests

```bash
# All unit tests (no DB required)
make test

# Fast mode (skip integration tests)
make test-fast
```

---

## 📊 Performance Estimates (CPU-only)

| Stage | Speed | Notes |
|---|---|---|
| PDF Extraction (PyMuPDF) | ~50 PDFs/min | 4 workers |
| Text Cleaning | ~500 docs/min | In-memory |
| Metadata Extraction | ~300 docs/min | Regex-only |
| Chunking | ~200 docs/min | ~20 chunks/doc avg |
| Embedding (all-MiniLM) | ~150 chunks/min | CPU, batch=32 |
| FAISS Indexing | ~5,000 vecs/sec | IndexFlatIP |

**Estimated total time for 26K PDFs: 6–10 hours on CPU**

---

## 🛠️ Troubleshooting

**PostgreSQL connection refused:**
```bash
make docker-up
sleep 5 && python scripts/init_db.py
```

**Model download slow (first run):**
The `all-MiniLM-L6-v2` model (~85MB) downloads once to `~/.cache/huggingface/`.

**OCR not working:**
```bash
brew install tesseract       # macOS
sudo apt install tesseract-ocr  # Ubuntu
```

**Resume after crash:**
Simply re-run `make run-pipeline` — the checkpoint DB ensures no duplicates.
