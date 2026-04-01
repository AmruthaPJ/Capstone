# Judicial AI — Agentic Decision Support System
## Data Preprocessing Pipeline

> **Bias-aware case prioritization and legal document intelligence for ~26,000 Supreme Court of India judgments (1950–2024)**

---

## 🏗️ Architecture

```
PDFs (data/raw/)
      │
      ▼
[1] PDF Extraction      ← PyMuPDF → pdfplumber → pytesseract (cascade)
      │
      ▼
[2] Text Cleaning       ← Unicode normalization, boilerplate removal
      │
      ▼
[3] Metadata Extraction ← RegEx: case number, judges, acts, citations, outcome
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
[4] Chunking (512 tok)          [5] PostgreSQL (metadata + chunks)
      │
      ▼
[6] Embedding (all-MiniLM-L6-v2)
      │
      ▼
[7] FAISS Vector Index
```

---

## ⚡ Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Docker Desktop (must be **open and running**)
- ~5GB disk space (models + data)

---

## 🚀 Step-by-Step Setup (First Time Only)

### Step 1 — Navigate to the project folder
```bash
cd /Users/amrutha/Downloads/Capstone
```

### Step 2 — Activate the virtual environment
```bash
source venv/bin/activate
```
Your terminal prompt should now show `(venv)` at the start.

> ⚠️ **Every time you open a new terminal**, you MUST run this before anything else.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> This takes 5–10 minutes on first run (downloads PyTorch ~80MB and other packages).

### Step 4 — Download the spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Open Docker Desktop
Open the **Docker Desktop** app on your Mac. Wait until it says "Docker Desktop is running".

### Step 6 — Start PostgreSQL
```bash
make docker-up
```
Expected output:
```
✔ Container judicial_db    Healthy
✓ PostgreSQL started. pgAdmin available at http://localhost:5050
```

### Step 7 — Initialize the database
```bash
python scripts/init_db.py
```
Expected output:
```
INFO  | Database tables created / verified.
INFO  | ✓ Database initialized successfully.
```

---

## 📂 Add Your PDF Files

Place all Supreme Court judgment PDFs in `data/raw/`:
```bash
mkdir -p data/raw
cp /path/to/your/pdfs/*.pdf data/raw/
```

> ⚠️ Make sure files end in `.pdf` (lowercase). If they are `.PDF`, rename them:
> ```bash
> cd data/raw && for f in *.PDF; do mv "$f" "${f%.PDF}.pdf"; done
> ```

---

## ▶️ Running the Pipeline

### Quick test — process 5 PDFs only (recommended first time)
```bash
python scripts/run_pipeline.py --limit 5
```

### Full run — process all PDFs (resumable if interrupted)
```bash
python scripts/run_pipeline.py
```

### Retry failed files
```bash
python scripts/run_pipeline.py --retry-failed
```

> ⏱️ Full run on 26K PDFs takes **6–10 hours on CPU**. It resumes automatically if stopped — just re-run the same command.

Expected output after a successful run:
```
       Pipeline Run Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                 ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total PDFs             │     5 │
│ Processed              │     5 │
│ Skipped (already done) │     0 │
│ Failed                 │     0 │
│ FAISS vectors          │    77 │
└────────────────────────┴───────┘
✓ Pipeline run complete.
```

---

## 🌐 Starting the API Server

Open a **new terminal tab** (keep the pipeline terminal open), then:

```bash
cd /Users/amrutha/Downloads/Capstone
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> ⚠️ Use `python -m uvicorn` (NOT just `uvicorn`) to ensure it uses the venv Python.

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Open in browser: **http://localhost:8000/docs**

> ⚠️ If you ran the pipeline AFTER starting the API, restart the API so it loads the new FAISS index.

---

## 🔌 API Usage

### Semantic Search
Go to **http://localhost:8000/docs** → `POST /api/v1/search` → Try it out:
```json
{
  "query": "bail application murder case",
  "top_k": 5
}
```

### Trigger Ingestion via API
```json
POST /api/v1/ingest
{
  "pdf_dir": "data/raw",
  "limit": null,
  "retry_failed": false,
  "embedding_batch_size": 64
}
```

### System Stats
```
GET /api/v1/stats
```

---

## 🗄️ Viewing Data in pgAdmin

Open **http://localhost:5050**

Login:
- Email: `admin@judicial.ai`
- Password: `admin`

### Connect to the database:
1. Right-click **Servers** → **Register** → **Server...**
2. **General tab** → Name: `judicial_db`
3. **Connection tab**:
   - Host: `judicial_db`  ← use container name, NOT localhost
   - Port: `5432`
   - Database: `judicial_db`
   - Username: `judicial`
   - Password: `judicial`
4. Click **Save**

Browse tables: `judicial_db → Schemas → public → Tables`
- `cases` — extracted metadata for each judgment
- `chunks` — text chunks per case
- `pipeline_runs` — history of pipeline runs

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
└── .env
```

---

## 🔁 Every Time You Return to This Project

When you open a new terminal session, always run these first:

```bash
cd /Users/amrutha/Downloads/Capstone

# 1. Activate venv
source venv/bin/activate

# 2. Make sure Docker Desktop is open, then start PostgreSQL
make docker-up

# 3. Start the API (in a separate terminal tab)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `externally-managed-environment` error | Run `source venv/bin/activate` first |
| `ModuleNotFoundError` | Use `python -m uvicorn` instead of `uvicorn` |
| Port 5432 already in use | Run `make docker-down` then `make docker-up` |
| "Search index is empty" | Restart the API after running the pipeline |
| pgAdmin "connection refused" | Use `judicial_db` as host, NOT `localhost` |
| Pipeline finds 0 PDFs | Check file extensions — must be `.pdf` not `.PDF` |
| Docker not starting | Open Docker Desktop app first |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

---

## 📊 Performance Estimates (CPU-only)

| Stage | Speed |
|---|---|
| PDF Extraction | ~50 PDFs/min (4 workers) |
| Embedding (all-MiniLM) | ~150 chunks/min |
| FAISS Indexing | ~5,000 vectors/sec |

**Estimated total time for 26K PDFs: 6–10 hours on CPU**
