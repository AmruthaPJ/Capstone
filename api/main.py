"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Endpoints:
  POST /api/v1/ingest        — Trigger pipeline for a directory of PDFs
  GET  /api/v1/ingest/{run}  — Check run status
  POST /api/v1/search        — Semantic search over indexed judgments
  GET  /api/v1/cases/{id}    — Retrieve case metadata
  GET  /api/v1/stats         — System stats (total cases, chunks, vectors)
  GET  /health               — Health check

Start with:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config import settings
from pipeline.db import init_engine, create_tables
from pipeline.embedder import Embedder
from pipeline.vector_store import FAISSVectorStore

from api.routes.ingest import router as ingest_router
from api.routes.search import router as search_router


# ──────────────────────────────────────────────────────────────────
#  Application state (shared across requests)
# ──────────────────────────────────────────────────────────────────

class AppState:
    """Holds singletons initialized at startup."""
    embedder: Embedder
    vector_store: FAISSVectorStore


app_state = AppState()


# ──────────────────────────────────────────────────────────────────
#  Lifespan handler (startup / shutdown)
# ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Judicial AI API...")

    # Create required directories
    settings.create_dirs()

    # Initialize database
    init_engine(settings.db_url)
    await create_tables()
    logger.info("Database ready.")

    # Load embedding model (lazy — first call triggers download if needed)
    app_state.embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    # Load FAISS index from disk (if it exists)
    app_state.vector_store = FAISSVectorStore.load(
        index_dir=settings.faiss_index_dir,
        dim=settings.embedding_dim,
        use_ivf=settings.use_ivf,
        nlist=settings.faiss_nlist,
    )
    logger.info(f"FAISS index loaded: {app_state.vector_store.total_vectors} vectors")

    logger.info("✓ Judicial AI API ready.")
    yield

    # Shutdown: save FAISS index
    app_state.vector_store.save()
    logger.info("FAISS index saved on shutdown.")


# ──────────────────────────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Judicial AI — Decision Support API",
    description=(
        "Bias-aware case prioritization and legal document intelligence "
        "for Supreme Court of India judgments (1950–2024)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
#  Routers
# ──────────────────────────────────────────────────────────────────

app.include_router(ingest_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


# ──────────────────────────────────────────────────────────────────
#  Health & Stats endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — returns 200 if the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/stats", tags=["System"])
async def get_stats():
    """
    Return system-wide statistics:
    - Total cases in PostgreSQL
    - Total chunks in PostgreSQL
    - Total vectors in FAISS index
    """
    from pipeline.db import get_session, CaseRepository, ChunkRepository

    async with get_session() as session:
        case_repo = CaseRepository(session)
        chunk_repo = ChunkRepository(session)
        total_cases = await case_repo.count()
        total_chunks = await chunk_repo.count()

    return {
        "total_cases": total_cases,
        "total_chunks": total_chunks,
        "faiss_vectors": app_state.vector_store.total_vectors,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
    }


@app.get("/api/v1/cases/{case_id}", tags=["Cases"])
async def get_case(case_id: str):
    """Retrieve full metadata for a single case by case_id."""
    from pipeline.db import get_session, CaseRepository

    async with get_session() as session:
        repo = CaseRepository(session)
        case = await repo.get_by_case_id(case_id)

    if case is None:
        return JSONResponse(status_code=404, content={"detail": f"Case '{case_id}' not found"})

    return {
        "case_id": case.case_id,
        "case_number": case.case_number,
        "case_type": case.case_type,
        "year": case.year,
        "judgment_date": str(case.judgment_date) if case.judgment_date else None,
        "petitioner": case.petitioner,
        "respondent": case.respondent,
        "bench": case.bench,
        "judges": case.judges,
        "outcome": case.outcome,
        "acts_cited": case.acts_cited,
        "citations": case.citations,
        "page_count": case.page_count,
        "word_count": case.word_count,
        "extraction_confidence": case.extraction_confidence,
        "file_name": case.file_name,
    }
