"""
api/routes/ingest.py
─────────────────────────────────────────────────────────────────────────────
Ingest endpoints — trigger the preprocessing pipeline via API.

POST /api/v1/ingest
  Body: { "pdf_dir": "data/raw", "limit": null, "retry_failed": false }
  Returns: { "run_id": "...", "status": "started", "total_files": N }
  
  Runs the pipeline in a background thread (non-blocking).

GET /api/v1/ingest/{run_id}
  Returns current status and progress of a pipeline run.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from config import settings

router = APIRouter(tags=["Ingest"])

# In-memory run tracker (for simple status reporting without DB query)
_active_runs: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────
#  Request / Response models
# ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    pdf_dir: str = Field(
        default=str(settings.raw_pdf_dir),
        description="Directory containing PDF files to ingest",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of files to process (null = all)",
        ge=1,
    )
    retry_failed: bool = Field(
        default=False,
        description="If true, retry files that previously failed",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Number of chunks to embed per batch",
        ge=1,
        le=512,
    )


class IngestResponse(BaseModel):
    run_id: str
    status: str
    message: str
    total_files: int
    pdf_dir: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str             # running / completed / failed
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    started_at: Optional[str]
    completed_at: Optional[str]


# ──────────────────────────────────────────────────────────────────
#  Background task wrapper
# ──────────────────────────────────────────────────────────────────

def _run_pipeline_background(
    run_id: str,
    pdf_dir: Path,
    limit: Optional[int],
    retry_failed: bool,
    embedding_batch_size: int,
):
    """
    Wraps the async pipeline in a sync thread.
    FastAPI BackgroundTasks runs in the same event loop — we use a separate
    thread with its own event loop to avoid blocking the API.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.run_pipeline import run_pipeline

    _active_runs[run_id]["status"] = "running"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            run_pipeline(
                pdf_dir=pdf_dir,
                limit=limit,
                retry_failed=retry_failed,
                embedding_batch_size=embedding_batch_size,
            )
        )
        _active_runs[run_id]["status"] = "completed"
        logger.info(f"Pipeline run {run_id} completed successfully.")
    except Exception as e:
        _active_runs[run_id]["status"] = "failed"
        _active_runs[run_id]["error"] = str(e)
        logger.exception(f"Pipeline run {run_id} failed: {e}")
    finally:
        loop.close()


# ──────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger the full data preprocessing pipeline.

    - Discovers all PDFs in `pdf_dir`
    - Skips already-processed files (checkpointing)
    - Runs extraction → cleaning → metadata → chunking → embedding → storage
    - Returns immediately with a `run_id` — use GET /ingest/{run_id} to poll status

    **Note**: Processing ~26K PDFs takes 4–8 hours on CPU.
    """
    pdf_dir = Path(request.pdf_dir)
    if not pdf_dir.exists():
        raise HTTPException(status_code=400, detail=f"PDF directory not found: {pdf_dir}")

    # Count PDFs for response
    from pipeline.extractor import PDFExtractor
    all_pdfs = PDFExtractor.discover_pdfs(pdf_dir)
    if request.limit:
        all_pdfs = all_pdfs[: request.limit]

    run_id = str(uuid.uuid4())
    _active_runs[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "total_files": len(all_pdfs),
        "processed_files": 0,
        "failed_files": 0,
        "skipped_files": 0,
        "started_at": None,
        "completed_at": None,
    }

    # Launch in background thread
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(
            run_id,
            pdf_dir,
            request.limit,
            request.retry_failed,
            request.embedding_batch_size,
        ),
        daemon=True,
    )
    thread.start()

    logger.info(f"Pipeline run {run_id} started: {len(all_pdfs)} files to process")

    return IngestResponse(
        run_id=run_id,
        status="started",
        message=f"Pipeline started. Processing {len(all_pdfs)} PDFs in the background.",
        total_files=len(all_pdfs),
        pdf_dir=str(pdf_dir),
    )


@router.get("/ingest/{run_id}", response_model=RunStatusResponse)
async def get_ingest_status(run_id: str):
    """
    Get the current status of a pipeline run by run_id.

    Combines in-memory status with the PostgreSQL pipeline_runs record.
    """
    # Check in-memory first (fast)
    mem_run = _active_runs.get(run_id)

    # Also check PostgreSQL for persisted info
    try:
        from pipeline.db import get_session, PipelineRunRepository

        async with get_session() as session:
            run_repo = PipelineRunRepository(session)
            db_run = await run_repo.get(run_id)

        if db_run:
            return RunStatusResponse(
                run_id=db_run.run_id,
                status=mem_run["status"] if mem_run else db_run.status,
                total_files=db_run.total_files,
                processed_files=db_run.processed_files,
                failed_files=db_run.failed_files,
                skipped_files=db_run.skipped_files,
                started_at=str(db_run.started_at) if db_run.started_at else None,
                completed_at=str(db_run.completed_at) if db_run.completed_at else None,
            )
    except Exception:
        pass

    # Fall back to in-memory only
    if mem_run:
        return RunStatusResponse(
            run_id=run_id,
            status=mem_run.get("status", "unknown"),
            total_files=mem_run.get("total_files", 0),
            processed_files=mem_run.get("processed_files", 0),
            failed_files=mem_run.get("failed_files", 0),
            skipped_files=mem_run.get("skipped_files", 0),
            started_at=mem_run.get("started_at"),
            completed_at=mem_run.get("completed_at"),
        )

    raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
