"""
scripts/run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Standalone batch pipeline runner — no FastAPI required.

Orchestrates the full pipeline:
  1. Discover PDF files in data/raw/
  2. Check SQLite checkpoint — skip already-processed files
  3. Extract text (parallel, PyMuPDF → pdfplumber → OCR)
  4. Clean text
  5. Extract metadata
  6. Store case metadata in PostgreSQL
  7. Chunk text
  8. Generate embeddings (batched)
  9. Add to FAISS index
  10. Store chunks in PostgreSQL (with faiss_id)
  11. Mark checkpoint as done
  12. Save FAISS index to disk

Checkpointing:
  SQLite database at data/checkpoint.db tracks per-file status.
  Re-running the script will skip files with status='done'.
  Files with status='failed' are retried automatically.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --pdf-dir /path/to/pdfs
  python scripts/run_pipeline.py --limit 100       # process first 100 PDFs
  python scripts/run_pipeline.py --retry-failed    # only retry failed files
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from config import settings
from pipeline.extractor import PDFExtractor
from pipeline.cleaner import TextCleaner
from pipeline.metadata import MetadataExtractor
from pipeline.chunker import TextChunker
from pipeline.embedder import Embedder
from pipeline.vector_store import FAISSVectorStore
from pipeline.db import (
    init_engine, create_tables, get_session,
    CaseRepository, ChunkRepository, PipelineRunRepository,
)

console = Console()


# ──────────────────────────────────────────────────────────────────
#  SQLite checkpoint manager
# ──────────────────────────────────────────────────────────────────

class CheckpointDB:
    """
    Lightweight SQLite-based checkpoint tracker.
    Tracks per-file processing status to support resume.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint (
                    file_path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,       -- 'done' | 'failed' | 'in_progress'
                    case_id TEXT,
                    error TEXT,
                    processed_at TEXT
                )
            """)
            conn.commit()

    def get_status(self, file_path: str) -> str | None:
        """Return 'done', 'failed', 'in_progress', or None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM checkpoint WHERE file_path = ?", (file_path,)
            ).fetchone()
        return row[0] if row else None

    def mark_in_progress(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoint (file_path, status, processed_at)
                   VALUES (?, 'in_progress', ?)""",
                (file_path, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def mark_done(self, file_path: str, case_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoint (file_path, status, case_id, processed_at)
                   VALUES (?, 'done', ?, ?)""",
                (file_path, case_id, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def mark_failed(self, file_path: str, error: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoint (file_path, status, error, processed_at)
                   VALUES (?, 'failed', ?, ?)""",
                (file_path, error[:2000], datetime.utcnow().isoformat()),
            )
            conn.commit()

    def filter_unprocessed(
        self,
        file_paths: list[Path],
        retry_failed: bool = False,
    ) -> tuple[list[Path], int]:
        """
        Filter out files that are already done.

        Returns:
            (unprocessed_paths, skipped_count)
        """
        unprocessed: list[Path] = []
        skipped = 0
        for p in file_paths:
            status = self.get_status(str(p))
            if status == "done":
                skipped += 1
            elif status == "failed" and not retry_failed:
                skipped += 1
            else:
                unprocessed.append(p)
        return unprocessed, skipped

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM checkpoint GROUP BY status"
            ).fetchall()
        return {row[0]: row[1] for row in rows}


# ──────────────────────────────────────────────────────────────────
#  Pipeline processor (per-document)
# ──────────────────────────────────────────────────────────────────

async def process_document(
    raw_doc,
    cleaner: TextCleaner,
    metadata_extractor: MetadataExtractor,
    chunker: TextChunker,
    case_repo: CaseRepository,
    chunk_repo: ChunkRepository,
) -> tuple[str, list[str], list[int]] | None:
    """
    Process a single extracted document through cleaning → metadata → chunking
    and store it in PostgreSQL.

    Returns:
        (case_id, chunk_texts, chunk_indices) for embedding, or None on failure.
    """
    # ── Step 1: Clean text ───────────────────────────────────────
    clean_doc = cleaner.clean(
        raw_doc.raw_text,
        raw_doc.file_path,
        raw_doc.file_hash,
    )
    if clean_doc.is_empty:
        logger.warning(f"Empty document after cleaning: {raw_doc.file_name}")
        return None

    # ── Step 2: Extract metadata ─────────────────────────────────
    meta = metadata_extractor.extract(
        clean_doc.clean_text,
        raw_doc.file_path,
        raw_doc.file_name,
        raw_doc.file_hash,
    )

    # ── Step 3: Store case metadata in PostgreSQL ─────────────────
    case_data = {
        "case_id": meta.case_id,
        "file_path": meta.file_path,
        "file_name": meta.file_name,
        "file_hash": meta.file_hash,
        "case_number": meta.case_number,
        "case_type": meta.case_type,
        "year": meta.year,
        "judgment_date": meta.judgment_date,
        "filing_date": meta.filing_date,
        "petitioner": meta.petitioner,
        "respondent": meta.respondent,
        "bench": meta.bench,
        "judges": meta.judges,
        "acts_cited": meta.acts_cited,
        "citations": meta.citations,
        "outcome": meta.outcome,
        "page_count": raw_doc.page_count,
        "word_count": clean_doc.word_count,
        "extraction_method": raw_doc.extraction_method,
        "extraction_confidence": meta.extraction_confidence,
    }
    await case_repo.upsert(case_data)

    # ── Step 4: Chunk text ────────────────────────────────────────
    chunks = chunker.chunk(meta.case_id, clean_doc.clean_text)
    if not chunks:
        logger.warning(f"No chunks generated for case_id={meta.case_id}")
        return None

    # Store chunks WITHOUT faiss_id yet (will be backfilled after embedding)
    chunk_data = [
        {
            "case_id": c.case_id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "char_count": c.char_count,
            "approx_token_count": c.approx_token_count,
            "total_chunks": c.total_chunks,
            "faiss_id": -1,  # Placeholder — updated after FAISS indexing
        }
        for c in chunks
    ]
    db_chunks = await chunk_repo.bulk_insert(chunk_data)

    return meta.case_id, [c.text for c in chunks], [c.id for c in db_chunks]


# ──────────────────────────────────────────────────────────────────
#  Main async pipeline
# ──────────────────────────────────────────────────────────────────

async def run_pipeline(
    pdf_dir: Path,
    limit: int | None = None,
    retry_failed: bool = False,
    embedding_batch_size: int = 64,
):
    """
    Full pipeline execution with checkpointing.

    Args:
        pdf_dir:              Directory containing PDF files.
        limit:                Maximum number of PDFs to process (for testing).
        retry_failed:         If True, re-process previously failed files.
        embedding_batch_size: Number of chunks to embed at once.
    """
    run_id = str(uuid.uuid4())
    console.rule(f"[bold blue]Judicial AI Pipeline[/bold blue] — Run ID: {run_id}")

    # ── Setup ─────────────────────────────────────────────────────
    settings.create_dirs()
    init_engine(settings.db_url)
    await create_tables()

    checkpoint = CheckpointDB(settings.checkpoint_db)
    extractor = PDFExtractor(max_workers=settings.max_workers)
    cleaner = TextCleaner()
    metadata_extractor = MetadataExtractor()
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    # Load or create FAISS index
    vector_store = FAISSVectorStore.load(
        index_dir=settings.faiss_index_dir,
        dim=settings.embedding_dim,
        use_ivf=settings.use_ivf,
        nlist=settings.faiss_nlist,
    )

    # ── Discover PDFs ─────────────────────────────────────────────
    all_pdfs = extractor.discover_pdfs(pdf_dir)
    if limit:
        all_pdfs = all_pdfs[:limit]

    to_process, skipped = checkpoint.filter_unprocessed(all_pdfs, retry_failed)

    console.print(f"[green]Total PDFs:[/green] {len(all_pdfs)}")
    console.print(f"[yellow]Already done (skipped):[/yellow] {skipped}")
    console.print(f"[red]To process:[/red] {len(to_process)}")

    if not to_process:
        console.print("[bold green]✓ All files already processed. Nothing to do.[/bold green]")
        return

    # ── Create pipeline run record ────────────────────────────────
    async with get_session() as session:
        run_repo = PipelineRunRepository(session)
        await run_repo.create(run_id=run_id, total_files=len(to_process))
        await session.commit()

    # ── Accumulators for batch embedding ─────────────────────────
    # We collect chunks from multiple documents, then embed them in large batches
    pending_chunk_texts: list[str] = []
    pending_chunk_db_ids: list[int] = []
    pending_case_ids: list[str] = []

    stats = {"processed": 0, "failed": 0, "skipped": skipped}

    async def flush_embeddings():
        """Embed and index all accumulated chunks, then clear accumulators."""
        nonlocal pending_chunk_texts, pending_chunk_db_ids, pending_case_ids
        if not pending_chunk_texts:
            return

        logger.info(f"Embedding batch of {len(pending_chunk_texts)} chunks...")

        # Generate embeddings
        vectors = embedder.embed(pending_chunk_texts).vectors

        # Add to FAISS
        faiss_ids = vector_store.add(
            vectors=vectors,
            case_ids=pending_case_ids,
            chunk_indices=list(range(len(pending_case_ids))),
        )

        # Backfill faiss_id in PostgreSQL
        from sqlalchemy import update as sa_update
        from pipeline.db import Chunk

        async with get_session() as session:
            for db_id, faiss_id in zip(pending_chunk_db_ids, faiss_ids):
                await session.execute(
                    sa_update(Chunk)
                    .where(Chunk.id == db_id)
                    .values(faiss_id=faiss_id)
                )
            await session.commit()

        # Save FAISS index after each batch
        vector_store.save()

        # Clear accumulators
        pending_chunk_texts.clear()
        pending_chunk_db_ids.clear()
        pending_case_ids.clear()

    # ── Main processing loop ──────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing PDFs...", total=len(to_process))

        for raw_doc in extractor.extract_many(to_process, show_progress=False):
            file_path = raw_doc.file_path
            progress.advance(task)

            if not raw_doc.success:
                checkpoint.mark_failed(file_path, raw_doc.extraction_error or "unknown")
                stats["failed"] += 1
                continue

            checkpoint.mark_in_progress(file_path)

            try:
                async with get_session() as session:
                    case_repo = CaseRepository(session)
                    chunk_repo = ChunkRepository(session)

                    result = await process_document(
                        raw_doc, cleaner, metadata_extractor, chunker,
                        case_repo, chunk_repo,
                    )
                    await session.commit()

                if result is None:
                    checkpoint.mark_failed(file_path, "No valid content")
                    stats["failed"] += 1
                    continue

                case_id, chunk_texts, chunk_db_ids = result

                # Accumulate for batch embedding
                for text, db_id in zip(chunk_texts, chunk_db_ids):
                    pending_chunk_texts.append(text)
                    pending_chunk_db_ids.append(db_id)
                    pending_case_ids.append(case_id)

                # Flush embeddings when batch is large enough
                if len(pending_chunk_texts) >= embedding_batch_size:
                    await flush_embeddings()

                checkpoint.mark_done(file_path, case_id)
                stats["processed"] += 1

                # Update run progress every 10 files
                if stats["processed"] % 10 == 0:
                    async with get_session() as session:
                        run_repo = PipelineRunRepository(session)
                        await run_repo.update_progress(
                            run_id,
                            stats["processed"],
                            stats["failed"],
                            stats["skipped"],
                        )
                        await session.commit()

            except Exception as e:
                logger.exception(f"Pipeline error for {file_path}: {e}")
                checkpoint.mark_failed(file_path, str(e))
                stats["failed"] += 1

    # ── Final flush ───────────────────────────────────────────────
    await flush_embeddings()

    # ── Finalize run record ───────────────────────────────────────
    async with get_session() as session:
        run_repo = PipelineRunRepository(session)
        await run_repo.update_progress(
            run_id,
            stats["processed"],
            stats["failed"],
            stats["skipped"],
        )
        await run_repo.complete(run_id, status="completed")
        await session.commit()

    # ── Summary ───────────────────────────────────────────────────
    table = Table(title="Pipeline Run Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total PDFs", str(len(all_pdfs)))
    table.add_row("Processed", str(stats["processed"]))
    table.add_row("Skipped (already done)", str(stats["skipped"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("FAISS vectors", str(vector_store.total_vectors))

    console.print(table)
    console.print(f"\n[bold green]✓ Pipeline run complete. Run ID: {run_id}[/bold green]")


# ──────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Judicial AI data preprocessing pipeline"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=settings.raw_pdf_dir,
        help=f"Directory containing PDF files (default: {settings.raw_pdf_dir})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to process (useful for testing)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry files that previously failed",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per batch (default: 64)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            pdf_dir=args.pdf_dir,
            limit=args.limit,
            retry_failed=args.retry_failed,
            embedding_batch_size=args.embedding_batch_size,
        )
    )


if __name__ == "__main__":
    main()
