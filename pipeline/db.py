"""
pipeline/db.py
─────────────────────────────────────────────────────────────────────────────
PostgreSQL database layer using SQLAlchemy 2.0 async ORM.

Tables:
  • cases         — one row per judgment (full metadata)
  • chunks        — one row per text chunk (FK → cases)
  • pipeline_runs — audit log of each processing batch

Uses asyncpg for async operations (FastAPI compatibility) and
psycopg2 for sync operations (Alembic migrations, scripts).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional, Any

from loguru import logger
from sqlalchemy import (
    Column, String, Integer, BigInteger, Text, Date, DateTime,
    Float, Boolean, ForeignKey, JSON, func, select, update, exists,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ──────────────────────────────────────────────────────────────────
#  SQLAlchemy base
# ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────────
#  ORM Models
# ──────────────────────────────────────────────────────────────────

class Case(Base):
    """
    One row per Supreme Court judgment.
    Stores all extracted metadata.
    """
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(200), unique=True, nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    file_name = Column(String(500))
    file_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Case identifiers
    case_number = Column(String(500))
    case_type = Column(String(100), index=True)
    year = Column(Integer, index=True)

    # Dates
    judgment_date = Column(Date)
    filing_date = Column(Date)

    # Parties
    petitioner = Column(Text)
    respondent = Column(Text)

    # Bench
    bench = Column(Text)
    judges = Column(JSON)               # list[str]

    # Legal references
    acts_cited = Column(JSON)           # list[str]
    citations = Column(JSON)            # list[str]

    # Outcome
    outcome = Column(String(100), index=True)

    # Stats
    page_count = Column(Integer)
    word_count = Column(Integer)
    extraction_method = Column(String(50))
    extraction_confidence = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    chunks = relationship("Chunk", back_populates="case", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Case case_id={self.case_id} year={self.year} type={self.case_type}>"


class Chunk(Base):
    """
    One row per text chunk extracted from a judgment.
    Links to the FAISS index via faiss_id.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(200), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_count = Column(Integer)
    approx_token_count = Column(Integer)
    total_chunks = Column(Integer)
    faiss_id = Column(BigInteger, index=True)   # ID assigned in FAISS index (-1 if not yet indexed)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    case = relationship("Case", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk case_id={self.case_id} idx={self.chunk_index} faiss_id={self.faiss_id}>"


class PipelineRun(Base):
    """
    Audit log for each pipeline batch run.
    """
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), default="running")  # running / completed / failed
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    skipped_files = Column(Integer, default=0)      # already in checkpoint
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_log = Column(Text)

    def __repr__(self):
        return f"<PipelineRun run_id={self.run_id} status={self.status}>"


# ──────────────────────────────────────────────────────────────────
#  Engine & session factory
# ──────────────────────────────────────────────────────────────────

_engine = None
_session_factory = None


def init_engine(db_url: str):
    """
    Initialize the async SQLAlchemy engine and session factory.
    Call this once at application startup.

    Args:
        db_url: Async PostgreSQL URL, e.g.
                "postgresql+asyncpg://user:pass@localhost:5432/judicial_db"
    """
    global _engine, _session_factory
    _engine = create_async_engine(
        db_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    logger.info(f"Database engine initialized: {db_url[:50]}...")


def get_session() -> AsyncSession:
    """
    Dependency-injectable session factory.
    Use as FastAPI dependency: `session: AsyncSession = Depends(get_session)`.
    """
    if _session_factory is None:
        raise RuntimeError("Database engine not initialized. Call init_engine() first.")
    return _session_factory()


async def create_tables():
    """Create all tables if they don't already exist."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")


# ──────────────────────────────────────────────────────────────────
#  CRUD operations
# ──────────────────────────────────────────────────────────────────

class CaseRepository:
    """Async CRUD operations for the `cases` table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def exists_by_hash(self, file_hash: str) -> bool:
        """Check if a case with the given file_hash already exists."""
        stmt = select(exists().where(Case.file_hash == file_hash))
        result = await self.session.execute(stmt)
        return result.scalar()

    async def upsert(self, case_data: dict) -> Case:
        """
        Insert a new case or update if file_hash already exists.

        Args:
            case_data: Dict matching Case model fields.

        Returns:
            The Case ORM object.
        """
        existing = await self.session.execute(
            select(Case).where(Case.file_hash == case_data["file_hash"])
        )
        obj = existing.scalar_one_or_none()

        if obj is None:
            obj = Case(**case_data)
            self.session.add(obj)
        else:
            for k, v in case_data.items():
                if k not in ("id", "created_at"):
                    setattr(obj, k, v)
            obj.updated_at = datetime.utcnow()

        await self.session.flush()
        return obj

    async def get_by_case_id(self, case_id: str) -> Optional[Case]:
        """Retrieve a case by case_id."""
        result = await self.session.execute(
            select(Case).where(Case.case_id == case_id)
        )
        return result.scalar_one_or_none()

    async def get_by_ids(self, case_ids: list[str]) -> list[Case]:
        """Retrieve multiple cases by their case_ids."""
        result = await self.session.execute(
            select(Case).where(Case.case_id.in_(case_ids))
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Return total number of cases in the database."""
        result = await self.session.execute(select(func.count(Case.id)))
        return result.scalar()


class ChunkRepository:
    """Async CRUD operations for the `chunks` table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_insert(self, chunks_data: list[dict]) -> list[Chunk]:
        """
        Insert multiple chunks in a single transaction.

        Args:
            chunks_data: List of dicts matching Chunk model fields.

        Returns:
            List of created Chunk ORM objects.
        """
        objects = [Chunk(**data) for data in chunks_data]
        self.session.add_all(objects)
        await self.session.flush()
        return objects

    async def get_by_case_id(self, case_id: str) -> list[Chunk]:
        """Retrieve all chunks for a given case, ordered by index."""
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.case_id == case_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_faiss_ids(self, faiss_ids: list[int]) -> list[Chunk]:
        """Retrieve chunks by their FAISS IDs (for search result enrichment)."""
        result = await self.session.execute(
            select(Chunk).where(Chunk.faiss_id.in_(faiss_ids))
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Return total number of chunks in the database."""
        result = await self.session.execute(select(func.count(Chunk.id)))
        return result.scalar()


class PipelineRunRepository:
    """Async CRUD operations for the `pipeline_runs` table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, run_id: str, total_files: int) -> PipelineRun:
        """Create a new pipeline run record."""
        run = PipelineRun(run_id=run_id, total_files=total_files, status="running")
        self.session.add(run)
        await self.session.flush()
        return run

    async def update_progress(
        self,
        run_id: str,
        processed: int,
        failed: int,
        skipped: int,
    ):
        """Update counters on an existing run."""
        await self.session.execute(
            update(PipelineRun)
            .where(PipelineRun.run_id == run_id)
            .values(
                processed_files=processed,
                failed_files=failed,
                skipped_files=skipped,
            )
        )

    async def complete(self, run_id: str, status: str = "completed", error: Optional[str] = None):
        """Mark a pipeline run as completed or failed."""
        await self.session.execute(
            update(PipelineRun)
            .where(PipelineRun.run_id == run_id)
            .values(
                status=status,
                completed_at=datetime.utcnow(),
                error_log=error,
            )
        )

    async def get(self, run_id: str) -> Optional[PipelineRun]:
        """Get a pipeline run by ID."""
        result = await self.session.execute(
            select(PipelineRun).where(PipelineRun.run_id == run_id)
        )
        return result.scalar_one_or_none()
