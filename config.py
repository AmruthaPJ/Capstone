"""
config.py — Centralized configuration using Pydantic BaseSettings.

All values are read from environment variables (or .env file).
Override any value in your .env file (copy from .env.example).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Paths ──────────────────────────────────────────────────────
    raw_pdf_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    faiss_index_dir: Path = Path("data/faiss_index")
    checkpoint_db: Path = Path("data/checkpoint.db")

    # ── PostgreSQL ─────────────────────────────────────────────────
    db_url: str = "postgresql+asyncpg://judicial:judicial@localhost:5432/judicial_db"
    db_sync_url: str = "postgresql+psycopg2://judicial:judicial@localhost:5432/judicial_db"

    # ── Embedding ──────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_dim: int = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

    # ── Chunking ───────────────────────────────────────────────────
    chunk_size: int = 512        # approximate tokens per chunk
    chunk_overlap: int = 64      # overlap to preserve context across chunks

    # ── FAISS ──────────────────────────────────────────────────────
    use_ivf: bool = False        # False → IndexFlatIP; True → IndexIVFFlat
    faiss_nlist: int = 256       # Number of IVF clusters (production scale)

    # ── Pipeline ───────────────────────────────────────────────────
    max_workers: int = 4         # Workers for parallel PDF extraction

    # ── FastAPI ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("raw_pdf_dir", "processed_dir", "faiss_index_dir", mode="before")
    @classmethod
    def _make_path(cls, v):
        return Path(v)

    def create_dirs(self):
        """Ensure all required directories exist."""
        for directory in [self.raw_pdf_dir, self.processed_dir, self.faiss_index_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


# Singleton settings instance used throughout the project
settings = Settings()
