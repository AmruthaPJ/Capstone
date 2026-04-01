"""
scripts/init_db.py
─────────────────────────────────────────────────────────────────────────────
Database initialization script.

Use this to create/verify tables programmatically (alternative to Docker
initdb or Alembic).

Usage:
  python scripts/init_db.py
  python scripts/init_db.py --drop-all   # ⚠️  Drop all tables first
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine

from config import settings
from pipeline.db import Base, create_tables, init_engine


async def init(drop_all: bool = False):
    """Initialize or re-initialize the database schema."""
    init_engine(settings.db_url)

    if drop_all:
        logger.warning("⚠️  Dropping all tables...")
        engine = create_async_engine(settings.db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
        logger.info("All tables dropped.")

    await create_tables()
    logger.info("✓ Database initialized successfully.")


def main():
    parser = argparse.ArgumentParser(description="Initialize the judicial AI database")
    parser.add_argument(
        "--drop-all",
        action="store_true",
        help="⚠️  Drop all existing tables before creating (destructive!)",
    )
    args = parser.parse_args()
    asyncio.run(init(drop_all=args.drop_all))


if __name__ == "__main__":
    main()
