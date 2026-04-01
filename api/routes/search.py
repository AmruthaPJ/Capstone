"""
api/routes/search.py
─────────────────────────────────────────────────────────────────────────────
Semantic search endpoint.

POST /api/v1/search
  Body: {
    "query": "bail application in murder cases",
    "top_k": 10,
    "filters": {
      "year_from": 2010,
      "year_to": 2024,
      "case_type": "Criminal Appeal",
      "outcome": "Allowed"
    }
  }
  Returns: List of matching chunks with case metadata.

Flow:
  1. Embed the query using all-MiniLM-L6-v2
  2. Search FAISS for top-(k × 3) candidates (oversampling for filter headroom)
  3. Fetch chunk text + case metadata from PostgreSQL
  4. Apply metadata filters
  5. Return top-k results
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

router = APIRouter(tags=["Search"])


# ──────────────────────────────────────────────────────────────────
#  Request / Response models
# ──────────────────────────────────────────────────────────────────

class SearchFilters(BaseModel):
    year_from: Optional[int] = Field(default=None, ge=1950, le=2024)
    year_to: Optional[int] = Field(default=None, ge=1950, le=2024)
    case_type: Optional[str] = Field(default=None, description="e.g. 'Civil Appeal', 'Criminal Appeal'")
    outcome: Optional[str] = Field(default=None, description="e.g. 'Allowed', 'Dismissed'")
    judge_name: Optional[str] = Field(default=None, description="Partial judge name to filter by")
    act_cited: Optional[str] = Field(default=None, description="Act name to filter by, e.g. 'IPC'")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language legal query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(default=None)
    include_chunk_text: bool = Field(
        default=True,
        description="Include the matching chunk text in results (disable for lighter responses)",
    )


class CaseSnippet(BaseModel):
    """Metadata fields included with each search result."""
    case_id: str
    case_number: Optional[str]
    case_type: Optional[str]
    year: Optional[int]
    judgment_date: Optional[str]
    petitioner: Optional[str]
    respondent: Optional[str]
    bench: Optional[str]
    judges: list[str]
    outcome: Optional[str]
    acts_cited: list[str]
    citations: list[str]
    file_name: Optional[str]
    extraction_confidence: Optional[float]


class SearchHit(BaseModel):
    """A single search result."""
    rank: int
    score: float                    # Cosine similarity (0.0 – 1.0)
    case: CaseSnippet
    chunk_index: int
    chunk_text: Optional[str]       # The matching text passage


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    results: list[SearchHit]


# ──────────────────────────────────────────────────────────────────
#  Filtering helpers
# ──────────────────────────────────────────────────────────────────

def _case_matches_filters(case, filters: SearchFilters) -> bool:
    """Check if a case ORM object satisfies all active filters."""
    if filters.year_from and case.year and case.year < filters.year_from:
        return False
    if filters.year_to and case.year and case.year > filters.year_to:
        return False
    if filters.case_type and case.case_type:
        if filters.case_type.lower() not in case.case_type.lower():
            return False
    if filters.outcome and case.outcome:
        if filters.outcome.lower() not in case.outcome.lower():
            return False
    if filters.judge_name and case.judges:
        judges_lower = [j.lower() for j in (case.judges or [])]
        if not any(filters.judge_name.lower() in j for j in judges_lower):
            return False
    if filters.act_cited and case.acts_cited:
        acts_lower = [a.lower() for a in (case.acts_cited or [])]
        if not any(filters.act_cited.lower() in a for a in acts_lower):
            return False
    return True


def _case_to_snippet(case) -> CaseSnippet:
    """Convert a Case ORM object to a CaseSnippet Pydantic model."""
    return CaseSnippet(
        case_id=case.case_id,
        case_number=case.case_number,
        case_type=case.case_type,
        year=case.year,
        judgment_date=str(case.judgment_date) if case.judgment_date else None,
        petitioner=case.petitioner,
        respondent=case.respondent,
        bench=case.bench,
        judges=case.judges or [],
        outcome=case.outcome,
        acts_cited=case.acts_cited or [],
        citations=case.citations or [],
        file_name=case.file_name,
        extraction_confidence=case.extraction_confidence,
    )


# ──────────────────────────────────────────────────────────────────
#  Search endpoint
# ──────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search over indexed Supreme Court judgments.

    Uses FAISS for vector similarity + PostgreSQL for metadata filtering.

    **Example queries**:
    - "bail application in murder cases"
    - "constitutional validity of sedition law"  
    - "anticipatory bail grounds domestic violence"
    - "land acquisition compensation inadequate"
    """
    from api.main import app_state
    from pipeline.db import get_session, ChunkRepository, CaseRepository

    if app_state.vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=503,
            detail="Search index is empty. Run the ingestion pipeline first.",
        )

    # ── Step 1: Embed query ──────────────────────────────────────
    logger.info(f"Search query: '{request.query}'")
    query_vector = app_state.embedder.embed_query(request.query)

    # Oversample to leave headroom for metadata filtering
    oversample_k = min(request.top_k * 5, app_state.vector_store.total_vectors)
    faiss_results = app_state.vector_store.search(query_vector, top_k=oversample_k)

    if not faiss_results:
        return SearchResponse(query=request.query, total_hits=0, results=[])

    # ── Step 2: Fetch chunks and cases from PostgreSQL ───────────
    faiss_ids = [r.faiss_id for r in faiss_results]
    score_map = {r.faiss_id: r.score for r in faiss_results}
    case_id_map = {r.faiss_id: r.case_id for r in faiss_results}

    async with get_session() as session:
        chunk_repo = ChunkRepository(session)
        case_repo = CaseRepository(session)

        chunks = await chunk_repo.get_by_faiss_ids(faiss_ids)

        unique_case_ids = list({c.case_id for c in chunks})
        cases = await case_repo.get_by_ids(unique_case_ids)

    # Build lookup maps
    chunk_map = {c.faiss_id: c for c in chunks}
    case_map = {c.case_id: c for c in cases}

    # ── Step 3: Build results with metadata filtering ────────────
    hits: list[SearchHit] = []

    for faiss_result in faiss_results:
        if len(hits) >= request.top_k:
            break

        chunk = chunk_map.get(faiss_result.faiss_id)
        if chunk is None:
            continue

        case = case_map.get(chunk.case_id)
        if case is None:
            continue

        # Apply metadata filters
        if request.filters and not _case_matches_filters(case, request.filters):
            continue

        hits.append(
            SearchHit(
                rank=len(hits) + 1,
                score=round(faiss_result.score, 4),
                case=_case_to_snippet(case),
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text if request.include_chunk_text else None,
            )
        )

    logger.info(
        f"Search complete: query='{request.query}' → {len(hits)} results "
        f"(from {len(faiss_results)} FAISS candidates)"
    )

    return SearchResponse(
        query=request.query,
        total_hits=len(hits),
        results=hits,
    )
