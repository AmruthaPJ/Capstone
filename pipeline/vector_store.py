"""
pipeline/vector_store.py
─────────────────────────────────────────────────────────────────────────────
FAISS vector store management.

Supports two index types:
  • IndexFlatIP  (default)  — exact search, best for dev / <100K vectors
  • IndexIVFFlat (optional) — approximate ANN, faster at scale (>100K vectors)

FAISS IDs are sequential integers. We maintain a separate ID-mapping file
(faiss_id_map.json) to link FAISS IDs back to (case_id, chunk_index).

Index files:
  data/faiss_index/
    ├── index.faiss          — FAISS binary index
    └── id_map.json          — { faiss_id (int): {case_id, chunk_index} }
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger


# ──────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────

INDEX_FILE = "index.faiss"
ID_MAP_FILE = "id_map.json"


# ──────────────────────────────────────────────────────────────────
#  Search result dataclass
# ──────────────────────────────────────────────────────────────────

class SearchResult:
    """A single FAISS search result."""

    def __init__(self, faiss_id: int, score: float, case_id: str, chunk_index: int):
        self.faiss_id = faiss_id
        self.score = score          # cosine similarity (0.0 – 1.0 after L2 norm)
        self.case_id = case_id
        self.chunk_index = chunk_index

    def __repr__(self):
        return (
            f"SearchResult(case_id={self.case_id}, chunk={self.chunk_index}, "
            f"score={self.score:.4f})"
        )


# ──────────────────────────────────────────────────────────────────
#  FAISS vector store
# ──────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Manages a FAISS index for storing and searching legal text embeddings.

    Usage (write):
        store = FAISSVectorStore(index_dir=Path("data/faiss_index"), dim=384)
        store.add(vectors, case_ids, chunk_indices)
        store.save()

    Usage (read / search):
        store = FAISSVectorStore.load(index_dir=Path("data/faiss_index"), dim=384)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        index_dir: Path,
        dim: int = 384,
        use_ivf: bool = False,
        nlist: int = 256,
    ):
        self.index_dir = Path(index_dir)
        self.dim = dim
        self.use_ivf = use_ivf
        self.nlist = nlist

        self._index: Optional[faiss.Index] = None
        self._id_map: dict[int, dict] = {}   # faiss_id → {case_id, chunk_index}
        self._next_id: int = 0               # monotonically increasing FAISS ID

        self.index_dir.mkdir(parents=True, exist_ok=True)

    # ── Index initialization ──────────────────────────────────────

    def _build_flat_index(self) -> faiss.Index:
        """IndexFlatIP: exact inner-product search (= cosine after L2 norm)."""
        index = faiss.IndexFlatIP(self.dim)
        # Wrap in IDMap2 so we can assign our own IDs
        return faiss.IndexIDMap2(index)

    def _build_ivf_index(self) -> faiss.Index:
        """
        IndexIVFFlat: approximate nearest neighbor with inverted file.
        Requires training on representative data before use.
        """
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap2(index)

    def _ensure_index(self):
        """Create the index if it doesn't exist yet."""
        if self._index is None:
            if self.use_ivf:
                self._index = self._build_ivf_index()
                logger.info(f"Created FAISS IndexIVFFlat (nlist={self.nlist}, dim={self.dim})")
            else:
                self._index = self._build_flat_index()
                logger.info(f"Created FAISS IndexFlatIP (dim={self.dim})")

    # ── Adding vectors ─────────────────────────────────────────────

    def add(
        self,
        vectors: np.ndarray,
        case_ids: list[str],
        chunk_indices: list[int],
    ) -> list[int]:
        """
        Add a batch of embedding vectors to the index.

        Args:
            vectors:       Float32 numpy array of shape (N, dim). Must be L2-normalized.
            case_ids:      Corresponding case_id per vector.
            chunk_indices: Corresponding chunk_index per vector.

        Returns:
            List of assigned FAISS IDs (sequential integers).
        """
        self._ensure_index()

        n = len(vectors)
        assert vectors.shape == (n, self.dim), \
            f"Expected shape ({n}, {self.dim}), got {vectors.shape}"
        assert len(case_ids) == n and len(chunk_indices) == n

        vectors = vectors.astype(np.float32)

        # Assign sequential IDs
        faiss_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)

        # Train IVF index if needed (first add)
        if self.use_ivf and not self._index.is_trained:  # type: ignore[union-attr]
            if n >= self.nlist:
                logger.info(f"Training IVF index on {n} vectors...")
                self._index.train(vectors)  # type: ignore[union-attr]
            else:
                logger.warning(
                    f"Not enough vectors ({n}) to train IVF index (need >= {self.nlist}). "
                    "Falling back to IndexFlatIP for this batch."
                )
                self._index = self._build_flat_index()
                self._index.train(vectors)  # type: ignore[union-attr]

        self._index.add_with_ids(vectors, faiss_ids)  # type: ignore[union-attr]

        # Update ID map
        for i, fid in enumerate(faiss_ids.tolist()):
            self._id_map[fid] = {
                "case_id": case_ids[i],
                "chunk_index": chunk_indices[i],
            }

        self._next_id += n
        logger.debug(f"Added {n} vectors to FAISS. Total: {self._next_id}")
        return faiss_ids.tolist()

    # ── Searching ─────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search for the top-k most similar vectors to the query.

        Args:
            query_vector: 1D or 2D float32 array (must be L2-normalized).
            top_k:        Number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity score.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        # Reshape to (1, dim) if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)

        # Clamp top_k to index size
        effective_k = min(top_k, self._index.ntotal)

        scores, faiss_ids = self._index.search(query_vector, effective_k)  # type: ignore[union-attr]
        scores = scores[0]
        faiss_ids = faiss_ids[0]

        results: list[SearchResult] = []
        for score, fid in zip(scores, faiss_ids):
            if fid == -1:
                continue  # FAISS returns -1 for missing entries
            entry = self._id_map.get(int(fid))
            if entry is None:
                continue
            results.append(
                SearchResult(
                    faiss_id=int(fid),
                    score=float(score),
                    case_id=entry["case_id"],
                    chunk_index=entry["chunk_index"],
                )
            )

        return results

    # ── Persistence ───────────────────────────────────────────────

    def save(self):
        """Persist the FAISS index and ID map to disk."""
        if self._index is None:
            logger.warning("Nothing to save — index is empty.")
            return

        index_path = self.index_dir / INDEX_FILE
        id_map_path = self.index_dir / ID_MAP_FILE

        faiss.write_index(self._index, str(index_path))
        with open(id_map_path, "w") as f:
            json.dump(self._id_map, f)

        logger.info(
            f"FAISS index saved: {index_path} "
            f"({self._index.ntotal} vectors, {len(self._id_map)} entries in ID map)"
        )

    @classmethod
    def load(
        cls,
        index_dir: Path,
        dim: int = 384,
        use_ivf: bool = False,
        nlist: int = 256,
    ) -> "FAISSVectorStore":
        """
        Load a previously saved FAISS index from disk.

        Args:
            index_dir: Directory containing index.faiss and id_map.json.
            dim:       Expected embedding dimension.

        Returns:
            FAISSVectorStore with loaded index.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / INDEX_FILE
        id_map_path = index_dir / ID_MAP_FILE

        store = cls(index_dir=index_dir, dim=dim, use_ivf=use_ivf, nlist=nlist)

        if index_path.exists():
            store._index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index: {index_path} ({store._index.ntotal} vectors)")
        else:
            logger.warning(f"No FAISS index found at {index_path}. Starting fresh.")

        if id_map_path.exists():
            with open(id_map_path) as f:
                raw = json.load(f)
            # JSON keys are always strings — convert back to int
            store._id_map = {int(k): v for k, v in raw.items()}
            if store._id_map:
                store._next_id = max(store._id_map.keys()) + 1
            logger.info(f"Loaded ID map: {len(store._id_map)} entries")

        return store

    # ── Properties ────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Total number of vectors stored in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def __repr__(self):
        return (
            f"FAISSVectorStore(dim={self.dim}, total={self.total_vectors}, "
            f"type={'IVF' if self.use_ivf else 'Flat'})"
        )
