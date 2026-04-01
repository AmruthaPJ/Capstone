"""
pipeline/embedder.py
─────────────────────────────────────────────────────────────────────────────
Embedding generation layer using sentence-transformers.

Model: sentence-transformers/all-MiniLM-L6-v2
  • Output dimension: 384
  • CPU-optimized, ~80MB model size
  • Suitable for semantic similarity search

All embeddings are L2-normalized so inner-product in FAISS equals
cosine similarity.

The model is loaded once as a singleton to avoid repeated HuggingFace
downloads and avoid reloading for each batch.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────
#  Singleton model loader
# ──────────────────────────────────────────────────────────────────

_model_instance = None  # Global cache — loaded once per process


def _get_model(model_name: str):
    """
    Lazily load and cache the SentenceTransformer model.
    Thread-safe on CPython due to GIL.
    """
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _model_instance = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Embedding dimension: {_model_instance.get_sentence_embedding_dimension()}")
    return _model_instance


# ──────────────────────────────────────────────────────────────────
#  Embedding result
# ──────────────────────────────────────────────────────────────────

class EmbeddingResult:
    """Holds generated embeddings for a batch of text chunks."""

    def __init__(self, vectors: np.ndarray, texts: list[str]):
        self.vectors = vectors          # shape: (N, embedding_dim), dtype float32
        self.texts = texts
        self.count = len(texts)
        self.dim = vectors.shape[1] if vectors.ndim == 2 else 0

    def __repr__(self):
        return f"EmbeddingResult(count={self.count}, dim={self.dim})"


# ──────────────────────────────────────────────────────────────────
#  Embedder class
# ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Generates L2-normalized embeddings for text chunks using
    sentence-transformers/all-MiniLM-L6-v2.

    Usage:
        embedder = Embedder()
        result = embedder.embed(["text 1", "text 2"])
        # result.vectors.shape → (2, 384)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._dim: Optional[int] = None

    @property
    def model(self):
        return _get_model(self.model_name)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = self.model.get_sentence_embedding_dimension()
        return self._dim

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of text strings.

        Processes in batches to avoid OOM on large inputs.
        Applies L2 normalization if normalize=True (required for FAISS IndexFlatIP).

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with numpy array of shape (len(texts), 384).
        """
        if not texts:
            empty = np.zeros((0, self.embedding_dim), dtype=np.float32)
            return EmbeddingResult(empty, [])

        logger.debug(f"Embedding {len(texts)} texts in batches of {self.batch_size}")

        # sentence_transformers handles batching internally but we log per batch
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,  # built-in L2 normalization
        )

        vectors = vectors.astype(np.float32)

        logger.debug(f"Embeddings generated: shape={vectors.shape}")
        return EmbeddingResult(vectors, texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for semantic search.

        Returns:
            1D numpy array of shape (384,), float32, L2-normalized.
        """
        result = self.embed([query])
        return result.vectors[0]

    def embed_chunks(
        self,
        chunks,  # list[TextChunk]
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Embed a list of TextChunk objects.

        Args:
            chunks:        List of TextChunk objects.
            show_progress: Whether to display a progress bar.

        Returns:
            Tuple of (embeddings array shape (N, dim), list of case_ids per chunk)
        """
        from tqdm import tqdm

        texts = [c.text for c in chunks]
        case_ids = [c.case_id for c in chunks]

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        all_vectors: list[np.ndarray] = []
        total = len(texts)

        batch_iter = range(0, total, self.batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc="Generating embeddings", unit="batch")

        for start in batch_iter:
            batch_texts = texts[start: start + self.batch_size]
            result = self.embed(batch_texts)
            all_vectors.append(result.vectors)

        embeddings = np.vstack(all_vectors).astype(np.float32)
        return embeddings, case_ids
