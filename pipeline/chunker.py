"""
pipeline/chunker.py
─────────────────────────────────────────────────────────────────────────────
Text chunking layer for RAG (Retrieval-Augmented Generation).

Strategy: Recursive character splitter that respects natural text boundaries
in the following priority order:
  1. Paragraph breaks (\n\n)
  2. Single newlines (\n)
  3. Sentence-ending punctuation (. ! ?)
  4. Commas and spaces
  5. Character-level fallback

Each chunk is tagged with its case_id and chunk index so it can be
linked back to the parent case in PostgreSQL and FAISS.

Target: ~512 tokens per chunk with 64-token overlap.
Token counting: approximate via word count × 1.35 (sub-word factor).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from loguru import logger


# ──────────────────────────────────────────────────────────────────
#  Data model
# ──────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk ready for embedding."""
    case_id: str
    chunk_index: int            # 0-based index within the document
    text: str
    char_count: int
    approx_token_count: int     # Approximate (word × 1.35)
    total_chunks: int = 0       # Filled in after all chunks are generated


# ──────────────────────────────────────────────────────────────────
#  Token counting helper
# ──────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """
    Approximate token count without loading a tokenizer.
    Sub-word multiplier ~1.35 accounts for wordpiece tokenization.
    """
    return int(len(text.split()) * 1.35)


# ──────────────────────────────────────────────────────────────────
#  Recursive character splitter
# ──────────────────────────────────────────────────────────────────

class RecursiveCharacterSplitter:
    """
    Splits text recursively using a priority list of separators.

    Mirrors the behavior of LangChain's RecursiveCharacterTextSplitter
    but is self-contained (no LangChain dependency).

    Args:
        chunk_size:    Target maximum tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks.
        separators:    Priority list of separator strings to try.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """
        Merge small splits into chunks of up to chunk_size tokens,
        with overlap between consecutive chunks.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for split in splits:
            split_tokens = _approx_tokens(split)

            # If adding this split would exceed chunk_size, flush current chunk
            if current_tokens + split_tokens > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap: pop from front until we're within overlap budget
                while current_parts and current_tokens > self.chunk_overlap:
                    removed_tokens = _approx_tokens(current_parts[0])
                    current_tokens -= removed_tokens
                    current_parts.pop(0)

            current_parts.append(split)
            current_tokens += split_tokens

        # Flush remaining
        if current_parts:
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks using recursive separator strategy.

        Returns a flat list of text chunks.
        """
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Internal recursive splitting logic."""
        if not text.strip():
            return []

        # If the text already fits in one chunk, return it
        if _approx_tokens(text) <= self.chunk_size:
            return [text.strip()]

        # Try each separator in priority order
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: hard character split
                return self._hard_split(text)

            if sep in text:
                splits = text.split(sep)
                # Filter out empty splits
                splits = [s for s in splits if s.strip()]

                # Recursively split any oversized individual splits
                good_splits: list[str] = []
                for split in splits:
                    if _approx_tokens(split) > self.chunk_size:
                        # This split is still too big — recurse with remaining seps
                        good_splits.extend(
                            self._split_recursive(split, separators[i + 1:])
                        )
                    else:
                        good_splits.append(split)

                return self._merge_splits(good_splits, sep)

        return [text.strip()]

    def _hard_split(self, text: str) -> list[str]:
        """
        Hard character-level split as absolute last resort.
        Splits text into fixed-size character windows with overlap.
        """
        # Convert tokens to chars approx (1 token ≈ 5 chars)
        char_size = self.chunk_size * 5
        char_overlap = self.chunk_overlap * 5
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + char_size
            chunks.append(text[start:end].strip())
            start += char_size - char_overlap
        return [c for c in chunks if c]


# ──────────────────────────────────────────────────────────────────
#  TextChunker — main public class
# ──────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Converts a cleaned document into a list of TextChunk objects.

    Usage:
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(case_id, clean_text)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, case_id: str, clean_text: str) -> list[TextChunk]:
        """
        Split clean text into overlapping chunks for embedding.

        Args:
            case_id:    Unique identifier for the parent case.
            clean_text: Cleaned judgment text.

        Returns:
            List of TextChunk objects with sequential indices.
        """
        raw_chunks = self.splitter.split_text(clean_text)

        # Filter very short chunks (likely noise)
        raw_chunks = [c for c in raw_chunks if len(c.split()) >= 10]

        chunks: list[TextChunk] = []
        for i, text in enumerate(raw_chunks):
            chunks.append(
                TextChunk(
                    case_id=case_id,
                    chunk_index=i,
                    text=text,
                    char_count=len(text),
                    approx_token_count=_approx_tokens(text),
                    total_chunks=len(raw_chunks),
                )
            )
            # Update total_chunks field for all chunks
            for chunk in chunks:
                chunk.total_chunks = len(raw_chunks)

        logger.debug(
            f"Chunked case_id={case_id} → {len(chunks)} chunks "
            f"(avg ~{int(sum(c.approx_token_count for c in chunks) / max(len(chunks),1))} tokens each)"
        )
        return chunks

    def chunk_batch(self, documents: list[tuple[str, str]]) -> Generator[list[TextChunk], None, None]:
        """
        Chunk multiple documents.

        Args:
            documents: List of (case_id, clean_text) tuples.

        Yields:
            List of TextChunks for each document.
        """
        for case_id, clean_text in documents:
            try:
                yield self.chunk(case_id, clean_text)
            except Exception as e:
                logger.error(f"Chunking failed for case_id={case_id}: {e}")
                yield []
