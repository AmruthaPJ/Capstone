"""pipeline/__init__.py — Public API for the pipeline package."""

from pipeline.extractor import PDFExtractor, RawDocument
from pipeline.cleaner import TextCleaner, CleanDocument
from pipeline.metadata import MetadataExtractor, CaseMetadata
from pipeline.chunker import TextChunker, TextChunk
from pipeline.embedder import Embedder
from pipeline.vector_store import FAISSVectorStore

__all__ = [
    "PDFExtractor", "RawDocument",
    "TextCleaner", "CleanDocument",
    "MetadataExtractor", "CaseMetadata",
    "TextChunker", "TextChunk",
    "Embedder",
    "FAISSVectorStore",
]
