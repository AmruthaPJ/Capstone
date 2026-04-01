"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the preprocessing pipeline modules.

Run with: pytest tests/ -v

Uses synthetic test data — no real PDFs required for unit tests.
Integration tests (marked @pytest.mark.integration) require Docker DB.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────
#  Sample data fixtures
# ──────────────────────────────────────────────────────────────────

SAMPLE_JUDGMENT = """
IN THE SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION

CIVIL APPEAL NO. 1234 OF 2020
(Arising out of SLP (C) No. 5678 of 2019)

UNION OF INDIA                                           ... APPELLANT(S)

VERSUS

STATE OF MAHARASHTRA & ORS.                              ... RESPONDENT(S)

CORAM:
HON'BLE MR. JUSTICE A.K. SIKRI
HON'BLE MR. JUSTICE S. ABDUL NAZEER

JUDGMENT

Delivered by HON'BLE MR. JUSTICE A.K. SIKRI

1. This appeal arises from the judgment of the High Court of Bombay
dated 15th March 2019.

2. The petitioner challenges the constitutional validity of Section 124A
of the Indian Penal Code (IPC), which deals with sedition.

3. The respondent State relies upon the provisions of the Code of Criminal
Procedure (CrPC) and the Constitution of India, Article 19(2).

Reported in AIR 2020 SC 1234 and (2020) 5 SCC 123.

This appeal is allowed. The order of the High Court is set aside.

DATED: 20th January 2020
"""

SAMPLE_NOISY_TEXT = """
IN THE SUPREME COURT OF INDIA
Page 1

Indian Kanoon - http://indiankanoon.org/doc/12345/

CIVIL APPELLATE JURISDICTION

Page 2

CIVIL APPEAL NO. 555 OF 2021

The petitioner...
"""


# ──────────────────────────────────────────────────────────────────
#  TextCleaner tests
# ──────────────────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from pipeline.cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_removes_control_chars(self):
        text = "Hello\x00World\x0bTest"
        result = self.cleaner.clean(text, "test.pdf", "abc123")
        assert "\x00" not in result.clean_text
        assert "\x0b" not in result.clean_text

    def test_removes_indian_kanoon_boilerplate(self):
        result = self.cleaner.clean(SAMPLE_NOISY_TEXT, "test.pdf", "abc123")
        assert "indiankanoon.org" not in result.clean_text.lower()

    def test_removes_page_numbers(self):
        result = self.cleaner.clean(SAMPLE_NOISY_TEXT, "test.pdf", "abc123")
        # Standalone "Page 1" should be gone
        assert "Page 1\n" not in result.clean_text

    def test_word_count_positive(self):
        result = self.cleaner.clean(SAMPLE_JUDGMENT, "test.pdf", "abc123")
        assert result.word_count > 0

    def test_empty_text_flagged(self):
        result = self.cleaner.clean("   \n  ", "test.pdf", "abc123")
        assert result.is_empty is True

    def test_normalized_unicode(self):
        # Ligature ﬁ → fi
        text = "The ﬁnal judgment of the court."
        result = self.cleaner.clean(text, "test.pdf", "abc123")
        assert "ﬁ" not in result.clean_text
        assert "fi" in result.clean_text

    def test_hyphenated_word_rejoin(self):
        text = "The appli-\ncation was dismissed by the court."
        result = self.cleaner.clean(text, "test.pdf", "abc123")
        assert "application" in result.clean_text


# ──────────────────────────────────────────────────────────────────
#  MetadataExtractor tests
# ──────────────────────────────────────────────────────────────────

class TestMetadataExtractor:
    def setup_method(self):
        from pipeline.metadata import MetadataExtractor
        self.extractor = MetadataExtractor()

    def _extract(self, text=SAMPLE_JUDGMENT):
        return self.extractor.extract(
            clean_text=text,
            file_path="test.pdf",
            file_name="test.pdf",
            file_hash="abc123def456",
        )

    def test_extracts_case_number(self):
        meta = self._extract()
        assert meta.case_number is not None
        assert "1234" in meta.case_number
        assert "2020" in meta.case_number

    def test_extracts_case_type(self):
        meta = self._extract()
        assert meta.case_type is not None
        assert "Civil" in meta.case_type

    def test_extracts_year(self):
        meta = self._extract()
        assert meta.year == 2020

    def test_extracts_judges(self):
        meta = self._extract()
        assert len(meta.judges) >= 1
        # Should contain one of the known judge names
        names_lower = [j.lower() for j in meta.judges]
        assert any("sikri" in n or "nazeer" in n for n in names_lower)

    def test_extracts_petitioner(self):
        meta = self._extract()
        assert meta.petitioner is not None
        assert "Union of India" in meta.petitioner or "UNION OF INDIA" in meta.petitioner.upper()

    def test_extracts_respondent(self):
        meta = self._extract()
        assert meta.respondent is not None
        assert "Maharashtra" in meta.respondent or "MAHARASHTRA" in meta.respondent.upper()

    def test_extracts_acts(self):
        meta = self._extract()
        assert len(meta.acts_cited) > 0
        acts_upper = [a.upper() for a in meta.acts_cited]
        # Should find IPC
        assert any("PENAL" in a or "IPC" in a for a in acts_upper)

    def test_extracts_citations(self):
        meta = self._extract()
        assert len(meta.citations) > 0
        # Should find AIR 2020 SC 1234 and (2020) 5 SCC 123
        citations_str = " ".join(meta.citations)
        assert "2020" in citations_str

    def test_extracts_outcome(self):
        meta = self._extract()
        assert meta.outcome is not None
        assert "allow" in meta.outcome.lower() or "set aside" in meta.outcome.lower()

    def test_confidence_positive(self):
        meta = self._extract()
        assert meta.extraction_confidence > 0.0

    def test_hash_fallback_case_id(self):
        # If no case number found, case_id should start with "sha_"
        meta = self.extractor.extract(
            clean_text="This is a short text with no case number.",
            file_path="unknown.pdf",
            file_name="unknown.pdf",
            file_hash="deadbeef12345678",
        )
        assert meta.case_id.startswith("sha_deadbeef")


# ──────────────────────────────────────────────────────────────────
#  TextChunker tests
# ──────────────────────────────────────────────────────────────────

class TestTextChunker:
    def setup_method(self):
        from pipeline.chunker import TextChunker
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_basic_chunking(self):
        text = " ".join(["word"] * 500)
        chunks = self.chunker.chunk("TEST_CASE_001", text)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self):
        text = " ".join(["word"] * 500)
        chunks = self.chunker.chunk("TEST_CASE_001", text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_case_id_preserved(self):
        chunks = self.chunker.chunk("MY_CASE_ID", SAMPLE_JUDGMENT)
        assert all(c.case_id == "MY_CASE_ID" for c in chunks)

    def test_total_chunks_filled(self):
        chunks = self.chunker.chunk("MY_CASE_ID", SAMPLE_JUDGMENT)
        expected = len(chunks)
        assert all(c.total_chunks == expected for c in chunks)

    def test_no_empty_chunks(self):
        chunks = self.chunker.chunk("MY_CASE_ID", SAMPLE_JUDGMENT)
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_single_short_text(self):
        chunks = self.chunker.chunk("SHORT", "This is a very short text.")
        # Should produce at most 1 chunk (or 0 if too short)
        assert len(chunks) <= 1


# ──────────────────────────────────────────────────────────────────
#  Embedder tests
# ──────────────────────────────────────────────────────────────────

class TestEmbedder:
    def setup_method(self):
        from pipeline.embedder import Embedder
        self.embedder = Embedder(batch_size=4)

    def test_embed_single(self):
        result = self.embedder.embed(["bail application in murder cases"])
        assert result.vectors.shape == (1, 384)
        assert result.vectors.dtype == np.float32

    def test_embed_batch(self):
        texts = ["text one", "text two", "text three"]
        result = self.embedder.embed(texts)
        assert result.vectors.shape == (3, 384)

    def test_l2_normalized(self):
        """L2 norm of each vector should be ~1.0 after normalization."""
        result = self.embedder.embed(["constitutional law of india"])
        norms = np.linalg.norm(result.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_input(self):
        result = self.embedder.embed([])
        assert result.count == 0
        assert result.vectors.shape[0] == 0

    def test_embed_query_shape(self):
        vec = self.embedder.embed_query("test query")
        assert vec.shape == (384,)

    def test_semantic_similarity(self):
        """Semantically similar texts should have higher cosine similarity."""
        legal1 = self.embedder.embed_query("bail application criminal case")
        legal2 = self.embedder.embed_query("bail petition in criminal proceedings")
        unrelated = self.embedder.embed_query("recipe for chocolate cake")

        cos_similar = float(np.dot(legal1, legal2))
        cos_unrelated = float(np.dot(legal1, unrelated))

        assert cos_similar > cos_unrelated, (
            f"Similar texts should score higher: {cos_similar:.3f} vs {cos_unrelated:.3f}"
        )


# ──────────────────────────────────────────────────────────────────
#  FAISSVectorStore tests
# ──────────────────────────────────────────────────────────────────

class TestFAISSVectorStore:
    def setup_method(self, tmp_path_factory=None):
        import tempfile
        self.tmp_dir = Path(tempfile.mkdtemp())
        from pipeline.vector_store import FAISSVectorStore
        self.store = FAISSVectorStore(index_dir=self.tmp_dir, dim=384)

    def _random_vectors(self, n: int) -> np.ndarray:
        """Generate random L2-normalized vectors."""
        vecs = np.random.randn(n, 384).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_add_and_search(self):
        vecs = self._random_vectors(10)
        case_ids = [f"case_{i}" for i in range(10)]
        chunk_indices = list(range(10))

        self.store.add(vecs, case_ids, chunk_indices)
        results = self.store.search(vecs[0], top_k=5)

        assert len(results) > 0
        # Top result should be the query itself (score ~1.0)
        assert results[0].score > 0.99

    def test_total_vectors(self):
        vecs = self._random_vectors(20)
        self.store.add(vecs, [f"c{i}" for i in range(20)], list(range(20)))
        assert self.store.total_vectors == 20

    def test_save_and_load(self):
        vecs = self._random_vectors(5)
        self.store.add(vecs, [f"c{i}" for i in range(5)], list(range(5)))
        self.store.save()

        from pipeline.vector_store import FAISSVectorStore
        loaded = FAISSVectorStore.load(self.tmp_dir, dim=384)
        assert loaded.total_vectors == 5
        assert len(loaded._id_map) == 5

    def test_search_returns_correct_case_ids(self):
        vecs = self._random_vectors(5)
        case_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
        self.store.add(vecs, case_ids, list(range(5)))

        results = self.store.search(vecs[2], top_k=1)
        assert results[0].case_id == "gamma"

    def test_empty_index_returns_empty(self):
        results = self.store.search(np.random.randn(384).astype(np.float32), top_k=5)
        assert results == []
