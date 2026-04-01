"""
pipeline/cleaner.py
─────────────────────────────────────────────────────────────────────────────
Text cleaning layer.

Takes raw extracted text from PDFs and returns clean, normalized text
suitable for metadata extraction, chunking, and embedding.

Cleaning steps:
  1. Null byte / control character removal
  2. Unicode normalization (NFC)
  3. Ligature and special character expansion
  4. Whitespace normalization (collapse runs, fix line breaks)
  5. Repeated punctuation removal
  6. Header/footer boilerplate detection and removal
  7. Page number artifacts removal
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────
#  Data model
# ──────────────────────────────────────────────────────────────────

@dataclass
class CleanDocument:
    """Cleaned, normalized text ready for downstream processing."""
    case_id_hint: str           # file_hash — used until real case_id is extracted
    file_path: str
    clean_text: str
    word_count: int
    char_count: int
    is_empty: bool              # True if after cleaning text is too short to be useful


# ──────────────────────────────────────────────────────────────────
#  Unicode / character normalization
# ──────────────────────────────────────────────────────────────────

# Map common ligatures and special chars found in legal text to ASCII equivalents
_LIGATURE_MAP: dict[str, str] = {
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb00": "ff",   # ﬀ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\u2018": "'",    # left single quotation mark
    "\u2019": "'",    # right single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u00ad": "",     # soft hyphen
    "\u200b": "",     # zero-width space
    "\u200c": "",     # zero-width non-joiner
    "\u200d": "",     # zero-width joiner
    "\ufeff": "",     # BOM
}

_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))


# ──────────────────────────────────────────────────────────────────
#  Boilerplate patterns (SC India specific)
# ──────────────────────────────────────────────────────────────────

# Patterns that appear on every page in Indian Kanoon PDFs
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # "Page X of Y" or "Page X"
    re.compile(r"\bPage\s+\d+\s+(?:of\s+\d+)?\b", re.IGNORECASE),
    # Page number alone on a line
    re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE),
    # "Indian Kanoon - http://..."
    re.compile(r"Indian\s+Kanoon\s*[-–]\s*http[s]?://\S+", re.IGNORECASE),
    # URL-only lines
    re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE),
    # "Reportable" / "Non-reportable" header
    re.compile(r"^\s*(NON[\s-])?REPORTABLE\s*$", re.MULTILINE | re.IGNORECASE),
    # "IN THE SUPREME COURT OF INDIA" repeated headers
    re.compile(
        r"IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA\s*\n"
        r"(?:CIVIL|CRIMINAL|ORIGINAL|APPELLATE|WRIT)?\s*"
        r"(?:APPELLATE\s+)?JURISDICTION",
        re.IGNORECASE,
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Core cleaning functions
# ──────────────────────────────────────────────────────────────────

def _remove_control_chars(text: str) -> str:
    """Remove null bytes and other non-printable control characters."""
    # Keep: \n (0x0a), \t (0x09), \r (0x0d) — strip the rest below 0x20
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize \r\n and lone \r to \n
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def _normalize_unicode(text: str) -> str:
    """Normalize to NFC and expand ligatures."""
    text = unicodedata.normalize("NFC", text)
    text = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group(0)], text)
    return text


def _remove_boilerplate(text: str) -> str:
    """Remove Indian Kanoon / SC-specific boilerplate."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Collapse multiple spaces/tabs on a single line → single space
    - Collapse 3+ consecutive newlines → double newline (paragraph break)
    - Strip leading/trailing whitespace from each line
    """
    # Collapse inline whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing spaces per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_punctuation(text: str) -> str:
    """Collapse repeated punctuation: '....' → '...', '----' → '-'"""
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"-{3,}", "--", text)
    text = re.sub(r"_{3,}", "__", text)
    text = re.sub(r"\*{3,}", "***", text)
    return text


def _fix_hyphenated_words(text: str) -> str:
    """
    Re-join words hyphenated across line breaks.
    Example: "appli-\ncation" → "application"
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# ──────────────────────────────────────────────────────────────────
#  Main cleaner class
# ──────────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Applies a sequential cleaning pipeline to raw extracted text.

    Usage:
        cleaner = TextCleaner()
        clean_doc = cleaner.clean(raw_doc)
    """

    MIN_WORDS = 50  # Documents with fewer words are considered empty/useless

    def clean(self, raw_text: str, file_path: str, file_hash: str) -> CleanDocument:
        """
        Run the full cleaning pipeline on raw text from one PDF.

        Args:
            raw_text:   Raw text from extractor.
            file_path:  Source file path (for provenance).
            file_hash:  SHA-256 of the PDF (used as case_id_hint).

        Returns:
            CleanDocument with normalized text and stats.
        """
        text = raw_text

        # Apply cleaning steps in order
        text = _remove_control_chars(text)
        text = _normalize_unicode(text)
        text = _remove_boilerplate(text)
        text = _fix_hyphenated_words(text)
        text = _remove_repeated_punctuation(text)
        text = _normalize_whitespace(text)

        word_count = len(text.split())
        is_empty = word_count < self.MIN_WORDS

        if is_empty:
            logger.warning(
                f"Document appears empty after cleaning: {file_path} "
                f"(only {word_count} words remaining)"
            )

        return CleanDocument(
            case_id_hint=file_hash,
            file_path=file_path,
            clean_text=text,
            word_count=word_count,
            char_count=len(text),
            is_empty=is_empty,
        )

    def clean_batch(self, raw_docs) -> list[CleanDocument]:
        """
        Clean a list of RawDocuments. Skips failed extractions.

        Args:
            raw_docs: Iterable of RawDocument objects.

        Returns:
            List of CleanDocument objects (only successful ones).
        """
        results: list[CleanDocument] = []
        for raw_doc in raw_docs:
            if not raw_doc.success or not raw_doc.raw_text:
                logger.debug(f"Skipping failed extraction: {raw_doc.file_name}")
                continue
            try:
                clean_doc = self.clean(
                    raw_doc.raw_text,
                    raw_doc.file_path,
                    raw_doc.file_hash,
                )
                results.append(clean_doc)
            except Exception as e:
                logger.error(f"Cleaning failed for {raw_doc.file_name}: {e}")
        return results
