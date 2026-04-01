"""
pipeline/extractor.py
─────────────────────────────────────────────────────────────────────────────
PDF text extraction layer.

Strategy (ordered by preference):
  1. PyMuPDF (fitz)  — fastest, handles most text-based PDFs
  2. pdfplumber      — better for complex column/table layouts
  3. pytesseract     — OCR fallback for image-only / scanned PDFs

Parallel processing via ProcessPoolExecutor for bulk ingestion of ~26K files.

Returns a RawDocument dataclass per PDF.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────
#  Data model
# ──────────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """Holds the raw extracted text and provenance info for one PDF."""
    file_path: str
    file_name: str
    raw_text: str
    page_count: int
    extraction_method: str          # "pymupdf" | "pdfplumber" | "tesseract" | "failed"
    file_hash: str                  # SHA-256 of raw file bytes — used as fallback case_id
    extraction_error: Optional[str] = None
    success: bool = True


# ──────────────────────────────────────────────────────────────────
#  Single-file extraction helpers
# ──────────────────────────────────────────────────────────────────

def _extract_with_pymupdf(path: Path) -> tuple[str, int]:
    """Extract text using PyMuPDF (fitz). Returns (text, page_count)."""
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    doc = fitz.open(str(path))
    page_count = len(doc)

    for page in doc:
        text_parts.append(page.get_text("text"))  # type: ignore[arg-type]

    doc.close()
    return "\n".join(text_parts), page_count


def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    """Extract text using pdfplumber. Returns (text, page_count)."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)

    return "\n".join(text_parts), page_count


def _extract_with_tesseract(path: Path) -> tuple[str, int]:
    """
    OCR-based extraction using pytesseract.
    Converts each PDF page to an image and runs OCR.
    Slowest method — only used as last resort.
    """
    import fitz
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(str(path))
    page_count = len(doc)
    text_parts: list[str] = []

    for page in doc:
        # Render page at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(img, lang="eng")
        text_parts.append(ocr_text)

    doc.close()
    return "\n".join(text_parts), page_count


def _compute_file_hash(path: Path) -> str:
    """SHA-256 hash of raw file bytes — stable unique identifier."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _is_text_sufficient(text: str, min_chars: int = 200) -> bool:
    """Check if extracted text is meaningful (not just whitespace / noise)."""
    clean = "".join(text.split())
    return len(clean) >= min_chars


# ──────────────────────────────────────────────────────────────────
#  Main single-file extraction function (runs in worker process)
# ──────────────────────────────────────────────────────────────────

def extract_single_pdf(pdf_path_str: str) -> RawDocument:
    """
    Extract text from a single PDF using a cascade of strategies.
    Designed to be executed in a worker process (picklable function).

    Args:
        pdf_path_str: Absolute path to the PDF file as string.

    Returns:
        RawDocument with extracted text and metadata.
    """
    path = Path(pdf_path_str)
    file_hash = _compute_file_hash(path)

    # ── Strategy 1: PyMuPDF ──────────────────────────────────────
    try:
        text, pages = _extract_with_pymupdf(path)
        if _is_text_sufficient(text):
            return RawDocument(
                file_path=str(path),
                file_name=path.name,
                raw_text=text,
                page_count=pages,
                extraction_method="pymupdf",
                file_hash=file_hash,
            )
        logger.debug(f"PyMuPDF text insufficient for {path.name}, trying pdfplumber")
    except Exception as e:
        logger.debug(f"PyMuPDF failed for {path.name}: {e}")

    # ── Strategy 2: pdfplumber ───────────────────────────────────
    try:
        text, pages = _extract_with_pdfplumber(path)
        if _is_text_sufficient(text):
            return RawDocument(
                file_path=str(path),
                file_name=path.name,
                raw_text=text,
                page_count=pages,
                extraction_method="pdfplumber",
                file_hash=file_hash,
            )
        logger.debug(f"pdfplumber text insufficient for {path.name}, trying OCR")
    except Exception as e:
        logger.debug(f"pdfplumber failed for {path.name}: {e}")

    # ── Strategy 3: Tesseract OCR ────────────────────────────────
    try:
        text, pages = _extract_with_tesseract(path)
        if _is_text_sufficient(text):
            return RawDocument(
                file_path=str(path),
                file_name=path.name,
                raw_text=text,
                page_count=pages,
                extraction_method="tesseract",
                file_hash=file_hash,
            )
    except Exception as e:
        logger.debug(f"Tesseract OCR failed for {path.name}: {e}")

    # ── All strategies failed ────────────────────────────────────
    return RawDocument(
        file_path=str(path),
        file_name=path.name,
        raw_text="",
        page_count=0,
        extraction_method="failed",
        file_hash=file_hash,
        extraction_error="All extraction strategies failed",
        success=False,
    )


# ──────────────────────────────────────────────────────────────────
#  Bulk extraction with parallel workers
# ──────────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Bulk PDF extractor with parallel processing support.

    Usage:
        extractor = PDFExtractor(max_workers=4)
        for doc in extractor.extract_many(pdf_paths):
            process(doc)
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def extract_one(self, pdf_path: Path | str) -> RawDocument:
        """Extract a single PDF (in the calling process — no parallelism)."""
        return extract_single_pdf(str(pdf_path))

    def extract_many(
        self,
        pdf_paths: list[Path | str],
        show_progress: bool = True,
    ):
        """
        Extract text from multiple PDFs using a process pool.

        Yields RawDocument objects as they complete (unordered).
        Failures are yielded as RawDocument(success=False) rather
        than raising exceptions, so the pipeline can continue.
        """
        str_paths = [str(p) for p in pdf_paths]
        total = len(str_paths)

        logger.info(f"Starting bulk extraction of {total} PDFs with {self.max_workers} workers")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(extract_single_pdf, p): p
                for p in str_paths
            }

            iterator = as_completed(future_to_path)
            if show_progress:
                iterator = tqdm(iterator, total=total, desc="Extracting PDFs", unit="pdf")

            for future in iterator:
                pdf_path = future_to_path[future]
                try:
                    result = future.result()
                    yield result
                except Exception as exc:
                    logger.error(f"Unexpected error processing {pdf_path}: {exc}")
                    file_hash = _compute_file_hash(Path(pdf_path))
                    yield RawDocument(
                        file_path=pdf_path,
                        file_name=Path(pdf_path).name,
                        raw_text="",
                        page_count=0,
                        extraction_method="failed",
                        file_hash=file_hash,
                        extraction_error=traceback.format_exc(),
                        success=False,
                    )

        logger.info("Bulk extraction complete")

    @staticmethod
    def discover_pdfs(directory: Path | str) -> list[Path]:
        """
        Recursively discover all PDF files in a directory.

        Returns a sorted list of absolute Paths.
        """
        directory = Path(directory)
        pdfs = sorted(directory.rglob("*.pdf"))
        logger.info(f"Discovered {len(pdfs)} PDF files in {directory}")
        return pdfs
