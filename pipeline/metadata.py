"""
pipeline/metadata.py
─────────────────────────────────────────────────────────────────────────────
Metadata extraction layer for Supreme Court of India judgments.

Extracts from document text:
  • case_number    — e.g. "W.P.(C) No. 123 of 2020"
  • case_id        — slug derived from case_number+year, or SHA256 fallback
  • case_type      — Civil / Criminal / Constitutional / etc.
  • year           — judgment year
  • judgment_date  — full date if parseable
  • petitioner     — first party name
  • respondent     — second party name
  • bench          — full bench description
  • judges         — list of individual judge names
  • outcome        — "Allowed" | "Dismissed" | "Disposed" | etc.
  • acts_cited     — list of Acts / sections referenced
  • citations      — AIR/SCC/SCR citation strings
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from dateutil import parser as dateutil_parser
from loguru import logger


# ──────────────────────────────────────────────────────────────────
#  Data model
# ──────────────────────────────────────────────────────────────────

@dataclass
class CaseMetadata:
    """All extracted metadata for one Supreme Court judgment."""
    # Identity
    case_id: str                            # Unique identifier for this case
    file_path: str
    file_name: str
    file_hash: str

    # Case identifiers
    case_number: Optional[str] = None       # Official case number
    case_type: Optional[str] = None         # Civil / Criminal / Constitutional / etc.
    year: Optional[int] = None

    # Dates
    judgment_date: Optional[date] = None
    filing_date: Optional[date] = None

    # Parties
    petitioner: Optional[str] = None
    respondent: Optional[str] = None

    # Bench
    bench: Optional[str] = None
    judges: list[str] = field(default_factory=list)

    # Legal references
    acts_cited: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    # Outcome
    outcome: Optional[str] = None          # Allowed / Dismissed / Disposed / etc.

    # Extraction quality
    extraction_confidence: float = 0.0     # 0.0–1.0 based on how many fields found


# ──────────────────────────────────────────────────────────────────
#  Regex patterns
# ──────────────────────────────────────────────────────────────────

# ── Case number patterns (SC India) ─────────────────────────────
# Covers: Civil/Criminal Appeal, Writ Petition, SLP, Transfer Petition, etc.
_CASE_NUMBER_PATTERNS: list[re.Pattern] = [
    # W.P.(C) No. 123 of 2020 / W.P.(CRL.) No. 45/2019
    re.compile(
        r"\bW\.?\s*P\.?\s*\(\s*[A-Za-z.]+\s*\)\s*No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Civil Appeal No. 1234 of 2020
    re.compile(
        r"\bCivil\s+Appeal\s+(?:Nos?\.?)?\s*(\d+(?:\s*[-–&]\s*\d+)?)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Criminal Appeal No. 456 of 2019
    re.compile(
        r"\bCriminal\s+Appeal\s+(?:Nos?\.?)?\s*(\d+(?:\s*[-–&]\s*\d+)?)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # SLP (Civil/Crl.) No. 789 of 2021
    re.compile(
        r"\bS\.?\s*L\.?\s*P\.?\s*\(\s*(?:Civil|Crl\.?|C|Criminal)\s*\)?\s*No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Transfer Petition No.
    re.compile(
        r"\bTransfer\s+Petition\s+\(\s*[A-Za-z.]+\s*\)\s*No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Contempt Petition
    re.compile(
        r"\bContempt\s+Petition\s+\(\s*[A-Za-z.]+\s*\)\s*No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Review Petition
    re.compile(
        r"\bReview\s+Petition\s+\(\s*[A-Za-z.]+\s*\)\s*No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
    # Original Suit / Arbitration
    re.compile(
        r"\b(?:Original\s+Suit|Arb\.?\s+Pet\.?)\s+No\.?\s*(\d+)\s*(?:OF|of|/)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
]

_CASE_NUMBER_RAW_RE = re.compile(
    r"\b(?:Civil\s+Appeal|Criminal\s+Appeal|Writ\s+Petition|W\.P\.|SLP|S\.L\.P\.|"
    r"Transfer\s+Petition|Contempt\s+Petition|Review\s+Petition|"
    r"Original\s+Suit)\s*\(?[A-Za-z.]*\)?\s*No\.?\s*\d+\s*(?:of|OF|/)\s*\d{4}",
    re.IGNORECASE,
)

# ── Case type classification ─────────────────────────────────────
_CASE_TYPE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bCivil\s+Appeal\b", re.IGNORECASE), "Civil Appeal"),
    (re.compile(r"\bCriminal\s+Appeal\b", re.IGNORECASE), "Criminal Appeal"),
    (re.compile(r"\bW\.?P\.?\s*\(\s*C\s*\)", re.IGNORECASE), "Writ Petition (Civil)"),
    (re.compile(r"\bW\.?P\.?\s*\(\s*Crl\.?\s*\)", re.IGNORECASE), "Writ Petition (Criminal)"),
    (re.compile(r"\bSLP\s*\(\s*C\s*\)|\bS\.L\.P\.\s*\(Civil\)", re.IGNORECASE), "SLP (Civil)"),
    (re.compile(r"\bSLP\s*\(\s*Crl\.?\s*\)|\bS\.L\.P\.\s*\(Criminal\)", re.IGNORECASE), "SLP (Criminal)"),
    (re.compile(r"\bTransfer\s+Petition\b", re.IGNORECASE), "Transfer Petition"),
    (re.compile(r"\bContempt\s+Petition\b", re.IGNORECASE), "Contempt Petition"),
    (re.compile(r"\bReview\s+Petition\b", re.IGNORECASE), "Review Petition"),
    (re.compile(r"\bConstitution\s+Bench\b", re.IGNORECASE), "Constitutional"),
    (re.compile(r"\bOriginal\s+Suit\b", re.IGNORECASE), "Original Suit"),
]

# ── Judge name patterns ──────────────────────────────────────────
# Matches: HON'BLE MR./MRS./MS. JUSTICE <Name>
_JUDGE_HONBLE_RE = re.compile(
    r"HON['\u2019]?BLE\s+(?:MR\.?|MRS\.?|MS\.?|DR\.?)\s+JUSTICE\s+([A-Z][A-Z\s.]+?)(?=\n|HON|$)",
    re.IGNORECASE,
)
# Matches: JUSTICE <Name> in CORAM section
_JUDGE_CORAM_RE = re.compile(
    r"CORAM\s*:\s*\n?((?:HON['\u2019]?BLE\s+)?(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*JUSTICE\s+[A-Z][A-Z\s.,]+(?:\n(?:HON['\u2019]?BLE\s+)?(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*JUSTICE\s+[A-Z][A-Z\s.,]+)*)",
    re.IGNORECASE,
)
_JUSTICE_NAME_RE = re.compile(
    r"JUSTICE\s+([A-Z][A-Z\s.]{2,40}?)(?=\n|,|AND\b|$|\bAND\b)",
    re.IGNORECASE,
)

# ── Parties ──────────────────────────────────────────────────────
_PETITIONER_RE = re.compile(
    r"^(.{3,150}?)\s*\.{3,}\s*(?:APPELLANT|PETITIONER|PLAINTIFF|COMPLAINANT)S?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_RESPONDENT_RE = re.compile(
    r"^(.{3,150}?)\s*\.{3,}\s*(?:RESPONDENT|DEFENDANT|ACCUSED)S?\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# ── Outcome patterns ─────────────────────────────────────────────
_OUTCOME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bappeal\s+is\s+(allowed|dismissed|disposed\s+of)\b", re.IGNORECASE), None),
    (re.compile(r"\bpetition\s+is\s+(allowed|dismissed|disposed\s+of)\b", re.IGNORECASE), None),
    (re.compile(r"\b(allowed|dismissed|disposed\s+of|upheld|quashed|set\s+aside|remanded)\s+with\s+costs?\b", re.IGNORECASE), None),
    (re.compile(r"\bWe\s+(?:hereby\s+)?(allow|dismiss|dispose\s+of)\b", re.IGNORECASE), None),
    (re.compile(r"\bConviction\s+(?:is|are)\s+(confirmed|set\s+aside|upheld|reversed)\b", re.IGNORECASE), None),
]

# ── Date patterns ────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"\b(?:DATED?|ORDER\s+DATED?|JUDGMENT\s+DATED?)[\s:]*"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

# ── Citation patterns ────────────────────────────────────────────
# AIR 2020 SC 1234 | (2020) 5 SCC 123 | 2020 SCR 100
_CITATION_RE = re.compile(
    r"\b(?:"
    r"AIR\s+\d{4}\s+SC\s+\d+"              # AIR 2020 SC 1234
    r"|\(\d{4}\)\s+\d+\s+SCC\s+\d+"        # (2020) 5 SCC 123
    r"|\d{4}\s+(?:SCR|SCC|AIR)\s+\d+"      # 2020 SCR 100
    r"|\[\d{4}\]\s+\d+\s+SCC\s+\d+"        # [2020] 1 SCC 100
    r"|\d{4}\s+SC\s+\d+"                   # 2020 SC 100
    r")",
    re.IGNORECASE,
)

# ── Acts & Statutes ──────────────────────────────────────────────
_ACTS = [
    # Criminal
    r"Indian\s+Penal\s+Code(?:\s*[\(,]\s*I\.?P\.?C\.?\s*[\))]?)?",
    r"\bI\.?P\.?C\.?\b",
    r"Code\s+of\s+Criminal\s+Procedure(?:\s*[\(,]\s*Cr\.?P\.?C\.?\s*[\))]?)?",
    r"\bCr\.?P\.?C\.\b",
    r"Prevention\s+of\s+Corruption\s+Act",
    r"NDPS\s+Act|Narcotic\s+Drugs\s+and\s+Psychotropic\s+Substances\s+Act",
    r"Protection\s+of\s+Children\s+from\s+Sexual\s+Offences\s+Act|POCSO",
    r"Scheduled\s+Castes\s+and\s+Scheduled\s+Tribes\s+(?:\(Prevention\s+of\s+Atrocities\))\s+Act|SC\/ST\s+Act",
    r"Unlawful\s+Activities\s+(?:\(Prevention\))\s+Act|UAPA",
    # Civil / Constitutional
    r"Constitution\s+of\s+India",
    r"Code\s+of\s+Civil\s+Procedure(?:\s*[\(,]\s*C\.?P\.?C\.?\s*[\))]?)?",
    r"\bC\.?P\.?C\.\b",
    r"Transfer\s+of\s+Property\s+Act",
    r"Specific\s+Relief\s+Act",
    r"Limitation\s+Act",
    r"Arbitration\s+and\s+Conciliation\s+Act",
    # Labour / Service
    r"Industrial\s+Disputes\s+Act",
    r"Employees'\s+Provident\s+Funds?\s+(?:and\s+Miscellaneous\s+Provisions\s+)?Act",
    r"Payment\s+of\s+Gratuity\s+Act",
    # Revenue / Tax
    r"Income[\s-]Tax\s+Act",
    r"Goods\s+and\s+Services\s+Tax\s+Act|GST\s+Act",
    r"Central\s+Excise\s+Act",
    r"Customs\s+Act",
    # Land
    r"Land\s+Acquisition\s+Act",
    r"Right\s+to\s+Fair\s+Compensation\s+and\s+Transparency\s+in\s+Land\s+Acquisition",
    # Family
    r"Hindu\s+Marriage\s+Act",
    r"Hindu\s+Succession\s+Act",
    r"Special\s+Marriage\s+Act",
    r"Guardians\s+and\s+Wards\s+Act",
    r"Muslim\s+Women\s+(?:\(Protection\s+of\s+Rights\s+on\s+(?:Divorce|Marriage)\))\s+Act",
    # Other notable
    r"Right\s+to\s+Information\s+Act|RTI\s+Act",
    r"Motor\s+Vehicles\s+Act",
    r"Electricity\s+Act",
    r"Insolvency\s+and\s+Bankruptcy\s+Code|IBC",
    r"Companies\s+Act",
    r"Consumer\s+Protection\s+Act",
    r"Environment(?:al)?\s+Protection\s+Act",
    r"Forest\s+Conservation\s+Act",
]
_ACTS_RE = re.compile(r"(?:" + "|".join(_ACTS) + r")", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────
#  Extraction helpers
# ──────────────────────────────────────────────────────────────────

def _extract_case_number(text: str) -> Optional[str]:
    """Return the first matching case number string."""
    match = _CASE_NUMBER_RAW_RE.search(text)
    if match:
        # Normalize whitespace in the matched string
        raw = match.group(0)
        return re.sub(r"\s+", " ", raw).strip()
    return None


def _build_case_id(case_number: Optional[str], year: Optional[int], file_hash: str) -> str:
    """
    Build a human-readable slug for case_id when possible.
    Falls back to first 16 chars of SHA-256 hash.

    Examples:
        "Civil Appeal 1234 of 2020" → "civil_appeal_1234_2020"
        None                        → "sha256abc123def456"
    """
    if case_number and year:
        slug = re.sub(r"[^a-z0-9]+", "_", case_number.lower())
        slug = slug.strip("_")
        if not slug.endswith(str(year)):
            slug = f"{slug}_{year}"
        return slug[:120]  # cap length
    # Fallback: deterministic hash-based ID
    return f"sha_{file_hash[:16]}"


def _classify_case_type(text: str, case_number: Optional[str]) -> Optional[str]:
    """Classify case type from text and case number."""
    search_text = (case_number or "") + "\n" + text[:2000]
    for pattern, label in _CASE_TYPE_MAP:
        if pattern.search(search_text):
            return label
    return None


def _extract_year(case_number: Optional[str], text: str) -> Optional[int]:
    """Extract most likely judgment year."""
    # Try case number first (most reliable)
    if case_number:
        m = _YEAR_RE.findall(case_number)
        if m:
            return int(m[-1])
    # Fall back to years in first 500 chars of text
    years = _YEAR_RE.findall(text[:500])
    if years:
        return int(years[0])
    return None


def _extract_judgment_date(text: str) -> Optional[date]:
    """Parse the judgment date from text."""
    match = _DATE_RE.search(text)
    if match:
        try:
            return dateutil_parser.parse(match.group(1), dayfirst=True).date()
        except Exception:
            pass
    return None


def _extract_judges(text: str) -> list[str]:
    """Extract individual judge names from CORAM / HON'BLE mentions."""
    judges: list[str] = []
    seen: set[str] = set()

    # Try CORAM block first
    coram_m = _JUDGE_CORAM_RE.search(text[:3000])
    if coram_m:
        for m in _JUSTICE_NAME_RE.finditer(coram_m.group(1)):
            name = re.sub(r"\s+", " ", m.group(1)).strip().title()
            if name and name not in seen:
                judges.append(name)
                seen.add(name)

    # If CORAM didn't work, try individual HON'BLE patterns
    if not judges:
        for m in _JUDGE_HONBLE_RE.finditer(text[:3000]):
            name = re.sub(r"\s+", " ", m.group(1)).strip().title()
            if name and name not in seen:
                judges.append(name)
                seen.add(name)

    return judges


def _extract_bench(text: str, judges: list[str]) -> Optional[str]:
    """Construct a bench string."""
    if judges:
        return " & ".join(judges)
    # Generic bench mentions
    m = re.search(r"(?:BENCH|CORAM)\s*:\s*(.{10,200}?)(?:\n\n|\Z)", text[:3000], re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def _extract_parties(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract petitioner and respondent names."""
    petitioner = respondent = None
    m = _PETITIONER_RE.search(text[:3000])
    if m:
        petitioner = re.sub(r"\s+", " ", m.group(1)).strip()

    m = _RESPONDENT_RE.search(text[:3000])
    if m:
        respondent = re.sub(r"\s+", " ", m.group(1)).strip()

    return petitioner, respondent


def _extract_outcome(text: str) -> Optional[str]:
    """Extract the case outcome (Allowed / Dismissed / etc.) from the last ~2000 chars."""
    tail = text[-2000:]
    for pattern, _ in _OUTCOME_PATTERNS:
        m = pattern.search(tail)
        if m:
            raw = m.group(1) if m.lastindex else m.group(0)
            return raw.strip().title()
    return None


def _extract_acts(text: str) -> list[str]:
    """Extract all Acts / statutes cited in the text."""
    matches = _ACTS_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for match in matches:
        normalized = re.sub(r"\s+", " ", match).strip()
        key = normalized.upper()
        if key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def _extract_citations(text: str) -> list[str]:
    """Extract AIR / SCC / SCR citation strings."""
    matches = _CITATION_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        norm = re.sub(r"\s+", " ", m).strip()
        if norm.upper() not in seen:
            result.append(norm)
            seen.add(norm.upper())
    return result


def _compute_confidence(meta: CaseMetadata) -> float:
    """
    Heuristic confidence score based on how many metadata fields were found.
    Weighted by importance.
    """
    scores = {
        "case_number": 0.25 if meta.case_number else 0.0,
        "case_type":   0.15 if meta.case_type else 0.0,
        "year":        0.10 if meta.year else 0.0,
        "judges":      0.20 if meta.judges else 0.0,
        "outcome":     0.15 if meta.outcome else 0.0,
        "parties":     0.10 if (meta.petitioner or meta.respondent) else 0.0,
        "citations":   0.05 if meta.citations else 0.0,
    }
    return round(sum(scores.values()), 2)


# ──────────────────────────────────────────────────────────────────
#  Main extractor class
# ──────────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extracts structured metadata from clean judgment text.

    Usage:
        extractor = MetadataExtractor()
        metadata = extractor.extract(clean_doc)
    """

    def extract(self, clean_text: str, file_path: str, file_name: str, file_hash: str) -> CaseMetadata:
        """
        Run full metadata extraction pipeline on a cleaned document.

        Args:
            clean_text: Cleaned text from TextCleaner.
            file_path:  Source PDF path.
            file_name:  PDF filename.
            file_hash:  SHA-256 of the PDF file.

        Returns:
            CaseMetadata with all extracted fields.
        """
        # ── Core identifiers ────────────────────────────────────
        case_number = _extract_case_number(clean_text)
        year = _extract_year(case_number, clean_text)
        case_id = _build_case_id(case_number, year, file_hash)
        case_type = _classify_case_type(clean_text, case_number)

        # ── Dates ────────────────────────────────────────────────
        judgment_date = _extract_judgment_date(clean_text)

        # ── People ───────────────────────────────────────────────
        judges = _extract_judges(clean_text)
        bench = _extract_bench(clean_text, judges)
        petitioner, respondent = _extract_parties(clean_text)

        # ── Legal references ─────────────────────────────────────
        acts_cited = _extract_acts(clean_text)
        citations = _extract_citations(clean_text)

        # ── Outcome ──────────────────────────────────────────────
        outcome = _extract_outcome(clean_text)

        meta = CaseMetadata(
            case_id=case_id,
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            case_number=case_number,
            case_type=case_type,
            year=year,
            judgment_date=judgment_date,
            petitioner=petitioner,
            respondent=respondent,
            bench=bench,
            judges=judges,
            acts_cited=acts_cited,
            citations=citations,
            outcome=outcome,
        )
        meta.extraction_confidence = _compute_confidence(meta)

        logger.debug(
            f"Metadata extracted | case_id={case_id} | type={case_type} | "
            f"year={year} | judges={len(judges)} | confidence={meta.extraction_confidence}"
        )
        return meta
