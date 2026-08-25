"""Adapter from FinanceBench rows to CorpCheck's provenance vocabulary.

中文：本模块只做基准数据与 CorpCheck 溯源字段之间的显式映射。映射不完整时
保持失败可见，避免把不确定的文档身份误当作命中。

FinanceBench identifies a source document as ``AMAZON_2019_10K`` plus the
structured fields ``company`` / ``doc_type`` / ``doc_period``. CorpCheck
identifies a filing as ticker + filing_type + fiscal_year (+ quarter). This
module translates between the two so the evaluation can check whether retrieved
evidence actually came from the document the question is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from corpcheck.ingestion.config import TICKER_TO_COMPANY_NAME

# No \b before the year: doc names separate fields with "_", which is itself a
# word character, so there is no word boundary at "JPMORGAN_2021Q1".
_QUARTER_IN_DOC_NAME_RE = re.compile(r"(?<!\d)20\d{2}(Q[1-4])(?!\d)", re.IGNORECASE)

_DOC_TYPE_TO_FILING_TYPE = {
    "10k": "10-K",
    "10q": "10-Q",
    "8k": "8-K",
}


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _build_name_index() -> dict[str, str]:
    """Map normalised company-name prefixes to tickers, from the project universe."""
    index: dict[str, str] = {}
    for ticker, name in TICKER_TO_COMPANY_NAME.items():
        normalized = _normalize_name(name)
        index[normalized] = ticker
        if normalized.startswith("the "):
            index[normalized.removeprefix("the ")] = ticker
    return index


_NAME_INDEX = _build_name_index()
_KNOWN_TICKERS = set(TICKER_TO_COMPANY_NAME)


def resolve_ticker(company: str) -> Optional[str]:
    """Resolve a FinanceBench ``company`` label to a project ticker.

    Tries the literal ticker first — FinanceBench writes some companies by their
    symbol ("AMD") — then falls back to prefix-matching the canonical name, which
    covers "Amazon" → "Amazon.com, Inc." and "Costco" → "Costco Wholesale
    Corporation".
    """
    if not company:
        return None

    upper = company.strip().upper()
    if upper in _KNOWN_TICKERS:
        return upper

    normalized = _normalize_name(company)
    if not normalized:
        return None

    exact = _NAME_INDEX.get(normalized)
    if exact:
        return exact

    matches = {
        ticker
        for name, ticker in _NAME_INDEX.items()
        if name.startswith(normalized) or normalized.startswith(name)
    }
    return next(iter(matches)) if len(matches) == 1 else None


@dataclass(frozen=True)
class GoldDoc:
    """The filing a FinanceBench question is answerable from."""

    ticker: Optional[str]
    filing_type: Optional[str]
    fiscal_year: Optional[int]
    quarter: Optional[str]
    doc_name: str

    @property
    def is_resolvable(self) -> bool:
        """True when we know enough to enforce a provenance check."""
        return (
            self.ticker is not None
            and self.filing_type is not None
            and self.fiscal_year is not None
        )


def parse_gold_doc(row: dict[str, Any]) -> GoldDoc:
    """Extract the gold source document from a FinanceBench row."""
    doc_name = str(row.get("doc_name") or "")

    doc_type = str(row.get("doc_type") or "").strip().lower()
    filing_type = _DOC_TYPE_TO_FILING_TYPE.get(doc_type)

    period = row.get("doc_period")
    try:
        fiscal_year = int(period) if period is not None else None
    except (TypeError, ValueError):
        fiscal_year = None

    quarter_match = _QUARTER_IN_DOC_NAME_RE.search(doc_name)
    quarter = quarter_match.group(1).upper() if quarter_match else None

    return GoldDoc(
        ticker=resolve_ticker(str(row.get("company") or "")),
        filing_type=filing_type,
        fiscal_year=fiscal_year,
        quarter=quarter,
        doc_name=doc_name,
    )


def chunk_matches_gold_doc(chunk: Any, gold: GoldDoc, *, match_quarter: bool = True) -> bool:
    """Whether a retrieved chunk is attributable to the gold filing.

    An auditor would not credit the right sentence pulled from the wrong
    document, so provenance is checked before content overlap is scored. When
    ``gold`` is not fully resolvable the check passes — an unknown gate should
    not silently suppress every chunk.
    """
    if not gold.is_resolvable:
        return True

    if (chunk.company or "").upper() != gold.ticker:
        return False
    if (chunk.filing_type or "") != gold.filing_type:
        return False
    if chunk.fiscal_year != gold.fiscal_year:
        return False

    if match_quarter and gold.quarter:
        if (chunk.period_label or "").strip().upper() != gold.quarter:
            return False

    return True


def gold_spans(row: dict[str, Any]) -> list[str]:
    """Return the gold evidence texts for a row, skipping empty entries."""
    evidence = row.get("evidence") or []
    spans = [str(e.get("evidence_text") or "").strip() for e in evidence]
    return [s for s in spans if s]
