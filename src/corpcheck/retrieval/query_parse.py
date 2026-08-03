"""Query understanding: resolve companies, filing types, and fiscal years from free text.

The caches here are process-global and populated once at startup by
:func:`load_known_tickers`. They are read-only after that, so the retrieval path
can consult them without touching the database per request.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import asyncpg

from corpcheck.retrieval.filing_type_patterns import FILING_TYPE_PATTERNS

logger = logging.getLogger(__name__)

_known_tickers: set[str] = set()
_company_name_to_ticker: dict[str, str] = {}
_company_alias_to_ticker: dict[str, str] = {}
_ticker_to_company_names: dict[str, set[str]] = {}
_ticker_to_company_aliases: dict[str, set[str]] = {}

_LEADING_COMPANY_TOKENS = {"the"}
_TRAILING_COMPANY_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "plc",
}
_DOMAIN_COMPANY_TOKENS = {"com", "net", "org"}

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*(?:-|–|to)\s*(20\d{2})\b", re.IGNORECASE)
_FISCAL_YEAR_RE = re.compile(r"\b(?:FY\s*20\d{2}|fiscal\s+year\s*20\d{2})\b", re.IGNORECASE)
_ANNUAL_HINT_RE = re.compile(
    r"\b(?:annual|yearly|full.?year|year.?end)\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r"\b(?:q[1-4]|quarter(?:ly)?|first quarter|second quarter|third quarter|fourth quarter)\b",
    re.IGNORECASE,
)


def _normalize_company_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _generate_company_aliases(company_name: str) -> set[str]:
    normalized = _normalize_company_text(company_name)
    if not normalized:
        return set()

    aliases: set[str] = set()

    def add_alias(tokens: list[str]) -> None:
        if not tokens:
            return
        alias = " ".join(tokens).strip()
        compact_alias = alias.replace(" ", "")
        if len(compact_alias) < 3:
            return
        aliases.add(alias)
        aliases.add(compact_alias)

    base_tokens = normalized.split()
    add_alias(base_tokens)

    tokens = base_tokens[:]
    while tokens and tokens[0] in _LEADING_COMPANY_TOKENS:
        tokens = tokens[1:]
        add_alias(tokens)

    trimmed_tokens = tokens[:] if tokens else base_tokens[:]
    while trimmed_tokens and trimmed_tokens[-1] in _TRAILING_COMPANY_TOKENS:
        trimmed_tokens = trimmed_tokens[:-1]
        add_alias(trimmed_tokens)

    if len(trimmed_tokens) >= 2 and trimmed_tokens[1] in _DOMAIN_COMPANY_TOKENS:
        add_alias([trimmed_tokens[0]])

    return aliases


async def load_known_tickers(pool: asyncpg.Pool) -> None:
    """Populate ticker and company-name caches from the database."""
    global _known_tickers, _company_name_to_ticker, _company_alias_to_ticker
    global _ticker_to_company_names, _ticker_to_company_aliases
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker, company_name FROM v_chunk_search WHERE ticker IS NOT NULL"
        )
    _known_tickers = {r["ticker"].upper() for r in rows}
    _company_name_to_ticker = {
        r["company_name"].lower(): r["ticker"].upper()
        for r in rows
        if r["company_name"]
    }
    _ticker_to_company_names = {}
    _ticker_to_company_aliases = {}
    alias_candidates: dict[str, set[str]] = {}
    for row in rows:
        company_name = row["company_name"]
        if not company_name:
            continue
        ticker = row["ticker"].upper()
        _ticker_to_company_names.setdefault(ticker, set()).add(company_name)
        aliases = _generate_company_aliases(company_name)
        if aliases:
            _ticker_to_company_aliases.setdefault(ticker, set()).update(aliases)
            for alias in aliases:
                alias_candidates.setdefault(alias, set()).add(ticker)

    _company_alias_to_ticker = {
        alias: next(iter(tickers))
        for alias, tickers in alias_candidates.items()
        if len(tickers) == 1
    }
    logger.info(
        "Loaded %d tickers and %d unambiguous company aliases",
        len(_known_tickers),
        len(_company_alias_to_ticker),
    )


def resolve_company_filter(company: Optional[str]) -> Optional[str]:
    """Resolve a company filter to its canonical ticker when possible."""
    if company is None:
        return None

    candidate = company.strip()
    if not candidate:
        return company

    maybe_ticker = candidate.upper()
    if maybe_ticker in _known_tickers:
        return maybe_ticker

    if candidate.lower() in _company_name_to_ticker:
        return _company_name_to_ticker[candidate.lower()]

    normalized = _normalize_company_text(candidate)
    return _company_alias_to_ticker.get(normalized, company)


def detect_company_in_query(query: str) -> Optional[str]:
    """Return a ticker matched from the query — by ticker symbol first, then full company name.

    Scans the original query (not uppercased) so only tokens already written in
    ALL-CAPS match — avoids false positives like 'are' → 'ARE' (a real ticker).
    """
    for match in _TICKER_RE.finditer(query):
        if match.group(1) in _known_tickers:
            return match.group(1)

    normalized_query = _normalize_company_text(query)
    aliases = sorted(_company_alias_to_ticker.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, ticker in aliases:
        if re.search(r"\b" + re.escape(alias) + r"\b", normalized_query):
            return ticker
    return None


def detect_filing_type_in_query(query: str) -> Optional[str]:
    """Return an explicit filing-type match when the query clearly names one type.

    Returns None for ambiguous queries that mention multiple filing types.
    """
    matches = {
        filing_type
        for filing_type, pattern in FILING_TYPE_PATTERNS.items()
        if pattern.search(query)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def detect_filing_type_hint_in_query(query: str) -> Optional[str]:
    """Infer a likely filing type from weaker annual/quarter signals for boosting only."""
    explicit_match = detect_filing_type_in_query(query)
    if explicit_match is not None:
        return explicit_match

    quarter_hint = bool(_QUARTER_RE.search(query))
    annual_hint = bool(_FISCAL_YEAR_RE.search(query) or _ANNUAL_HINT_RE.search(query))

    if quarter_hint and not annual_hint:
        return "10-Q"
    if annual_hint and not quarter_hint:
        return "10-K"
    return None


def detect_years_in_query(query: str) -> list[str]:
    """Return distinct 20xx years mentioned directly or via small ranges."""
    years: set[int] = set()

    for start_text, end_text in _YEAR_RANGE_RE.findall(query):
        start = int(start_text)
        end = int(end_text)
        if start > end:
            start, end = end, start
        # Keep expansion bounded to avoid boosting implausibly large spans.
        if end - start <= 10:
            years.update(range(start, end + 1))

    for match in _YEAR_RE.findall(query):
        years.add(int(match))

    return [str(year) for year in sorted(years)]


def detect_year_in_query(query: str) -> Optional[str]:
    """Return the first detected year for backwards compatibility."""
    years = detect_years_in_query(query)
    return years[0] if years else None


def sanitize_bm25_query(query: str, company: Optional[str]) -> str:
    """Drop explicit company mentions from BM25 when the SQL filter already scopes the corpus.

    This prevents queries like "AMZN revenue 2019 vs 2018" with company="AMZN"
    from producing an empty tsquery branch just because AMZN never appears in the
    chunk text. If cleanup removes everything, fall back to the original query.
    """
    if not company:
        return query

    aliases: set[str] = {company.strip()}
    company_key = company.strip().lower()
    ticker = _company_name_to_ticker.get(company_key)
    if ticker is None:
        maybe_ticker = company.strip().upper()
        if maybe_ticker in _known_tickers:
            ticker = maybe_ticker

    if ticker is not None:
        aliases.add(ticker)
        aliases.update(_ticker_to_company_names.get(ticker, set()))
        aliases.update(_ticker_to_company_aliases.get(ticker, set()))

    cleaned = query
    for alias in sorted((alias for alias in aliases if alias), key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(alias)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query


def query_prefers_quantitative_chunks(query: str) -> bool:
    lowered = query.lower()
    keywords = (
        "revenue",
        "income",
        "cash flow",
        "gross margin",
        "operating margin",
        "operating income",
        "eps",
        "earnings per share",
        "guidance",
        "grew",
        "growth",
        "decline",
        "decrease",
        "increase",
        "how much",
        "what was",
        "million",
        "billion",
        "%",
    )
    has_year = any(ch.isdigit() for ch in lowered)
    return has_year or any(keyword in lowered for keyword in keywords)


def query_prefers_explanatory_chunks(query: str) -> bool:
    lowered = query.lower()
    keywords = (
        "why",
        "risk",
        "strategy",
        "demand",
        "outlook",
        "guidance",
        "commentary",
        "management",
        "discussion",
        "explain",
        "because",
    )
    return any(keyword in lowered for keyword in keywords)
