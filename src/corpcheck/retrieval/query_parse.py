"""Query understanding: resolve companies, filing types, and fiscal years from free text.

The caches here are process-global and populated once at startup by
:func:`load_known_tickers`. They are read-only after that, so the retrieval path
can consult them without touching the database per request.

中文：启动时加载的全局缓存只读；解析阶段做保守的公司、文件类型和财年识别，不负责最终过滤或排序。
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

# Camel-cased ticker shorthand: "JnJ", "GoogL". Requires at least two capitals so
# a sentence-initial ordinary word ("Are", "Cost") cannot masquerade as a ticker;
# a lowercase-only token is never considered.
_MIXED_CASE_TICKER_RE = re.compile(r"\b([A-Za-z]{2,5})\b")

# A first word this long is distinctive enough to stand in for the whole company
# name ("Costco" for Costco Wholesale Corporation). Shorter ones -- "CVS", "3M" --
# are already covered by the ticker path, and admitting them here would let
# common words collide with issuers.
_MIN_HEAD_ALIAS_LEN = 5

# Years are matched with digit lookarounds rather than \b. "FY2019" has no word
# boundary between "Y" and "2" — both are word characters — so \b(20\d{2})\b
# silently misses the notation analysts actually use. The lookarounds still
# prevent matching inside a longer digit run such as an accession number.
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# "FY22" / "FY 22". Requires the FY prefix, so plain two-digit numbers elsewhere
# in the question are not mistaken for years. The (?!\d) keeps it from firing on
# the first two digits of a four-digit year.
_SHORT_FISCAL_YEAR_RE = re.compile(r"\bFY\s?(\d{2})(?!\d)", re.IGNORECASE)

# "Q2 2023" and the compact "Q22023" form that appears in filing shorthand.
_QUARTER_YEAR_RE = re.compile(r"\bQ[1-4]\s?(20\d{2})(?!\d)", re.IGNORECASE)

_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)(?:FY\s?)?(20\d{2})\s*(?:-|–|to|through|and)\s*(?:FY\s?)?(20\d{2})(?!\d)",
    re.IGNORECASE,
)
_SHORT_YEAR_RANGE_RE = re.compile(
    r"\bFY\s?(\d{2})\s*(?:-|–|to|through|and)\s*FY\s?(\d{2})(?!\d)",
    re.IGNORECASE,
)

# Two-digit fiscal years are read as 20xx. SEC electronic filing makes anything
# earlier irrelevant for this corpus, and modern financial writing ("FY22")
# universally means the 2000s.
_SHORT_YEAR_CENTURY = 2000
# Accepts two- and four-digit fiscal years alike, so "FY22" is recognised as an
# annual-period signal just as "FY2022" is.
_FISCAL_YEAR_RE = re.compile(
    r"\b(?:FY\s*\d{2,4}|fiscal\s+year\s*\d{2,4})\b", re.IGNORECASE
)
_ANNUAL_HINT_RE = re.compile(
    r"\b(?:annual|yearly|full.?year|year.?end)\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r"\b(?:q[1-4]|quarter(?:ly)?|first quarter|second quarter|third quarter|fourth quarter)\b",
    re.IGNORECASE,
)

# Unknown-issuer detection is intentionally narrower than company resolution.
# It only supports a possessive proper name in a question that explicitly asks
# about an SEC filing. That covers clear requests such as "OpenAI's SEC annual
# filing" without treating every capitalised word as a company.
_SEC_FILING_INTENT_RE = re.compile(
    r"\b(?:SEC|10-K|10-Q|annual\s+filing|quarterly\s+filing)\b", re.IGNORECASE
)
_POSSESSIVE_PROPER_NAME_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9]*(?:[.&-][A-Za-z0-9]+)*"
    r"(?:\s+(?:of|and|&|[A-Z][A-Za-z0-9]*(?:[.&-][A-Za-z0-9]+)*)){0,4}"
    r")[\u2019']s\b"
)
_GENERIC_POSSESSIVE_NAMES = {"company", "filing", "issuer", "management"}


def _normalize_company_text(text: str) -> str:
    """Canonicalize a company name for alias-map keys.

    中文：仅保留小写字母数字和单空格，使标点、大小写和空白差异不影响别名匹配。
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _generate_company_aliases(company_name: str) -> set[str]:
    """Derive conservative searchable aliases for one canonical company name.

    中文：生成完整名、去公司后缀和足够长的首词等别名；冲突别名会在缓存加载时被剔除。
    """
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

    # People say "Costco", not "Costco Wholesale Corporation". Register the head
    # word on its own when it is long enough to be a name rather than a common
    # word; `load_known_tickers` then discards it if two issuers claim it.
    if len(trimmed_tokens) >= 2 and len(trimmed_tokens[0]) >= _MIN_HEAD_ALIAS_LEN:
        add_alias([trimmed_tokens[0]])

    return aliases


async def load_known_tickers(pool: asyncpg.Pool) -> None:
    """Populate immutable ticker and unambiguous company-alias caches from the DB.

    中文：同一别名若对应多个股票代码则不注册，宁可不识别也不把查询静默指向错误公司。
    """
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
    """Resolve an explicit company filter to a canonical ticker when possible.

    中文：无法解析时保留原输入，让 SQL 层仍可尝试匹配公司全名而非丢弃用户过滤条件。
    """
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

    中文：优先严格股票代码，再尝试混合大小写缩写和唯一公司别名，以平衡召回和误判。
    """
    for match in _TICKER_RE.finditer(query):
        if match.group(1) in _known_tickers:
            return match.group(1)

    # Then camel-cased shorthand. "JnJ" is how people write JNJ, and requiring two
    # capitals keeps ordinary capitalised words out: "Are JnJ's FY2022 ..." yields
    # JNJ and not ARE.
    for match in _MIXED_CASE_TICKER_RE.finditer(query):
        token = match.group(1)
        if sum(ch.isupper() for ch in token) >= 2 and token.upper() in _known_tickers:
            return token.upper()

    normalized_query = _normalize_company_text(query)
    aliases = sorted(_company_alias_to_ticker.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, ticker in aliases:
        if re.search(r"\b" + re.escape(alias) + r"\b", normalized_query):
            return ticker
    return None


def detect_unresolved_company_in_query(query: str) -> Optional[str]:
    """Return a clear named issuer that is absent from the loaded corpus.

    ``detect_company_in_query()`` returning ``None`` is normally ambiguous: the
    question may not name a company at all. This deliberately conservative
    detector only distinguishes the unknown-company case when SEC-filing intent
    and possessive proper-name syntax occur together.

    中文：只有“明确 SEC 意图 + 所有格专名”才报未知公司，避免把普通大写词误判为发行人。
    """
    if detect_company_in_query(query) is not None:
        return None
    if _SEC_FILING_INTENT_RE.search(query) is None:
        return None

    for match in _POSSESSIVE_PROPER_NAME_RE.finditer(query):
        candidate = match.group(1).strip()
        if _normalize_company_text(candidate) not in _GENERIC_POSSESSIVE_NAMES:
            return candidate
    return None


def detect_filing_type_in_query(query: str) -> Optional[str]:
    """Return an explicit filing-type match when the query clearly names one type.

    Returns None for ambiguous queries that mention multiple filing types.

    中文：多个文件类型同时出现时不强行选择，调用方可继续进行不受类型限制的检索。
    """
    matches = {
        filing_type
        for filing_type, pattern in FILING_TYPE_PATTERNS.items()
        if pattern.search(query)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def detect_filing_type_hint_in_query(query: str) -> Optional[str]:
    """Infer a likely filing type from weak annual/quarter hints for boosting only.

    中文：弱信号只影响排序加权，绝不作为硬过滤条件，以免误删相关证据。
    """
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


def _expand_range(start: int, end: int, years: set[int]) -> None:
    """Add a bounded inclusive year range to ``years``.

    中文：超过十年的范围不展开，防止模糊文本把整个语料库都当作年份加权目标。
    """
    if start > end:
        start, end = end, start
    # Keep expansion bounded to avoid boosting a span that covers the corpus.
    if end - start <= 10:
        years.update(range(start, end + 1))


def detect_years_in_query(query: str) -> list[str]:
    """Return distinct 20xx years mentioned in the query.

    Handles the notations that appear in real analyst questions: bare ``2019``,
    ``FY2019``, the two-digit ``FY19``, quarterly ``Q2 2023`` / ``Q22023``, and
    ranges written either way (``FY2018 - FY2020``, ``FY20 to FY21``).

    中文：结果排序且去重，供过滤覆盖检查和 SQL boost 共用；它不是严格的自然语言时间解析器。
    """
    years: set[int] = set()

    for start_text, end_text in _YEAR_RANGE_RE.findall(query):
        _expand_range(int(start_text), int(end_text), years)

    for start_text, end_text in _SHORT_YEAR_RANGE_RE.findall(query):
        _expand_range(
            _SHORT_YEAR_CENTURY + int(start_text),
            _SHORT_YEAR_CENTURY + int(end_text),
            years,
        )

    for match in _YEAR_RE.findall(query):
        years.add(int(match))

    for match in _QUARTER_YEAR_RE.findall(query):
        years.add(int(match))

    for match in _SHORT_FISCAL_YEAR_RE.findall(query):
        years.add(_SHORT_YEAR_CENTURY + int(match))

    return [str(year) for year in sorted(years)]


def detect_year_in_query(query: str) -> Optional[str]:
    """Return the first detected year for backwards compatibility.

    中文：新代码应使用支持多年的 ``detect_years_in_query``；该包装器只维持旧接口。
    """
    years = detect_years_in_query(query)
    return years[0] if years else None


def sanitize_bm25_query(query: str, company: Optional[str]) -> str:
    """Drop explicit company mentions from BM25 when the SQL filter already scopes the corpus.

    This prevents queries like "AMZN revenue 2019 vs 2018" with company="AMZN"
    from producing an empty tsquery branch just because AMZN never appears in the
    chunk text. If cleanup removes everything, fall back to the original query.

    中文：公司已被 SQL 硬过滤时移除它的名称，避免词项未出现在正文而使稀疏检索分支为空。
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
    """Heuristically detect questions that benefit from numerical/table evidence.

    中文：这是候选过取和重排的软信号，不改变用户的过滤边界，也不保证问题一定是数值题。
    """
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
    """Heuristically detect questions that favor narrative explanatory evidence.

    中文：这是轻量重排信号，用于让风险、原因和策略类披露更容易进入结果集。
    """
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
