"""
Shared company metadata resolution helpers.

中文：把外部市场数据的宽松字段归一为项目需要的公司元数据，并在网络数据缺失时回退到
受版本控制的配置表。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yfinance as yf

from corpcheck.ingestion.config import TICKER_TO_COMPANY_NAME, TICKER_TO_SECTOR

UNKNOWN_SECTOR = "unknown"

_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "banking": (
        "bank",
        "banks",
        "capital markets",
        "credit",
        "financial services",
        "asset management",
        "insurance",
        "brokerage",
    ),
    "tech": (
        "technology",
        "software",
        "semiconductor",
        "internet",
        "communication equipment",
        "hardware",
        "electronics",
        "it services",
        "artificial intelligence",
    ),
    "healthcare": (
        "healthcare",
        "health care",
        "biotech",
        "biotechnology",
        "pharmaceutical",
        "pharma",
        "medical",
        "drug manufacturers",
        "diagnostics",
    ),
    "energy": (
        "energy",
        "oil",
        "gas",
        "petroleum",
        "midstream",
        "upstream",
        "downstream",
        "exploration",
        "refining",
        "drilling",
    ),
    "consumer": (
        "consumer",
        "retail",
        "restaurants",
        "apparel",
        "footwear",
        "household",
        "discount stores",
        "specialty retail",
        "home improvement",
        "packaged foods",
        "beverages",
    ),
}


def _clean_string(value: Any) -> str | None:
    """Return a stripped string or ``None`` when the value is empty-like.

    中文：统一处理供应商可能返回的 ``None``、空白或字符串化空值。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _is_unresolved_sector(value: str | None) -> bool:
    return value is None or value.strip().lower() in {"", UNKNOWN_SECTOR}


def _classify_sector(*values: str | None) -> str | None:
    """
    Map upstream sector/industry text into the project's coarse taxonomy.

    中文：上游行业标签不稳定，因此只映射到项目定义的粗粒度行业；未知文本保持未解析状态。
    """
    haystack = " ".join(v.lower() for v in values if v).strip()
    if not haystack:
        return None

    for canonical_sector, keywords in _SECTOR_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return canonical_sector
    return None


@lru_cache(maxsize=512)
def fetch_upstream_company_metadata(ticker: str) -> dict[str, Any]:
    """
    Fetch raw company metadata for *ticker* from yfinance.

    中文：网络或供应商错误被转换为空字典，让调用方能使用本地兜底数据继续执行。
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return dict(info)
    except Exception:
        return {}


def resolve_company_metadata(
    ticker: str,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve company metadata for *ticker* using upstream data first, with
    curated config values as fallback.

    中文：优先使用上游的公司名与描述，行业则标准化为项目枚举；不会修改传入的 ``info``。
    """
    upstream = info if info is not None else fetch_upstream_company_metadata(ticker)

    upstream_name = _clean_string(upstream.get("longName")) or _clean_string(upstream.get("shortName"))
    canonical_name = TICKER_TO_COMPANY_NAME.get(ticker)
    name = upstream_name or canonical_name or ticker

    upstream_sector = _clean_string(upstream.get("sector"))
    upstream_industry = _clean_string(upstream.get("industry"))
    normalized_sector = _classify_sector(upstream_sector, upstream_industry)
    sector = normalized_sector or TICKER_TO_SECTOR.get(ticker) or UNKNOWN_SECTOR

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "industry": upstream_industry,
        "market_cap": upstream.get("marketCap"),
        "description": _clean_string(upstream.get("longBusinessSummary")),
        "resolved_from_upstream": bool(upstream),
    }


def is_unresolved_company_name(name: str | None, ticker: str) -> bool:
    """Return whether *name* is missing or still ticker-like.

    中文：ticker 本身不是可展示的公司名，因此也视为未解析。
    """
    cleaned_name = _clean_string(name)
    if cleaned_name is None:
        return True
    return cleaned_name.upper() == ticker.upper()


def is_unresolved_sector(sector: str | None) -> bool:
    """Return whether *sector* is empty-like or unknown.

    中文：仅判断缺失状态，不尝试在此函数中重新分类。
    """
    return _is_unresolved_sector(_clean_string(sector))
