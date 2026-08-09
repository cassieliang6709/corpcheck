"""Revision-aware filtering: amended sections replace their original versions.

When a company files a ``10-K/A``, the amended sections of the original
``10-K`` are no longer authoritative — but unchanged sections often remain
available only in the original. Similarity search cannot determine that
composition boundary, so supersession is resolved from filing and section
metadata.

That is the single most damaging error this system can make: not a vague answer,
but a precise, well-cited, obsolete number. So supersession is resolved by
metadata, never by similarity.

Two deliberate properties:

* **Suppression is unconditional inside an amended section.** A matching
  original chunk is dropped even when no amendment chunk was retrieved to
  replace it. Unchanged original sections remain available.
* **Filtering happens before ranking.** Dropping candidates after top-k selection
  would silently return fewer than ``k`` results; dropping them from the
  candidate pool lets valid chunks move up and fill those slots.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional

import asyncpg

logger = logging.getLogger(__name__)

AMENDMENT_SUFFIX = "/A"

_10K_ITEM_SECTIONS = {
    "ITEM 1": "BUSINESS DESCRIPTION",
    "ITEM 1A": "RISK FACTORS",
    "ITEM 1B": "UNRESOLVED STAFF COMMENTS",
    "ITEM 2": "PROPERTIES",
    "ITEM 3": "LEGAL PROCEEDINGS",
    "ITEM 4": "MINE SAFETY DISCLOSURES",
    "ITEM 5": "MARKET FOR COMMON EQUITY",
    "ITEM 6": "SELECTED FINANCIAL DATA",
    "ITEM 7": "MD&A",
    "ITEM 7A": "QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
    "ITEM 8": "FINANCIAL STATEMENTS",
    "ITEM 9": "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS",
    "ITEM 9A": "CONTROLS AND PROCEDURES",
    "ITEM 9B": "OTHER INFORMATION",
    "ITEM 10": "DIRECTORS AND CORPORATE GOVERNANCE",
    "ITEM 11": "EXECUTIVE COMPENSATION",
    "ITEM 12": "SECURITY OWNERSHIP",
    "ITEM 13": "CERTAIN RELATIONSHIPS",
    "ITEM 14": "PRINCIPAL ACCOUNTANT FEES",
    "ITEM 15": "EXHIBITS",
}


class FilingKey(NamedTuple):
    """Identity of an amended filing section.

    ``period`` is ``None`` for an amendment that does not name the period it
    revises; such an amendment is treated as covering the whole fiscal year.
    ``section`` is ``None`` only when no amendment section metadata is
    available, and conservatively covers the whole filing period.
    """

    ticker: str
    base_filing_type: str
    fiscal_year: int
    period: Optional[str]
    section: Optional[str] = None


def is_amendment(filing_type: Optional[str]) -> bool:
    """True for amended filing types such as ``10-K/A`` or ``10-Q/A``."""
    return bool(filing_type) and filing_type.strip().upper().endswith(AMENDMENT_SUFFIX)


def base_filing_type(filing_type: Optional[str]) -> Optional[str]:
    """Strip the amendment suffix: ``10-K/A`` → ``10-K``. Non-amendments pass through."""
    if not filing_type:
        return None
    stripped = filing_type.strip()
    if is_amendment(stripped):
        return stripped[: -len(AMENDMENT_SUFFIX)].strip() or None
    return stripped


def _normalize_period(period: Optional[str]) -> Optional[str]:
    if period is None:
        return None
    cleaned = period.strip().upper()
    return cleaned or None


def _normalize_section(
    section: Optional[str], filing_type: Optional[str] = None
) -> Optional[str]:
    if section is None:
        return None
    cleaned = " ".join(section.split()).upper()
    if not cleaned:
        return None
    if cleaned in {"FULL DOCUMENT", "UNKNOWN"}:
        return None
    if (filing_type or "").upper() == "10-K":
        mapped = _10K_ITEM_SECTIONS.get(cleaned)
        if mapped is not None:
            return mapped
        if not cleaned.startswith("ITEM "):
            return cleaned
    if cleaned.startswith("ITEM "):
        # 10-Q repeats item numbers across Parts I and II. A bare item label is
        # not enough to identify its section, so preserve the safe wildcard.
        return None
    return cleaned or None


def filing_key_for_row(row: dict[str, Any]) -> Optional[FilingKey]:
    """Build the supersession key for a retrieved chunk row.

    Returns ``None`` for anything that cannot be superseded by an SEC amendment —
    news and transcript chunks, or filings missing the metadata to identify a
    period.
    """
    if (row.get("source_type") or "sec") != "sec":
        return None

    ticker = row.get("company") or row.get("ticker")
    filing_type = row.get("filing_type")
    fiscal_year = row.get("fiscal_year")

    if not ticker or not filing_type or fiscal_year is None:
        return None

    base = base_filing_type(filing_type)
    if base is None:
        return None

    return FilingKey(
        ticker=str(ticker).upper(),
        base_filing_type=base,
        fiscal_year=int(fiscal_year),
        period=_normalize_period(row.get("period_label")),
        section=_normalize_section(row.get("section_name"), base),
    )


def is_superseded(key: FilingKey, superseded: frozenset[FilingKey]) -> bool:
    """Whether an amendment exists that supersedes this filing period.

    An amendment recorded without a period acts as a wildcard over the whole
    fiscal year. An amendment with no usable section metadata acts as a wildcard
    over the whole filing period. If the candidate itself has no section, any
    matching amendment suppresses it because the system cannot prove that the
    requested text is outside the amendment's scope.
    """
    for amended in superseded:
        if (
            amended.ticker != key.ticker
            or amended.base_filing_type != key.base_filing_type
            or amended.fiscal_year != key.fiscal_year
        ):
            continue
        if amended.period is not None and amended.period != key.period:
            continue
        if key.section is None or amended.section is None or amended.section == key.section:
            return True
    return False


async def load_superseded_filings(
    pool: asyncpg.Pool,
    rows: list[dict[str, Any]],
) -> frozenset[FilingKey]:
    """Find which of the candidates' filing periods have since been amended.

    Scoped to the tickers and fiscal years actually present in the candidate set,
    so this stays a small indexed lookup rather than a scan. Queried per request
    rather than cached at startup: an offline backfill can land a new amendment
    at any time, and a stale cache here means serving superseded data — exactly
    the failure this module exists to prevent.
    """
    keys = [k for k in (filing_key_for_row(r) for r in rows) if k is not None]
    if not keys:
        return frozenset()

    tickers = sorted({k.ticker for k in keys})
    years = sorted({k.fiscal_year for k in keys})

    records = await pool.fetch(
        """
        SELECT DISTINCT
            f.ticker,
            f.filing_type,
            f.fiscal_year,
            f.period,
            amendment_sections.section_name
        FROM filings AS f
        LEFT JOIN LATERAL (
            SELECT DISTINCT
                UPPER(REGEXP_REPLACE(BTRIM(c.section_name), '\\s+', ' ', 'g')) AS section_name
            FROM chunks AS c
            WHERE c.filing_id = f.id
              AND NULLIF(BTRIM(c.section_name), '') IS NOT NULL
              AND UPPER(REGEXP_REPLACE(BTRIM(c.section_name), '\\s+', ' ', 'g'))
                  NOT IN ('FULL DOCUMENT', 'UNKNOWN')
        ) AS amendment_sections ON TRUE
        WHERE f.filing_type LIKE '%/A'
          AND f.ticker = ANY($1::text[])
          AND f.fiscal_year = ANY($2::int[])
        """,
        tickers,
        years,
    )

    superseded: set[FilingKey] = set()
    for record in records:
        base = base_filing_type(record["filing_type"])
        if base is None or record["fiscal_year"] is None:
            continue
        superseded.add(
            FilingKey(
                ticker=str(record["ticker"]).upper(),
                base_filing_type=base,
                fiscal_year=int(record["fiscal_year"]),
                period=_normalize_period(record["period"]),
                section=_normalize_section(record.get("section_name"), base),
            )
        )
    return frozenset(superseded)


def filter_superseded_rows(
    rows: list[dict[str, Any]],
    superseded: frozenset[FilingKey],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into (kept, dropped) by supersession.

    Chunks from the amendment itself are always kept. Original chunks are
    dropped only when their section matches an amended section, unless missing
    amendment metadata forced a whole-period wildcard.
    """
    if not superseded:
        return rows, []

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in rows:
        if is_amendment(row.get("filing_type")):
            kept.append(row)
            continue

        key = filing_key_for_row(row)
        if key is not None and is_superseded(key, superseded):
            dropped.append(row)
        else:
            kept.append(row)

    return kept, dropped


def log_dropped(dropped: list[dict[str, Any]]) -> None:
    """Record what was suppressed — provenance decisions must be auditable."""
    if not dropped:
        return
    periods = sorted(
        {
            f"{r.get('company')} {r.get('filing_type')} FY{r.get('fiscal_year')}"
            f"{'/' + str(r.get('period_label')) if r.get('period_label') else ''}"
            for r in dropped
        }
    )
    logger.info(
        "Revision filter suppressed %d chunk(s) from amended filing sections: %s",
        len(dropped),
        ", ".join(periods),
    )
