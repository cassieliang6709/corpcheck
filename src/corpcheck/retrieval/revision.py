"""Revision-aware filtering: an amended filing supersedes the one it amends.

When a company files a ``10-K/A``, the original ``10-K`` is no longer the
authoritative record for that period — but the two documents are typically 99%
identical, so vector search cannot tell them apart. Left alone, the retriever
will happily serve the restated-away figure from the superseded original.

That is the single most damaging error this system can make: not a vague answer,
but a precise, well-cited, obsolete number. So supersession is resolved by
metadata, never by similarity.

Two deliberate properties:

* **Suppression is unconditional.** A superseded chunk is dropped even when no
  chunk from the amendment was retrieved to replace it. Losing evidence is
  recoverable — the answer degrades, and the abstain gate can decline. Serving a
  figure the filer has since corrected is not.
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


class FilingKey(NamedTuple):
    """Identity of a filing period, at the granularity SEC amendments apply to.

    ``period`` is ``None`` for an amendment that does not name the period it
    revises; such an amendment is treated as covering the whole fiscal year.
    """

    ticker: str
    base_filing_type: str
    fiscal_year: int
    period: Optional[str]


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
    )


def is_superseded(key: FilingKey, superseded: frozenset[FilingKey]) -> bool:
    """Whether an amendment exists that supersedes this filing period.

    An amendment recorded without a period acts as a wildcard over the whole
    fiscal year. That is the conservative reading: if we cannot tell which
    quarter was revised, we decline to serve any of them rather than guess and
    risk returning a restated figure.
    """
    if key in superseded:
        return True
    wildcard = FilingKey(key.ticker, key.base_filing_type, key.fiscal_year, None)
    return wildcard in superseded


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
        SELECT DISTINCT ticker, filing_type, fiscal_year, period
        FROM filings
        WHERE filing_type LIKE '%/A'
          AND ticker = ANY($1::text[])
          AND fiscal_year = ANY($2::int[])
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
            )
        )
    return frozenset(superseded)


def filter_superseded_rows(
    rows: list[dict[str, Any]],
    superseded: frozenset[FilingKey],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into (kept, dropped) by supersession.

    Chunks from the amendment itself are always kept — an amendment does not
    supersede itself, and its base-type key would otherwise match.
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
        "Revision filter suppressed %d chunk(s) from superseded filings: %s",
        len(dropped),
        ", ".join(periods),
    )
