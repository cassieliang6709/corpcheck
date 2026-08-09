"""Provenance enrichment and version governance for the MCP surface.

:class:`~corpcheck.models.ChunkResult` deliberately carries no accession number —
the retrieval view does not join ``filings``, and adding that join to the hot path
would cost every query for a field the ranking never reads. But an MCP client is a
different consumer: it hands evidence to an agent that may quote it, so each block
has to be traceable to a specific SEC document, not merely to a ticker and a year.

So the accession is attached here, in one batched lookup after ranking, keyed by
chunk id. Version governance (which amendment supersedes which filing) is *not*
reimplemented — :mod:`corpcheck.retrieval.revision` owns that, and this module
calls into it so the MCP path and the HTTP path can never disagree about which
filings are still authoritative.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import asyncpg

from corpcheck.models import ChunkResult
from corpcheck.retrieval.revision import (
    filing_key_for_row,
    is_amendment,
    is_superseded,
    load_superseded_filings,
)

logger = logging.getLogger(__name__)


async def load_filing_provenance(
    pool: asyncpg.Pool,
    chunks: Sequence[ChunkResult],
) -> dict[str, dict[str, Any]]:
    """Map ``chunk_id`` → filing-level provenance for the SEC chunks in *chunks*.

    Only SEC chunks are looked up: ``chunk_id`` is unique per source table rather
    than globally, so passing a news chunk's id into a ``chunks`` query could
    collide with an unrelated filing chunk. Non-SEC ids are simply absent from the
    returned map and the caller reports ``accession_number: null``.
    """
    sec_ids = [c.chunk_id for c in chunks if c.source_type == "sec"]
    numeric_ids = []
    for cid in sec_ids:
        try:
            numeric_ids.append(int(cid))
        except (TypeError, ValueError):
            logger.warning("Skipping non-numeric SEC chunk id %r", cid)
    if not numeric_ids:
        return {}

    rows = await pool.fetch(
        """
        SELECT
            c.id::text          AS chunk_id,
            c.chunk_index,
            c.section_name,
            f.id                AS filing_id,
            f.accession_number,
            f.cik,
            f.period,
            f.filed_date,
            f.period_of_report,
            f.source_url        AS filing_source_url
        FROM chunks c
        JOIN filings f ON f.id = c.filing_id
        WHERE c.id = ANY($1::bigint[])
        """,
        numeric_ids,
    )
    return {r["chunk_id"]: dict(r) for r in rows}


def evidence_block(
    chunk: ChunkResult,
    provenance: Optional[dict[str, Any]] = None,
    max_chars: Optional[int] = None,
) -> dict[str, Any]:
    """Render one retrieved chunk as an MCP evidence block.

    Provenance fields sit at the top level rather than in a nested object because
    the consumer is a language model: a flat record with unambiguous key names is
    quoted correctly more often than a nested one.
    """
    prov = provenance or {}
    text = chunk.text
    truncated = False
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
        truncated = True

    return {
        "chunk_id": chunk.chunk_id,
        "source_type": chunk.source_type,
        "company": chunk.company,
        "sector": chunk.sector,
        "filing_type": chunk.filing_type or None,
        "fiscal_year": chunk.fiscal_year,
        "period": chunk.period_label,
        "filed_date": chunk.filed_date.isoformat() if chunk.filed_date else None,
        "accession_number": prov.get("accession_number"),
        "cik": prov.get("cik"),
        "section": prov.get("section_name"),
        "chunk_index": prov.get("chunk_index"),
        "source_url": chunk.source_url or prov.get("filing_source_url"),
        "title": chunk.display_title or chunk.article_title,
        "content_kind": chunk.content_kind,
        "score": chunk.score,
        "cos_sim": chunk.cos_sim,
        "text": text,
        "text_truncated": truncated,
    }


async def superseded_check(
    pool: asyncpg.Pool,
    row: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """Whether a single filing row has been superseded by a later amendment.

    Used by ``get_filing_context``, which addresses a document directly and so
    never passes through :func:`retrieve`'s candidate-pool filter. Without this
    check the direct-lookup tool would become a hole in the governance the search
    tool enforces: an agent that saw a superseded accession elsewhere could still
    pull its text verbatim.
    """
    if is_amendment(row.get("filing_type")):
        return False, None

    key = filing_key_for_row(row)
    if key is None:
        return False, None

    superseded = await load_superseded_filings(pool, [row])
    if not is_superseded(key, superseded):
        return False, None

    scope = f" section {key.section!r}" if key.section else ""
    return True, (
        f"{key.ticker} {key.base_filing_type} FY{key.fiscal_year}"
        f"{'/' + key.period if key.period else ''}{scope} has been superseded by a "
        f"later amendment ({key.base_filing_type}/A). Its text is withheld: quoting a "
        "disclosure the filer has since changed is the most damaging error this "
        "system can make. Search for the amendment instead."
    )
