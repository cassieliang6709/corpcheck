"""Unit tests for the MCP adapter layer.

These exercise the pure logic — evidence-block shaping, coverage summarisation,
gate classification, and the version-governance check on direct lookup — without
a database or an embedding model. The end-to-end stdio check that actually
launches the server against the live corpus is documented in the README; it needs
a populated Postgres and so is not part of the unit suite.
"""

from __future__ import annotations

from datetime import date

import pytest

from corpcheck.mcp.provenance import evidence_block, superseded_check
from corpcheck.mcp.server import _coverage, _gate_status
from corpcheck.models import ChunkResult


def chunk(**overrides) -> ChunkResult:
    base = dict(
        chunk_id="1",
        text="Total net sales increased 8% during 2022.",
        score=1.0,
        cos_sim=0.7,
        company="AAPL",
        sector="tech",
        filing_type="10-K",
        fiscal_year=2022,
        period_label="annual",
        filed_date=date(2022, 10, 28),
        source_url="https://example.invalid/aapl",
        source_type="sec",
    )
    base.update(overrides)
    return ChunkResult(**base)


# --------------------------------------------------------------------------
# evidence_block
# --------------------------------------------------------------------------


def test_evidence_block_carries_accession_from_provenance():
    block = evidence_block(
        chunk(),
        {"accession_number": "0000320193-22-000108", "cik": "0000320193",
         "section_name": "Selected Financial Data", "chunk_index": 56},
    )
    assert block["accession_number"] == "0000320193-22-000108"
    assert block["cik"] == "0000320193"
    assert block["section"] == "Selected Financial Data"
    assert block["chunk_index"] == 56


def test_evidence_block_without_provenance_reports_null_accession():
    # A missing accession must read as unknown, never as an invented identifier —
    # a fabricated citation is worse than an absent one.
    block = evidence_block(chunk(source_type="news"), None)
    assert block["accession_number"] is None
    assert block["source_type"] == "news"


def test_evidence_block_truncation_is_flagged():
    long_text = "x" * 5000
    block = evidence_block(chunk(text=long_text), None, max_chars=100)
    assert block["text_truncated"] is True
    assert len(block["text"]) < 200

    untruncated = evidence_block(chunk(text="short"), None, max_chars=100)
    assert untruncated["text_truncated"] is False
    assert untruncated["text"] == "short"


def test_evidence_block_serialises_filed_date_as_iso_string():
    assert evidence_block(chunk(), None)["filed_date"] == "2022-10-28"


# --------------------------------------------------------------------------
# coverage / gate classification
# --------------------------------------------------------------------------


def test_coverage_summarises_the_evidence_pool():
    cov = _coverage(
        [
            chunk(chunk_id="1", company="AAPL", fiscal_year=2022),
            chunk(chunk_id="2", company="MSFT", fiscal_year=2023, filing_type="10-Q"),
            chunk(chunk_id="3", company="AAPL", fiscal_year=2022, cos_sim=None),
        ]
    )
    assert cov["retrieved"] == 3
    assert cov["with_dense_score"] == 2
    assert cov["sparse_only"] == 1
    assert cov["companies"] == ["AAPL", "MSFT"]
    assert cov["filing_types"] == ["10-K", "10-Q"]
    assert cov["fiscal_years"] == [2022, 2023]


def test_gate_status_codes():
    assert _gate_status([]) == "no_results"
    assert _gate_status([chunk(cos_sim=None)]) == "sparse_only_unmeasurable"
    assert _gate_status([chunk(cos_sim=0.1)]) == "below_top1_floor"
    assert _gate_status([chunk(cos_sim=0.8)]) == "pass"


def test_gate_status_catches_lone_strong_hit_surrounded_by_noise():
    # One good match plus two weak ones: top-1 clears its floor but the mean of
    # the top three does not. This is the failure mode the second threshold exists
    # for, and it must not be reported as a pass.
    results = [
        chunk(chunk_id="1", cos_sim=0.90),
        chunk(chunk_id="2", cos_sim=0.10),
        chunk(chunk_id="3", cos_sim=0.10),
    ]
    assert _gate_status(results) == "below_mean_top3_floor"


# --------------------------------------------------------------------------
# version governance on direct lookup
# --------------------------------------------------------------------------


class FakePool:
    """Minimal asyncpg.Pool stand-in returning a fixed amendment table."""

    def __init__(self, records):
        self._records = records
        self.queries = []

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        return self._records


@pytest.mark.asyncio
async def test_direct_lookup_without_section_uses_conservative_wildcard():
    pool = FakePool(
        [{"ticker": "AAPL", "filing_type": "10-K/A", "fiscal_year": 2022, "period": "annual"}]
    )
    blocked, message = await superseded_check(
        pool,
        {
            "source_type": "sec",
            "company": "AAPL",
            "filing_type": "10-K",
            "fiscal_year": 2022,
            "period_label": "annual",
        },
    )
    assert blocked is True
    assert "superseded" in message


@pytest.mark.asyncio
async def test_direct_lookup_withholds_an_original_amended_section():
    pool = FakePool(
        [
            {
                "ticker": "GME",
                "filing_type": "10-K/A",
                "fiscal_year": 2024,
                "period": "annual",
                "section_name": "Market for Common Equity",
            }
        ]
    )
    blocked, message = await superseded_check(
        pool,
        {
            "source_type": "sec",
            "company": "GME",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "period_label": "annual",
            "section_name": "Market for Common Equity",
        },
    )

    assert blocked is True
    assert "section 'MARKET FOR COMMON EQUITY'" in message


@pytest.mark.asyncio
async def test_direct_lookup_allows_an_unamended_original_section():
    pool = FakePool(
        [
            {
                "ticker": "GME",
                "filing_type": "10-K/A",
                "fiscal_year": 2024,
                "period": "annual",
                "section_name": "Market for Common Equity",
            }
        ]
    )
    blocked, message = await superseded_check(
        pool,
        {
            "source_type": "sec",
            "company": "GME",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "period_label": "annual",
            "section_name": "Financial Statements",
        },
    )

    assert blocked is False
    assert message is None


@pytest.mark.asyncio
async def test_the_amendment_itself_is_never_withheld():
    pool = FakePool(
        [{"ticker": "AAPL", "filing_type": "10-K/A", "fiscal_year": 2022, "period": "annual"}]
    )
    blocked, message = await superseded_check(
        pool,
        {
            "source_type": "sec",
            "company": "AAPL",
            "filing_type": "10-K/A",
            "fiscal_year": 2022,
            "period_label": "annual",
        },
    )
    assert blocked is False
    assert message is None
    # Short-circuits before touching the database at all.
    assert pool.queries == []


@pytest.mark.asyncio
async def test_unamended_filing_passes_through():
    pool = FakePool([])
    blocked, _ = await superseded_check(
        pool,
        {
            "source_type": "sec",
            "company": "AAPL",
            "filing_type": "10-K",
            "fiscal_year": 2022,
            "period_label": "annual",
        },
    )
    assert blocked is False
