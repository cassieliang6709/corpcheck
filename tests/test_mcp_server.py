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

from corpcheck.mcp import server as mcp_server
from corpcheck.mcp.provenance import evidence_block, superseded_check
from corpcheck.mcp.server import _coverage, _gate_status, build_server
from corpcheck.models import ChunkResult
from corpcheck.retrieval.abstain import AbstainDecision


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


def test_gate_status_uses_metadata_decision_code_when_supplied():
    decision = AbstainDecision(
        True, "No matching year.", status="year_mismatch"
    )
    assert _gate_status([chunk(cos_sim=0.8)], decision) == "year_mismatch"


@pytest.mark.asyncio
async def test_mcp_tools_report_metadata_gate_status(monkeypatch):
    chunks = [chunk(company="COST", fiscal_year=2023, cos_sim=0.8)]

    async def fake_get_pool():
        return object()

    async def fake_retrieve(**kwargs):
        return chunks

    async def fake_provenance(pool, results):
        return {}

    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    monkeypatch.setattr(mcp_server, "retrieve", fake_retrieve)
    monkeypatch.setattr(mcp_server, "load_filing_provenance", fake_provenance)

    server = build_server()
    check = server._tool_manager._tools["check_answerable"].fn
    search = server._tool_manager._tools["search_filings"].fn
    query = "What were Costco's total assets in FY2099?"

    check_result = await check(query=query)
    search_result = await search(query=query)

    assert check_result["answerable"] is False
    assert check_result["gate_status"] == "year_mismatch"
    assert search_result["abstain"]["would_abstain"] is True
    assert search_result["abstain"]["status"] == "year_mismatch"


@pytest.mark.asyncio
async def test_mcp_company_filter_prevents_possessive_false_positive(monkeypatch):
    chunks = [chunk(company="AAPL", fiscal_year=2023, cos_sim=0.8)]

    async def fake_get_pool():
        return object()

    async def fake_retrieve(**kwargs):
        return chunks

    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    monkeypatch.setattr(mcp_server, "retrieve", fake_retrieve)

    server = build_server()
    check = server._tool_manager._tools["check_answerable"].fn
    result = await check(
        query="According to the SEC filing, what was CEO's compensation in FY2023?",
        company="AAPL",
    )

    assert result["answerable"] is True
    assert result["gate_status"] == "pass"


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


class ContextPool:
    """Route direct-context SQL to realistic filing, chunk, and amendment rows."""

    def __init__(self, anchor_section="Financial Statements"):
        self.anchor_section = anchor_section
        self.context_rows_requested = False

    async def fetchrow(self, sql, *args):
        if "FROM chunks c" in sql and "WHERE c.id = $1" in sql:
            return {
                "filing_id": 10,
                "chunk_index": 82 if self.anchor_section == "Financial Statements" else 50,
                "section_name": self.anchor_section,
                "ticker": "GME",
                "filing_type": "10-K",
                "fiscal_year": 2024,
                "period": "annual",
            }
        if "WHERE f.accession_number = $1" in sql:
            return {
                "filing_id": 10,
                "ticker": "GME",
                "filing_type": "10-K",
                "fiscal_year": 2024,
                "period": "annual",
                "section_name": "Financial Statements",
            }
        if "LEFT JOIN companies" in sql:
            return {
                "ticker": "GME",
                "company_name": "GameStop Corp.",
                "filing_type": "10-K",
                "fiscal_year": 2024,
                "period": "annual",
                "filed_date": None,
                "period_of_report": None,
                "accession_number": "0001326380-24-000012",
                "cik": "0001326380",
                "source_url": "https://example.invalid/gme-10k",
            }
        raise AssertionError(f"Unexpected fetchrow SQL: {sql}")

    async def fetch(self, sql, *args):
        if "SELECT DISTINCT" in sql and "filing_type LIKE '%/A'" in sql:
            return [
                {
                    "ticker": "GME",
                    "filing_type": "10-K/A",
                    "fiscal_year": 2024,
                    "period": "annual",
                    "section_name": "Item 5",
                }
            ]
        if "SELECT id::text AS chunk_id" in sql:
            self.context_rows_requested = True
            return [
                {
                    "chunk_id": "50",
                    "chunk_index": 50,
                    "section_name": "Market for Common Equity",
                    "content": "obsolete Item 5 text",
                },
                {
                    "chunk_id": "82",
                    "chunk_index": 82,
                    "section_name": "Financial Statements",
                    "content": "still authoritative Item 8 text",
                },
            ]
        raise AssertionError(f"Unexpected fetch SQL: {sql}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lookup",
    [
        {"chunk_id": "82", "window": 10},
        {"accession_number": "0001326380-24-000012", "max_chunks": 10},
    ],
)
async def test_context_lookup_filters_amended_neighbours_for_both_paths(
    monkeypatch, lookup
):
    pool = ContextPool()

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    context = build_server()._tool_manager._tools["get_filing_context"].fn

    result = await context(**lookup)

    assert [row["chunk_id"] for row in result["chunks"]] == ["82"]
    assert result["superseded_chunks_omitted"] == 1


@pytest.mark.asyncio
async def test_context_lookup_explicitly_withholds_a_superseded_anchor(monkeypatch):
    pool = ContextPool(anchor_section="Market for Common Equity")

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    context = build_server()._tool_manager._tools["get_filing_context"].fn

    result = await context(chunk_id="50", window=10)

    assert result["superseded"] is True
    assert result["text_withheld"] is True
    assert "MARKET FOR COMMON EQUITY" in result["reason"]


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
