"""Validate revision filtering with a real SEC original/amendment pair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from corpcheck.retrieval.revision import (
    FilingKey,
    filter_superseded_rows,
    load_superseded_filings,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sec_real_amendment_gme_2024.json"


class AmendmentPool:
    """Return the fixture's amendment as if it came from ``filings``."""

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, args))
        grouping = self.fixture["corpcheck_grouping"]
        return [
            {
                "ticker": self.fixture["ticker"],
                "filing_type": self.fixture["amendment"]["filing_type"],
                "fiscal_year": grouping["fiscal_year"],
                "period": grouping["period"],
                "section_name": section,
            }
            for section in self.fixture["amendment"]["affected_sections"]
        ]


def _candidate(
    fixture: dict[str, Any], version: str, section_name: str, chunk_id: str
) -> dict[str, Any]:
    filing = fixture[version]
    grouping = fixture["corpcheck_grouping"]
    return {
        "chunk_id": chunk_id,
        "source_type": "sec",
        "company": fixture["ticker"],
        "filing_type": filing["filing_type"],
        "fiscal_year": grouping["fiscal_year"],
        "period_label": grouping["period"],
        "section_name": section_name,
        "filed_date": filing["filed_date"],
        "accession_number": filing["accession_number"],
        "source_url": filing["source_url"],
    }


@pytest.mark.asyncio
async def test_real_gamestop_amendment_composes_with_unchanged_original_sections() -> None:
    """GME's 10-K/A replaces Item 5 without hiding unchanged Item 8 evidence."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    amendment_section = fixture["amendment"]["affected_sections"][0]
    canonical_section = "Market for Common Equity"
    original_item_5 = _candidate(
        fixture, "original", canonical_section, "original-item-5"
    )
    original_item_8 = _candidate(
        fixture, "original", "Financial Statements", "original-item-8"
    )
    amendment_item_5 = _candidate(
        fixture, "amendment", amendment_section, "amendment-item-5"
    )
    pool = AmendmentPool(fixture)

    candidates = [original_item_5, original_item_8, amendment_item_5]
    superseded = await load_superseded_filings(pool, candidates)
    kept, dropped = filter_superseded_rows(candidates, superseded)

    grouping = fixture["corpcheck_grouping"]
    assert superseded == frozenset(
        {
            FilingKey(
                fixture["ticker"],
                fixture["original"]["filing_type"],
                grouping["fiscal_year"],
                grouping["period"].upper(),
                canonical_section.upper(),
            )
        }
    )
    assert [row["chunk_id"] for row in dropped] == ["original-item-5"]
    assert [row["chunk_id"] for row in kept] == [
        "original-item-8",
        "amendment-item-5",
    ]

    sql, args = pool.queries[0]
    assert "LEFT JOIN LATERAL" in sql
    assert "FROM chunks AS c" in sql
    assert "filing_type LIKE '%/A'" in sql
    assert args == ([fixture["ticker"]], [grouping["fiscal_year"]])
