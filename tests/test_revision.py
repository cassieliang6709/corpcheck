"""Unit tests for revision-aware filtering (Task 2).

These tests exercise the pure filtering logic in ``revision.py`` without any
database dependency.  The module's single DB query is tested separately via
an integration test that inserts synthetic filings.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from corpcheck.retrieval.revision import (
    FilingKey,
    base_filing_type,
    filing_key_for_row,
    filter_superseded_rows,
    is_amendment,
    is_superseded,
    log_dropped,
)


# ── is_amendment ────────────────────────────────────────────────────────────

class TestIsAmendment:
    """Verify the /A suffix detector."""

    @pytest.mark.parametrize("ft", ["10-K/A", "10-Q/A", "8-K/A", " 10-K/A ", "10-k/a"])
    def test_positive(self, ft: str) -> None:
        assert is_amendment(ft) is True

    @pytest.mark.parametrize("ft", ["10-K", "10-Q", "8-K", ""])
    def test_negative(self, ft: str) -> None:
        assert is_amendment(ft) is False

    def test_none(self) -> None:
        assert is_amendment(None) is False  # type: ignore[arg-type]

    def test_whitespace_only(self) -> None:
        assert is_amendment("   ") is False


# ── base_filing_type ────────────────────────────────────────────────────────

class TestBaseFilingType:
    """Strip the /A suffix; pass through for non-amendments."""

    def test_strips_amendment_suffix(self) -> None:
        assert base_filing_type("10-K/A") == "10-K"
        assert base_filing_type("10-Q/A") == "10-Q"
        assert base_filing_type("8-K/A") == "8-K"

    def test_passthrough_for_non_amendment(self) -> None:
        assert base_filing_type("10-K") == "10-K"
        assert base_filing_type("10-Q") == "10-Q"

    def test_none(self) -> None:
        assert base_filing_type(None) is None  # type: ignore[arg-type]

    def test_whitespace_handling(self) -> None:
        assert base_filing_type("  10-K/A  ") == "10-K"

    def test_case_insensitive(self) -> None:
        assert base_filing_type("10-k/a") == "10-k"


# ── filing_key_for_row ──────────────────────────────────────────────────────

def _row(**overrides: Any) -> dict[str, Any]:
    """Build a minimal chunk row dict, overriding specific fields."""
    base: dict[str, Any] = {
        "source_type": "sec",
        "company": "AAPL",
        "ticker": None,
        "filing_type": "10-K",
        "fiscal_year": 2022,
        "period_label": "annual",
    }
    base.update(overrides)
    return base


class TestFilingKeyForRow:
    def test_basic(self) -> None:
        key = filing_key_for_row(_row())
        assert key == FilingKey("AAPL", "10-K", 2022, "ANNUAL")

    def test_amendment_row_uses_base_type(self) -> None:
        key = filing_key_for_row(_row(filing_type="10-K/A"))
        assert key == FilingKey("AAPL", "10-K", 2022, "ANNUAL")

    def test_quarterly(self) -> None:
        key = filing_key_for_row(_row(filing_type="10-Q", period_label="Q2"))
        assert key == FilingKey("AAPL", "10-Q", 2022, "Q2")

    def test_none_for_non_sec(self) -> None:
        assert filing_key_for_row(_row(source_type="news")) is None

    def test_none_for_missing_ticker(self) -> None:
        assert filing_key_for_row(_row(company=None, ticker=None)) is None

    def test_none_for_missing_filing_type(self) -> None:
        assert filing_key_for_row(_row(filing_type=None)) is None

    def test_none_for_missing_fiscal_year(self) -> None:
        assert filing_key_for_row(_row(fiscal_year=None)) is None

    def test_uses_ticker_field_when_company_is_missing(self) -> None:
        key = filing_key_for_row(_row(company=None, ticker="MSFT"))
        assert key is not None
        assert key.ticker == "MSFT"

    def test_period_none_when_label_absent(self) -> None:
        key = filing_key_for_row(_row(period_label=None))
        assert key is not None
        assert key.period is None


# ── is_superseded ───────────────────────────────────────────────────────────

class TestIsSuperseded:
    def test_exact_match(self) -> None:
        key = FilingKey("AAPL", "10-K", 2022, "ANNUAL")
        superseded = frozenset({key})
        assert is_superseded(key, superseded) is True

    def test_wildcard_period(self) -> None:
        """An amendment without a period supersedes any period in that FY."""
        key = FilingKey("AAPL", "10-Q", 2022, "Q2")
        wildcard = FilingKey("AAPL", "10-Q", 2022, None)
        superseded = frozenset({wildcard})
        assert is_superseded(key, superseded) is True

    def test_no_match(self) -> None:
        key = FilingKey("AAPL", "10-K", 2022, "ANNUAL")
        superseded = frozenset({FilingKey("MSFT", "10-K", 2022, "ANNUAL")})
        assert is_superseded(key, superseded) is False

    def test_different_year_no_match(self) -> None:
        key = FilingKey("AAPL", "10-K", 2022, "ANNUAL")
        superseded = frozenset({FilingKey("AAPL", "10-K", 2023, "ANNUAL")})
        assert is_superseded(key, superseded) is False

    def test_different_type_no_match(self) -> None:
        key = FilingKey("AAPL", "10-K", 2022, "ANNUAL")
        superseded = frozenset({FilingKey("AAPL", "10-Q", 2022, "Q1")})
        assert is_superseded(key, superseded) is False

    def test_empty_superseded_set(self) -> None:
        key = FilingKey("AAPL", "10-K", 2022, "ANNUAL")
        assert is_superseded(key, frozenset()) is False


# ── filter_superseded_rows ──────────────────────────────────────────────────

class TestFilterSupersededRows:
    def _build_rows(self) -> list[dict[str, Any]]:
        """Create a candidate set with both an original and its amendment."""
        return [
            # Original 10-K: should be dropped when amendment exists
            _row(company="AAPL", filing_type="10-K", fiscal_year=2022,
                 period_label="annual", chunk_id="c1"),
            # Amendment 10-K/A: should always survive
            _row(company="AAPL", filing_type="10-K/A", fiscal_year=2022,
                 period_label="annual", chunk_id="c2"),
            # Different company: should survive (not affected)
            _row(company="MSFT", filing_type="10-K", fiscal_year=2022,
                 period_label="annual", chunk_id="c3"),
            # News chunk (non-SEC): should survive
            _row(company="AAPL", source_type="news", filing_type=None,
                 fiscal_year=None, period_label=None, chunk_id="c4"),
        ]

    def test_drops_superseded_originals(self) -> None:
        rows = self._build_rows()
        superseded = frozenset({FilingKey("AAPL", "10-K", 2022, "ANNUAL")})
        kept, dropped = filter_superseded_rows(rows, superseded)
        dropped_ids = {r["chunk_id"] for r in dropped}
        assert "c1" in dropped_ids
        assert "c2" not in dropped_ids  # amendment survives
        assert "c3" not in dropped_ids  # different company
        assert "c4" not in dropped_ids  # non-SEC

    def test_amendments_always_kept(self) -> None:
        """Even if the amendment's base key is in the superseded set."""
        rows = self._build_rows()
        superseded = frozenset({FilingKey("AAPL", "10-K", 2022, "ANNUAL")})
        kept, _ = filter_superseded_rows(rows, superseded)
        kept_ids = {r["chunk_id"] for r in kept}
        assert "c2" in kept_ids

    def test_empty_superseded_is_noop(self) -> None:
        rows = self._build_rows()
        kept, dropped = filter_superseded_rows(rows, frozenset())
        assert len(kept) == len(rows)
        assert dropped == []

    def test_non_sec_chunks_pass_through(self) -> None:
        rows = [
            _row(company="AAPL", source_type="news", filing_type=None,
                 fiscal_year=None, period_label=None),
        ]
        superseded = frozenset({FilingKey("AAPL", "10-K", 2022, "ANNUAL")})
        kept, dropped = filter_superseded_rows(rows, superseded)
        assert len(kept) == 1
        assert len(dropped) == 0

    def test_wildcard_amendment_suppresses_all_quarters(self) -> None:
        """An amendment with period=None suppresses Q1, Q2, Q3."""
        rows = [
            _row(company="AAPL", filing_type="10-Q", fiscal_year=2022,
                 period_label="Q1", chunk_id="q1"),
            _row(company="AAPL", filing_type="10-Q", fiscal_year=2022,
                 period_label="Q2", chunk_id="q2"),
            _row(company="AAPL", filing_type="10-Q", fiscal_year=2022,
                 period_label="Q3", chunk_id="q3"),
        ]
        # Amendment without specific period → wildcard
        superseded = frozenset({FilingKey("AAPL", "10-Q", 2022, None)})
        kept, dropped = filter_superseded_rows(rows, superseded)
        assert len(dropped) == 3
        assert len(kept) == 0


# ── log_dropped ─────────────────────────────────────────────────────────────

class TestLogDropped:
    def test_logs_message(self, caplog: pytest.LogCaptureFixture) -> None:
        rows = [
            _row(company="AAPL", filing_type="10-K", fiscal_year=2022,
                 period_label="annual"),
        ]
        with caplog.at_level(logging.INFO, logger="corpcheck.retrieval.revision"):
            log_dropped(rows)
        assert "Revision filter suppressed 1 chunk(s)" in caplog.text
        assert "AAPL" in caplog.text

    def test_no_log_on_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="corpcheck.retrieval.revision"):
            log_dropped([])
        assert caplog.text == ""
