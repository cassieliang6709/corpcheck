"""Tests for query understanding, focused on fiscal-year notation.

The regression these pin down: ``\\b(20\\d{2})\\b`` cannot match "FY2019",
because "Y" and "2" are both word characters and there is no boundary between
them. That silently disabled the fiscal-year boost for the great majority of
professionally phrased questions, so the retriever returned the right company
from the wrong year.
"""

from __future__ import annotations

import pytest

from corpcheck.retrieval.query_parse import (
    detect_filing_type_hint_in_query,
    detect_year_in_query,
    detect_years_in_query,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # The regression: a year glued to an "FY" prefix.
        ("What was Amazon's FY2019 net income?", ["2019"]),
        ("fiscal year 2019 revenue", ["2019"]),
        # Two-digit fiscal-year shorthand.
        ("Does AMD have a healthy quick ratio for FY22?", ["2022"]),
        ("What drove revenue change as of the FY22 for AMD?", ["2022"]),
        ("FY 22 results", ["2022"]),
        # Bare years still work.
        ("revenue in 2019", ["2019"]),
        ("2019 net income", ["2019"]),
        # Quarterly notation, spaced and compact.
        ("Q2 2023 revenue by region", ["2023"]),
        ("which region had the biggest drop in Q22023 revenues", ["2023"]),
    ],
)
def test_detect_years_handles_real_notations(query, expected):
    assert detect_years_in_query(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("FY2018 - FY2020 average margin", ["2018", "2019", "2020"]),
        ("from 2018 to 2020", ["2018", "2019", "2020"]),
        ("Did Pfizer grow its PPNE between FY20 and FY21?", ["2020", "2021"]),
        ("between FY2023 and the FY2022 period", ["2022", "2023"]),
    ],
)
def test_detect_years_expands_ranges(query, expected):
    assert detect_years_in_query(query) == expected


def test_detect_years_ignores_implausibly_wide_ranges():
    # Expanding this would boost most of the corpus and mean nothing.
    assert detect_years_in_query("from 2001 to 2025") == ["2001", "2025"]


def test_detect_years_does_not_match_inside_longer_digit_runs():
    # Accession numbers and similar identifiers must not read as years.
    assert detect_years_in_query("accession 0000320193200019") == []
    assert detect_years_in_query("item 12020304") == []


def test_two_digit_rule_requires_the_fy_prefix():
    # A bare two-digit number is not a year.
    assert detect_years_in_query("the top 22 customers") == []
    assert detect_years_in_query("grew by 19 percent") == []


def test_four_digit_year_is_not_split_by_the_two_digit_rule():
    # "FY2022" must yield 2022, never 2020 from its leading digits.
    assert detect_years_in_query("FY2022 revenue") == ["2022"]


def test_detect_years_returns_empty_when_no_year_present():
    assert detect_years_in_query("What are the major products AMD sells?") == []


def test_detect_years_deduplicates_and_sorts():
    assert detect_years_in_query("FY2019 versus 2019 and FY19") == ["2019"]


def test_detect_year_in_query_returns_the_earliest():
    assert detect_year_in_query("compare FY2018 with FY2020") == "2018"
    assert detect_year_in_query("no year here") is None


def test_two_digit_fiscal_year_signals_an_annual_filing():
    # Previously only the four-digit form set the annual hint.
    assert detect_filing_type_hint_in_query("AMD results for FY22") == "10-K"
    assert detect_filing_type_hint_in_query("AMD results for FY2022") == "10-K"
