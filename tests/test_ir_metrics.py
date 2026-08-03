"""Unit tests for the IR evaluation metrics and the FinanceBench adapter.

These pin down the measuring instrument. If a metric silently changes meaning,
every A/B comparison built on it becomes uninterpretable, so the invariants here
matter more than usual.
"""

from __future__ import annotations

import pytest
from evaluation.financebench import (
    GoldDoc,
    chunk_matches_gold_doc,
    gold_spans,
    parse_gold_doc,
    resolve_ticker,
)
from evaluation.metrics import (
    content_tokens,
    first_hit_rank,
    hit_at_k,
    mean,
    normalize_text,
    recall_at_k,
    reciprocal_rank,
    strip_boilerplate,
    token_overlap,
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Net Income: $11,588 (in millions)!") == (
        "net income 11588 in millions"
    )


def test_normalize_keeps_financial_figures_as_single_tokens():
    # A split into "11" and "588" would destroy the most discriminative token
    # in a financial table.
    assert "11588" in normalize_text("Net income $ 11,588").split()
    assert "280522" in normalize_text("Total net sales 280,522").split()


def test_normalize_handles_empty_and_whitespace():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


def test_strip_boilerplate_removes_known_filing_furniture():
    text = normalize_text("Table of Contents AMAZON.COM, INC. Net income 11,588")
    stripped = strip_boilerplate(text)
    assert "table of contents" not in stripped
    assert "11588" in stripped
    assert "amazon" in stripped


def test_content_tokens_drops_stopwords_and_boilerplate():
    tokens = content_tokens("See accompanying notes. The revenue was 500 for the year.")
    assert "revenue" in tokens
    assert "500" in tokens
    assert "the" not in tokens
    assert "was" not in tokens


def test_content_tokens_can_keep_everything():
    tokens = content_tokens(
        "The revenue was 500", remove_boilerplate=False, remove_stopwords=False
    )
    assert {"the", "revenue", "was", "500"} <= tokens


# ---------------------------------------------------------------------------
# Token overlap
# ---------------------------------------------------------------------------


def test_token_overlap_is_one_for_exact_match():
    text = "Total net sales were 280,522 million in fiscal 2019"
    assert token_overlap(text, text) == pytest.approx(1.0)


def test_token_overlap_is_recall_oriented_on_gold():
    # Candidate contains all gold content plus a lot more; gold is fully covered,
    # so the candidate must not be penalised for its extra length.
    gold = "net income 11,588"
    candidate = "net income 11,588 and operating income 14,541 and interest expense 1,600"
    assert token_overlap(candidate, gold) == pytest.approx(1.0)


def test_token_overlap_is_partial_when_gold_only_half_covered():
    gold = "revenue 100 expenses 200"
    candidate = "revenue 100"
    assert token_overlap(candidate, gold) == pytest.approx(0.5)


def test_token_overlap_is_zero_for_disjoint_text():
    assert token_overlap("cash flow statement", "9,999 goodwill impairment") == 0.0


def test_token_overlap_is_zero_when_gold_is_only_boilerplate():
    # Nothing left to match against; must not divide by zero or return 1.0.
    assert token_overlap("anything at all", "Table of Contents") == 0.0


def test_boilerplate_stripping_reduces_spurious_overlap():
    gold = "Table of Contents See accompanying notes 11,588"
    unrelated = "Table of Contents See accompanying notes 42"
    assert token_overlap(unrelated, gold, remove_boilerplate=True) < token_overlap(
        unrelated, gold, remove_boilerplate=False, remove_stopwords=False
    )


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def test_first_hit_rank_is_one_indexed():
    assert first_hit_rank([0.1, 0.9, 0.4], threshold=0.5) == 2


def test_first_hit_rank_returns_none_when_nothing_clears_threshold():
    assert first_hit_rank([0.1, 0.2], threshold=0.5) is None


def test_first_hit_rank_includes_exact_threshold():
    assert first_hit_rank([0.5], threshold=0.5) == 1


def test_recall_at_k_counts_each_gold_span_separately():
    # Two of three gold spans found within the top 10.
    assert recall_at_k([1, 4, None], k=10) == pytest.approx(2 / 3)


def test_recall_at_k_respects_the_cutoff():
    assert recall_at_k([1, 8], k=5) == pytest.approx(0.5)
    assert recall_at_k([1, 8], k=10) == pytest.approx(1.0)


def test_recall_at_k_is_zero_with_no_gold_spans():
    assert recall_at_k([], k=10) == 0.0


def test_hit_at_k_is_binary():
    assert hit_at_k([None, 7], k=10) == 1.0
    assert hit_at_k([None, 7], k=5) == 0.0
    assert hit_at_k([None, None], k=10) == 0.0


def test_reciprocal_rank_uses_the_earliest_hit():
    assert reciprocal_rank([5, 2, None]) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_is_found():
    assert reciprocal_rank([None, None]) == 0.0


def test_reciprocal_rank_truncates_at_k():
    assert reciprocal_rank([12], k=10) == 0.0
    assert reciprocal_rank([12], k=20) == pytest.approx(1 / 12)


def test_mean_handles_empty():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# FinanceBench adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("company", "expected"),
    [
        ("AMD", "AMD"),          # already a ticker
        ("Amazon", "AMZN"),
        ("Costco", "COST"),
        ("JPMorgan", "JPM"),
        ("Johnson & Johnson", "JNJ"),
        ("Walmart", "WMT"),
        ("CVS Health", "CVS"),
    ],
)
def test_resolve_ticker(company, expected):
    assert resolve_ticker(company) == expected


def test_resolve_ticker_returns_none_for_unknown():
    assert resolve_ticker("Definitely Not A Real Company") is None
    assert resolve_ticker("") is None


def test_parse_gold_doc_from_annual_row():
    gold = parse_gold_doc(
        {"company": "Amazon", "doc_name": "AMAZON_2019_10K", "doc_type": "10k", "doc_period": 2019}
    )
    assert (gold.ticker, gold.filing_type, gold.fiscal_year, gold.quarter) == (
        "AMZN",
        "10-K",
        2019,
        None,
    )
    assert gold.is_resolvable


def test_parse_gold_doc_extracts_quarter_from_doc_name():
    gold = parse_gold_doc(
        {
            "company": "JPMorgan",
            "doc_name": "JPMORGAN_2021Q1_10Q",
            "doc_type": "10q",
            "doc_period": 2021,
        }
    )
    assert (gold.ticker, gold.filing_type, gold.fiscal_year, gold.quarter) == (
        "JPM",
        "10-Q",
        2021,
        "Q1",
    )


def test_parse_gold_doc_is_not_resolvable_for_unknown_company():
    gold = parse_gold_doc(
        {"company": "Unknown Co", "doc_name": "X_2020_10K", "doc_type": "10k", "doc_period": 2020}
    )
    assert not gold.is_resolvable


class _Chunk:
    """Minimal stand-in exposing the provenance fields the gate reads."""

    def __init__(self, company, filing_type, fiscal_year, period_label=None):
        self.company = company
        self.filing_type = filing_type
        self.fiscal_year = fiscal_year
        self.period_label = period_label


_AMZN_2019 = GoldDoc("AMZN", "10-K", 2019, None, "AMAZON_2019_10K")
_JPM_2021Q1 = GoldDoc("JPM", "10-Q", 2021, "Q1", "JPMORGAN_2021Q1_10Q")


def test_doc_gate_accepts_matching_filing():
    assert chunk_matches_gold_doc(_Chunk("AMZN", "10-K", 2019), _AMZN_2019)


@pytest.mark.parametrize(
    "chunk",
    [
        _Chunk("MSFT", "10-K", 2019),   # wrong company
        _Chunk("AMZN", "10-Q", 2019),   # wrong filing type
        _Chunk("AMZN", "10-K", 2018),   # wrong year — the catastrophic case
        _Chunk("AMZN", None, None),     # news/transcript chunk, no filing provenance
    ],
)
def test_doc_gate_rejects_wrong_provenance(chunk):
    assert not chunk_matches_gold_doc(chunk, _AMZN_2019)


def test_doc_gate_checks_quarter_for_quarterly_filings():
    assert chunk_matches_gold_doc(_Chunk("JPM", "10-Q", 2021, "Q1"), _JPM_2021Q1)
    assert not chunk_matches_gold_doc(_Chunk("JPM", "10-Q", 2021, "Q2"), _JPM_2021Q1)


def test_doc_gate_can_ignore_quarter():
    assert chunk_matches_gold_doc(
        _Chunk("JPM", "10-Q", 2021, "Q2"), _JPM_2021Q1, match_quarter=False
    )


def test_doc_gate_passes_through_when_gold_is_unresolvable():
    # An unknown gate must not silently suppress every chunk.
    unresolvable = GoldDoc(None, "10-K", 2019, None, "MYSTERY_2019_10K")
    assert chunk_matches_gold_doc(_Chunk("AMZN", "10-K", 2019), unresolvable)


def test_gold_spans_skips_empty_entries():
    row = {"evidence": [{"evidence_text": "real evidence"}, {"evidence_text": "  "}, {}]}
    assert gold_spans(row) == ["real evidence"]
