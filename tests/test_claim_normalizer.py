from datetime import datetime
from decimal import Decimal

from corpcheck.claims.normalizer import (
    extract_numeric_bindings,
    normalize_claims,
    normalize_text,
    split_atomic_claims,
)
from corpcheck.claims.schema import ClaimEvaluationRequest


def test_normalize_text_canonicalizes_quotes_and_spaces():
    raw = '  NVIDIA\'s   "revenue" is  up 2024  '
    assert normalize_text(raw) == 'NVIDIA\'s "revenue" is up 2024'


def test_split_atomic_claims_basic():
    text = 'Revenue rose to $10.2B. We expect growth next quarter, and margin remains flat.'
    claims = split_atomic_claims(text)
    assert claims[0].startswith('Revenue rose')
    assert any('expect growth' in c.lower() for c in claims)


def test_normalize_claims_generates_ids_and_type():
    req = ClaimEvaluationRequest(
        source_text=(
            'NVIDIA reported revenue of $11.6 billion in 2025. '
            'They expect supply constraints to ease next quarter.'
        ),
        company_name="NVIDIA",
        as_of=datetime.fromisoformat("2026-08-20T00:00:00+00:00"),
    )
    claims = normalize_claims(req)

    assert len(claims) == 2
    assert claims[0].claim_id.startswith("claim_")
    assert claims[0].claim_type == "reported_numeric"
    assert claims[1].claim_type == "prediction"
    assert claims[1].checkability == "watch_later"
    assert claims[0].value is not None
    assert claims[0].unit in {"million", "billion"}


def test_extract_numeric_bindings_normalizes_scales():
    values = extract_numeric_bindings("Revenue was $11.6 billion and 11,600 million next")
    assert len(values) >= 2
    assert values[0][1] == "billion"
    assert values[1][1] == "million"

    # Both should map to the same absolute scale for deterministic comparison.
    assert values[0][0] == values[1][0]


def test_extract_numeric_bindings_supports_percent_and_chinese_units():
    values = extract_numeric_bindings("利润增长了 15%，净资产 5万 美元")
    units = [unit for _, unit, _ in values]
    assert "%" in units
    assert "万" in units
    # numeric normalization keeps 百分比 values as decimals with explicit unit.
    assert len(values) >= 2


def test_extract_numeric_bindings_drops_tokenizer_split_decimal_tails():
    """Some stored chunks arrive tokenized as "$ 98. 0 billion".

    The fragment after the period must not be read as a standalone
    "0 billion", or it shows up as counter-evidence in a verdict receipt.
    """
    values = extract_numeric_bindings("the fair value was $ 98. 0 billion in notes")
    assert Decimal("0") not in [value for value, _unit, _raw in values]


def test_extract_numeric_bindings_keeps_a_figure_after_a_sentence_ending_in_a_digit():
    values = extract_numeric_bindings("Sales grew during 2023. $11.0 billion matured.")
    assert Decimal("11000000000.0") in [value for value, _unit, _raw in values]
