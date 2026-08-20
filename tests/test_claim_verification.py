from datetime import date, datetime
from decimal import Decimal

from corpcheck.api import main
from corpcheck.models import ClaimVerificationRequest


def _chunk(**overrides):
    base = {
        "chunk_id": "sec-0001",
        "text": "Total net sales were $11,600 million in 2026.",
        "score": 1.0,
        "cos_sim": 0.91,
        "company": "NVIDIA",
        "sector": None,
        "filing_type": "10-K",
        "fiscal_year": 2026,
        "period_label": "FY",
        # Must predate the claims' as_of, otherwise the point-in-time cutoff
        # correctly refuses to use it as evidence.
        "filed_date": date(2026, 2, 26),
        "source_url": "https://www.sec.gov/Archives/example.txt",
        "article_title": "NVIDIA 2026 10-K",
        "page_num": None,
        "source_type": "sec",
        "content_kind": "narrative",
        "chunk_strategy": "sentence",
        "display_title": "NVIDIA 2026 Annual Report",
    }
    base.update(overrides)
    return base


async def test_claim_verify_verdict_verified_with_scaled_numeric_match(monkeypatch):
    calls = {}

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        calls.update(kwargs)
        return [_chunk()]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $11.6 billion in 2026.",
            company_name="NVIDIA",
        )
    )

    assert response.total_claims == 1
    assert response.verdicts[0].verdict == "verified"
    assert response.verdicts[0].reason_code == "evidence_binds_metric_period_and_value"
    assert len(response.verdicts[0].evidence_for) == 1
    assert calls["query"] == "NVIDIA reported revenue of $11.6 billion in 2026."


async def test_claim_verify_marks_predictions_not_yet_decidable(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return []

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA will report revenue above $60 million next quarter.",
            company_name="NVIDIA",
        )
    )

    assert response.total_claims == 1
    assert response.verdicts[0].verdict == "not_yet_decidable"
    assert response.verdicts[0].reason_code == "prediction_requires_future_evidence"


async def test_claim_verify_refutes_when_bound_evidence_contradicts_value(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text="Total net sales were $9.1 billion in 2026.",
            )
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $11.6 billion in 2026.",
            company_name="NVIDIA",
        )
    )

    assert response.verdicts[0].verdict == "refuted"
    assert response.verdicts[0].reason_code == (
        "evidence_binds_metric_and_period_but_value_differs"
    )
    assert len(response.verdicts[0].evidence_against) == 1


async def test_claim_verify_does_not_bind_a_number_from_a_different_metric(monkeypatch):
    """A dividend figure must not verify a claim about share repurchases.

    Both numbers live in the same sentence, so any "is this value present in
    the chunk" check reports ``verified`` here. Binding each number to its
    nearest metric mention is what makes the contradiction visible instead.
    """

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text=(
                    "The Company repurchased $19.1 billion of its common stock "
                    "and paid dividends and dividend equivalents of $3.7 billion "
                    "during the period."
                ),
            )
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA repurchased $3.7 billion of its common stock in 2026.",
            company_name="NVIDIA",
        )
    )

    verdict = response.verdicts[0]
    assert verdict.verdict == "refuted"
    assert not verdict.evidence_for
    # The bound counter-evidence is the repurchase figure, not the dividend one.
    assert verdict.evidence_against[0].value == Decimal("19100000000")


async def test_claim_verify_ignores_change_amounts_when_scoring_a_level(monkeypatch):
    """A reported decrease must not refute a claim about the resulting level.

    The 10-K sentence carries both the level ($11.6bn) and the movement
    ($1.1bn). Scoring the movement against the claim produced a false
    refutation of an accurate claim.
    """

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text=(
                    "Total net sales were $11.6 billion and net income was "
                    "$3.4 billion during 2026. Total net sales decreased 3% or "
                    "$1.1 billion during 2026."
                ),
            )
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $11.6 billion in 2026.",
            company_name="NVIDIA",
        )
    )

    verdict = response.verdicts[0]
    assert verdict.verdict == "verified"
    # The neighbouring net income figure must not be bound to revenue either.
    assert [item.value for item in verdict.evidence_for] == [Decimal("11600000000")]
    assert not verdict.evidence_against


async def test_claim_verify_tolerates_the_precision_the_claim_was_written_at(monkeypatch):
    """"$383.3 billion" and "$383,285 million" are the same number."""

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [_chunk(text="Total net sales were $383,285 million in 2026.")]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $383.3 billion in 2026.",
            company_name="NVIDIA",
        )
    )

    assert response.verdicts[0].verdict == "verified"


async def test_claim_verify_does_not_score_a_share_count_against_a_dollar_claim(
    monkeypatch,
):
    """133 million shares is not $3.7 billion, despite both carrying a scale."""

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text=(
                    "Under the share repurchase program the Company repurchased "
                    "133 million shares during the period."
                ),
            )
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA repurchased $3.7 billion of its common stock in 2026.",
            company_name="NVIDIA",
        )
    )

    verdict = response.verdicts[0]
    assert verdict.verdict == "insufficient_evidence"
    assert not verdict.evidence_against


async def test_claim_verify_fails_closed_when_metric_is_unknown(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [_chunk(text="The Company shipped 240 million units in 2026.")]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA shipped 240 million units in 2026.",
            company_name="NVIDIA",
        )
    )

    verdict = response.verdicts[0]
    assert verdict.verdict == "insufficient_evidence"
    assert verdict.reason_code == "claim_metric_not_identified"
    assert "claim_metric_not_identified" in verdict.missing_obligations


async def test_claim_verify_cutoff_is_day_granular_within_one_fiscal_year(monkeypatch):
    """Same fiscal year is not the same point in time.

    A filing published in July cannot support a claim written in May, even
    though both carry fiscal_year 2026.
    """

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text="Total net sales were $11.6 billion in 2026.",
                fiscal_year=2026,
                filed_date=date(2026, 7, 30),
            )
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $11.6 billion in 2026.",
            company_name="NVIDIA",
            as_of=datetime.fromisoformat("2026-05-01T00:00:00+00:00"),
        )
    )

    verdict = response.verdicts[0]
    assert verdict.verdict == "insufficient_evidence"
    assert verdict.reason_code == "no_evidence_within_period_and_cutoff_scope"


async def test_claim_verify_filters_future_fiscal_year_chunks(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text="Total net sales were $9.1 billion in 2027.",
                fiscal_year=2027,
                filed_date=date(2027, 2, 24),
            ),
            _chunk(text="Total net sales were $11.6 billion in 2026.", fiscal_year=2026),
        ]

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.claims_verify_endpoint(
        ClaimVerificationRequest(
            source_text="NVIDIA reported revenue of $11.6 billion in 2026.",
            company_name="NVIDIA",
            as_of=datetime.fromisoformat("2026-08-20T00:00:00+00:00"),
        )
    )

    assert response.verdicts[0].verdict == "verified"
    assert len(response.verdicts[0].evidence_for) == 1
    assert (
        response.verdicts[0].evidence_for[0].source
        == "https://www.sec.gov/Archives/example.txt"
    )
