from datetime import date, datetime

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
        "filed_date": date(2026, 10, 28),
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
    assert response.verdicts[0].reason_code == "evidence_binds_value_and_metric"
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


async def test_claim_verify_marks_conflict_when_only_opposing_numeric_found(monkeypatch):
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

    assert response.verdicts[0].verdict == "conflicting"
    assert response.verdicts[0].reason_code == (
        "retrieved_chunks_disagree_with_numeric_assertion"
    )
    assert len(response.verdicts[0].evidence_against) == 1


async def test_claim_verify_filters_future_fiscal_year_chunks(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return [
            _chunk(
                text="Total net sales were $9.1 billion in 2027.",
                fiscal_year=2027,
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
