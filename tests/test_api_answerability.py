from __future__ import annotations

from datetime import date

from corpcheck.api import main
from corpcheck.models import AnswerabilityRequest, ChunkResult
from corpcheck.retrieval.abstain import AbstainDecision


def chunk(**overrides) -> ChunkResult:
    values = {
        "chunk_id": "aapl-2022-1",
        "text": "Total net sales were $394.3 billion in 2022.",
        "score": 1.0,
        "cos_sim": 0.71,
        "company": "AAPL",
        "filing_type": "10-K",
        "fiscal_year": 2022,
        "filed_date": date(2022, 10, 28),
        "source_url": "https://www.sec.gov/Archives/example.txt",
        "source_type": "sec",
        "display_title": "Apple 2022 Form 10-K",
    }
    values.update(overrides)
    return ChunkResult(**values)


async def test_answerability_returns_gate_metrics_and_evidence(monkeypatch):
    evidence = [chunk(), chunk(chunk_id="aapl-2022-2", cos_sim=None)]
    calls = {}

    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        calls.update(kwargs)
        return evidence

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        main,
        "evaluate_answerability",
        lambda query, chunks, expected_company=None: AbstainDecision(
            False, top1=0.71, mean_top3=0.69, status="pass"
        ),
    )

    response = await main.answerability_endpoint(
        AnswerabilityRequest(
            query="What were Apple's 2022 net sales?",
            company="AAPL",
            filing_type="10-K",
            year=2022,
        )
    )

    assert response.answerable is True
    assert response.gate_status == "pass"
    assert response.reason == "Evidence passed both confidence floors."
    assert response.llm_consulted is False
    assert response.similarity.top1_cos_sim == 0.71
    assert response.coverage.retrieved == 2
    assert response.coverage.with_dense_score == 1
    assert response.coverage.sparse_only == 1
    assert response.coverage.companies == ["AAPL"]
    assert response.chunks == evidence
    assert response.chunks[0].source_url == evidence[0].source_url
    assert calls == {
        "pool": "pool",
        "query": "What were Apple's 2022 net sales?",
        "k": 5,
        "alpha": main.config.DEFAULT_ALPHA,
        "sector": None,
        "company": "AAPL",
        "filing_type": "10-K",
        "fiscal_year": 2022,
    }


async def test_answerability_exposes_abstention_without_calling_llm(monkeypatch):
    async def fake_get_pool():
        return "pool"

    async def fake_retrieve(**kwargs):
        return []

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    response = await main.answerability_endpoint(
        AnswerabilityRequest(query="How do I bake sourdough?")
    )

    assert response.answerable is False
    assert response.gate_status == "no_results"
    assert response.llm_consulted is False
    assert response.similarity.top1_cos_sim is None
    assert response.coverage.retrieved == 0
    assert response.chunks == []


async def test_http_lifespan_warms_cached_embedding_model(monkeypatch):
    events = []

    async def fake_get_pool():
        events.append("pool")
        return "pool"

    async def fake_load_known_tickers(pool):
        events.append(("tickers", pool))

    def fake_get_model():
        events.append("model")
        return object()

    async def fake_close_pool():
        events.append("close")

    monkeypatch.setattr(main, "get_pool", fake_get_pool)
    monkeypatch.setattr(main, "load_known_tickers", fake_load_known_tickers)
    monkeypatch.setattr(main, "get_model", fake_get_model)
    monkeypatch.setattr(main, "close_pool", fake_close_pool)

    async with main.lifespan(main.app):
        assert events == ["pool", ("tickers", "pool"), "model"]

    assert events == ["pool", ("tickers", "pool"), "model", "close"]
