"""Tests for the sequential answer-prediction collector."""

from __future__ import annotations

import json
from types import SimpleNamespace

from evaluation import collect_answer_predictions as collector

from corpcheck.models import ChunkResult


def chunk(chunk_id: str) -> ChunkResult:
    return ChunkResult(chunk_id=chunk_id, text="evidence", score=1.0, company="AMD")


async def test_retrieval_only_is_sequential_and_never_calls_chat():
    events = []

    async def fake_retrieve(**kwargs):
        events.append(
            (
                "retrieve",
                kwargs["query"],
                kwargs["sector"],
                kwargs["company"],
                kwargs["filing_type"],
                kwargs["fiscal_year"],
            )
        )
        return [chunk(kwargs["query"])]

    def fake_abstain(question, chunks):
        assert question == chunks[0].chunk_id
        events.append(("abstain", chunks[0].chunk_id))
        return SimpleNamespace(
            abstain=False, reason="", top1=0.8, mean_top3=0.7, status="pass"
        )

    async def forbidden_chat(*args, **kwargs):
        raise AssertionError("retrieval-only mode contacted the LLM")

    records = [
        {
            "id": "q1",
            "question": "first",
            "sector": "Technology",
            "company": "AMD",
            "filing_type": "10-K",
            "year": 2022,
        },
        {"id": "q2", "question": "second"},
    ]
    predictions = await collector.collect_predictions(
        records,
        object(),
        retrieval_only=True,
        retrieve_fn=fake_retrieve,
        abstain_fn=fake_abstain,
        chat_fn=forbidden_chat,
    )

    assert events == [
        ("retrieve", "first", None, None, None, None),
        ("abstain", "first"),
        ("retrieve", "second", None, None, None, None),
        ("abstain", "second"),
    ]
    assert [prediction["id"] for prediction in predictions] == ["q1", "q2"]
    assert predictions[0]["answer"] == ""
    assert predictions[0]["diagnostics"]["mode"] == "retrieval-only"


async def test_seed_filters_are_only_used_when_explicitly_enabled():
    captured = {}
    gate_context = {}

    async def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [chunk("evidence")]

    def fake_abstain(question, chunks, *, expected_company=None):
        gate_context["expected_company"] = expected_company
        return SimpleNamespace(
            abstain=False, reason="", top1=0.8, mean_top3=0.7, status="pass"
        )

    record = {
        "id": "q1",
        "question": "question",
        "sector": "Technology",
        "company": "AMD",
        "filing_type": "10-K",
        "year": 2022,
    }
    await collector.collect_predictions(
        [record],
        object(),
        retrieval_only=True,
        use_seed_filters=True,
        retrieve_fn=fake_retrieve,
        abstain_fn=fake_abstain,
    )

    assert captured["sector"] == "Technology"
    assert captured["company"] == "AMD"
    assert captured["filing_type"] == "10-K"
    assert captured["fiscal_year"] == 2022
    assert gate_context["expected_company"] == "AMD"


async def test_normal_mode_calls_chat_only_when_gate_passes():
    async def fake_retrieve(**kwargs):
        return [chunk(kwargs["query"])]

    def fake_abstain(question, chunks):
        should_stop = chunks[0].chunk_id == "weak"
        return SimpleNamespace(
            abstain=should_stop,
            reason="insufficient" if should_stop else "",
            top1=0.2 if should_stop else 0.8,
            mean_top3=0.1 if should_stop else 0.7,
            status="below_top1_floor" if should_stop else "pass",
        )

    calls = []

    async def fake_chat(question, chunks):
        calls.append(question)
        return {"answer": "supported [1]", "thinking": "internal"}

    records = [{"id": "q1", "question": "strong"}, {"id": "q2", "question": "weak"}]
    predictions = await collector.collect_predictions(
        records,
        object(),
        retrieval_only=False,
        retrieve_fn=fake_retrieve,
        abstain_fn=fake_abstain,
        chat_fn=fake_chat,
    )

    assert calls == ["strong"]
    assert predictions[0]["answer"] == "supported [1]"
    assert predictions[0]["abstained"] is False
    assert predictions[1]["answer"] == "insufficient"
    assert predictions[1]["abstained"] is True


async def test_collector_passes_question_to_answerability_gate():
    seen = []

    async def fake_retrieve(**kwargs):
        return [chunk("evidence")]

    def fake_answerability(question, chunks):
        seen.append((question, chunks[0].chunk_id))
        return SimpleNamespace(
            abstain=True,
            reason="metadata mismatch",
            top1=None,
            mean_top3=None,
            status="year_mismatch",
        )

    predictions = await collector.collect_predictions(
        [{"id": "q1", "question": "Costco FY2099"}],
        object(),
        retrieval_only=True,
        retrieve_fn=fake_retrieve,
        abstain_fn=fake_answerability,
    )

    assert seen == [("Costco FY2099", "evidence")]
    assert predictions[0]["diagnostics"]["gate_status"] == "year_mismatch"


def test_normal_cli_fails_before_database_when_llm_is_unset(tmp_path, monkeypatch, capsys):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([{"id": "q1", "question": "question"}]), encoding="utf-8")
    monkeypatch.setattr(collector.config, "SGLANG_BASE_URL", "")

    exit_code = collector.main(["--seed", str(seed), "--output", str(tmp_path / "out.jsonl")])

    assert exit_code == 2
    assert "SGLANG_BASE_URL is unset" in capsys.readouterr().err


def test_seed_is_valid_and_jsonl_writer_preserves_records(tmp_path):
    records = collector.read_seed(collector.DEFAULT_SEED)
    assert 6 <= len(records) <= 10
    assert sum(record["should_abstain"] for record in records) >= 2

    output = tmp_path / "predictions.jsonl"
    collector.write_jsonl([{"id": "q1", "answer": "ok"}], output)
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"id": "q1", "answer": "ok"}
    ]
