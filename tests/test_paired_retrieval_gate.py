from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from evaluation import paired_retrieval_gate as runner


def snapshot(*, chunks: int = 10, ticker: str = "AMD") -> runner.CorpusSnapshot:
    return runner.CorpusSnapshot(
        companies=frozenset({(ticker, "Advanced Micro Devices, Inc.")}),
        filings=frozenset(
            {
                (
                    ticker,
                    "10-K",
                    2022,
                    "ANNUAL",
                    None,
                    None,
                    "accession",
                    "cik",
                    "source",
                )
            }
        ),
        chunk_count=chunks,
    )


def dataset_row(case_id: str = "case-1") -> dict:
    return {
        "financebench_id": case_id,
        "question": "What was AMD revenue in FY2022?",
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "doc_type": "10k",
        "doc_period": 2022,
        "evidence": [{"evidence_text": "Revenue was 100."}],
    }


def test_same_database_target_is_rejected_even_with_different_credentials() -> None:
    with pytest.raises(runner.GateError, match="same target"):
        runner.validate_distinct_urls(
            "postgresql://old:one@localhost/db",
            "postgres://new:two@localhost:5432/db",
        )


def test_identity_comparison_allows_chunk_count_change_only() -> None:
    runner.assert_identity_sets_match(snapshot(chunks=10), snapshot(chunks=12))
    with pytest.raises(runner.GateError, match="company identity"):
        runner.assert_identity_sets_match(snapshot(), snapshot(ticker="AMZN"))


def test_nearest_rank_p95() -> None:
    assert runner.nearest_rank_p95(range(1, 21)) == 19
    assert runner.nearest_rank_p95([7]) == 7


@pytest.mark.asyncio
async def test_oracle_summary_uses_the_explicit_pool(monkeypatch) -> None:
    connection = object()

    class AcquireContext:
        async def __aenter__(self):
            return connection

        async def __aexit__(self, *args):
            return False

    class FakePool:
        def acquire(self):
            return AcquireContext()

    seen = []

    async def fake_fetch_filing_chunks(received_connection, gold):
        seen.append((received_connection, gold.doc_name))
        return ["Revenue was 100."]

    monkeypatch.setattr(runner, "_fetch_filing_chunks", fake_fetch_filing_chunks)
    summary = await runner.oracle_summary(FakePool(), [dataset_row()])

    assert seen == [(connection, "AMD_2022_10K")]
    assert summary["strict_reachable_spans"] == 1
    assert summary["strict_ceiling_recall"] == 1.0


def test_load_dataset_records_sha_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    raw = json.dumps([dataset_row()], separators=(",", ":")).encode()
    path.write_bytes(raw)
    rows, digest = runner.load_dataset(path)
    assert rows == [dataset_row()]
    assert digest == __import__("hashlib").sha256(raw).hexdigest()

    path.write_text(json.dumps([dataset_row(), dataset_row()]), encoding="utf-8")
    with pytest.raises(runner.GateError, match="duplicate"):
        runner.load_dataset(path)


@pytest.mark.asyncio
async def test_measurements_alternate_order_and_pass_question_only(monkeypatch) -> None:
    events: list[str] = []
    pools = {"old": object(), "new": object()}

    async def fake_retrieve(**kwargs):
        label = "old" if kwargs["pool"] is pools["old"] else "new"
        events.append(label)
        assert kwargs["company"] is None
        assert kwargs["filing_type"] is None
        assert kwargs["fiscal_year"] is None
        assert kwargs["sector"] is None
        assert kwargs["k"] == 10
        return []

    monkeypatch.setattr(runner, "retrieve", fake_retrieve)
    rows = [dataset_row("a"), dataset_row("b")]
    records, latencies = await runner.run_paired_measurements(
        pools["old"],
        pools["new"],
        rows,
        alpha=0.7,
        warmup_rounds=1,
        measured_rounds=1,
    )
    assert events == ["old", "new", "new", "old"] * 2
    assert [record.financebench_id for record in records["old"][0]] == ["a", "b"]
    assert len(latencies["old"]) == len(latencies["new"]) == 2


@pytest.mark.asyncio
async def test_any_retrieval_error_fails_the_run(monkeypatch) -> None:
    async def broken_retrieve(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(runner, "retrieve", broken_retrieve)
    with pytest.raises(runner.GateError, match="old retrieval failed for case-1"):
        await runner.run_paired_measurements(
            object(),
            object(),
            [dataset_row()],
            alpha=0.7,
            warmup_rounds=0,
            measured_rounds=1,
        )


def test_decision_applies_all_development_gates() -> None:
    old = {
        "rounds": [
            {
                "round": 1,
                "strict": {"recall_at_10": 0.14},
                "zero_provenance_queries": 4,
            },
            {
                "round": 2,
                "strict": {"recall_at_10": 0.14},
                "zero_provenance_queries": 4,
            },
        ],
        "latency_ms": {"p95_nearest_rank": 100.0},
    }
    new = {
        "rounds": [
            {
                "round": 1,
                "strict": {"recall_at_10": 0.25},
                "zero_provenance_queries": 4,
            },
            {
                "round": 2,
                "strict": {"recall_at_10": 0.25},
                "zero_provenance_queries": 4,
            },
        ],
        "latency_ms": {"p95_nearest_rank": 150.0},
    }
    assert runner.decision(
        old, new, profile="development", oracle_non_regression=True
    )["accepted"] is True
    new["rounds"][1]["zero_provenance_queries"] = 5
    assert runner.decision(
        old, new, profile="development", oracle_non_regression=True
    )["accepted"] is False


def test_development_requires_floor_and_non_regression_but_heldout_only_non_regression() -> None:
    old = {
        "rounds": [
            {
                "round": 1,
                "strict": {"recall_at_10": 0.2},
                "zero_provenance_queries": 4,
            }
        ],
        "latency_ms": {"p95_nearest_rank": 100.0},
    }
    new = {
        "rounds": [
            {
                "round": 1,
                "strict": {"recall_at_10": 0.2},
                "zero_provenance_queries": 4,
            }
        ],
        "latency_ms": {"p95_nearest_rank": 100.0},
    }
    assert runner.decision(
        old, new, profile="development", oracle_non_regression=True
    )["accepted"] is False
    assert runner.decision(
        old, new, profile="heldout", oracle_non_regression=True
    )["accepted"] is True

    new["rounds"][0]["strict"]["recall_at_10"] = 0.19
    assert runner.decision(
        old, new, profile="heldout", oracle_non_regression=True
    )["accepted"] is False


def test_oracle_reachability_regression_rejects_otherwise_passing_run() -> None:
    summary = {
        "rounds": [
            {
                "round": 1,
                "strict": {"recall_at_10": 0.25},
                "zero_provenance_queries": 0,
            }
        ],
        "latency_ms": {"p95_nearest_rank": 100.0},
    }
    result = runner.decision(
        summary,
        summary,
        profile="development",
        oracle_non_regression=False,
    )
    assert result["accepted"] is False
    assert result["checks"]["strict_oracle_reachability_not_decreased"] is False


def test_write_report_refuses_different_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    runner.write_report(path, {"accepted": True})
    runner.write_report(path, {"accepted": True})
    with pytest.raises(runner.GateError, match="refusing to overwrite"):
        runner.write_report(path, {"accepted": False})


@pytest.mark.asyncio
async def test_run_rejects_runtime_same_database_and_closes_pools(monkeypatch, tmp_path):
    closed: list[str] = []

    class FakePool:
        def __init__(self, label):
            self.label = label

        async def close(self):
            closed.append(self.label)

    pools = [FakePool("old"), FakePool("new")]

    async def fake_create_pool(url):
        return pools.pop(0)

    monkeypatch.setattr(runner, "create_pool", fake_create_pool)
    async def same_identity(pool):
        return ("db", "127.0.0.1", 5432)

    monkeypatch.setattr(runner, "runtime_database_identity", same_identity)
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([dataset_row()]), encoding="utf-8")
    args = Namespace(
        old_db_url="postgresql://u:p@old/db",
        new_db_url="postgresql://u:p@new/db",
        dataset=dataset,
        output=tmp_path / "out.json",
        gate_profile="development",
        alpha=0.7,
        warmup_rounds=0,
        measured_rounds=1,
    )
    with pytest.raises(runner.GateError, match="same PostgreSQL"):
        await runner.run(args)
    assert closed == ["old", "new"]
