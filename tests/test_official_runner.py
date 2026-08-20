import json
from argparse import Namespace
from pathlib import Path

from evaluation.official.manifest import BenchmarkPlan, BenchmarkResult, OfficialManifest
from evaluation.official.runner import run_plan, write_run_result


def _build_manifest() -> OfficialManifest:
    return OfficialManifest(
        version="2026-08-20",
        created_at_utc="2026-08-20T00:00:00Z",
        runs=[
            BenchmarkPlan(
                name="financebench",
                status="planned",
                source_repo="https://github.com/czyssrs/FinanceBench",
                notes="",
                dataset="evaluation/datasets/financebench_filtered.json",
                runner="financebench",
                required_files=None,
                default_args={"k": 10, "alpha": 0.7, "threshold": 0.5},
            )
        ],
    )


def test_write_run_result_writes_timestamped_payload_and_pointer(tmp_path: Path):
    payload = _build_manifest()
    args = Namespace(
        manifest=Path("evaluation/official/benchmark_manifest.yaml"),
        seed=42,
        output_root=tmp_path,
        label="manual-check",
        log_level="INFO",
        task=None,
        only_planned=False,
        dataset=None,
        k=None,
        alpha=None,
        threshold=None,
        limit=None,
        no_quarter_match=False,
        no_doc_gate=False,
        smoke=False,
        fusion_strategy=None,
    )

    result = BenchmarkResult(
        task="financebench",
        status="completed",
        timestamp_utc="2026-08-20T00:00:00Z",
        elapsed_seconds=0.0,
        summary={"metadata": {"dataset": "x"}},
        config={"k": 10},
    )

    write_run_result(payload, args, [result])

    stamped = tmp_path / "results_manual-check.json"
    latest = tmp_path / "results.json"
    assert stamped.exists()
    assert latest.exists()

    stamped_payload = (tmp_path / "results_manual-check.json").read_text(encoding="utf-8")
    latest_payload = latest.read_text(encoding="utf-8")
    assert stamped_payload == latest_payload

    decoded = json.loads(stamped_payload)
    assert decoded["run_metadata"]["seed"] == 42
    assert decoded["manifest_version"] == "2026-08-20"
    assert decoded["results"][0]["task"] == "financebench"


def test_financebench_smoke_run_produces_metadata_and_trace(tmp_path: Path):
    dataset = tmp_path / "financebench_smoke.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"financebench_id": "fb1", "question": "Revenue for Apple", '
                '"company": "Apple", "doc_name": "AAPL_2023_10K", '
                '"doc_type": "10-k", "doc_period": "2023"}',
                '{"financebench_id": "fb2", "question": "Margins are stable", '
                '"company": "MSFT", "doc_type": "10-k", "doc_period": "2024"}',
            ]
        ),
        encoding="utf-8",
    )

    plan = BenchmarkPlan(
        name="financebench",
        status="planned",
        source_repo="https://github.com/czyssrs/FinanceBench",
        dataset=str(dataset),
        runner="financebench",
    )

    args = Namespace(
        manifest=Path("evaluation/official/benchmark_manifest.yaml"),
        dataset=dataset,
        output_root=tmp_path,
        label="financebench-smoke-test",
        seed=77,
        k=4,
        alpha=0.7,
        threshold=0.5,
        limit=2,
        no_quarter_match=False,
        no_doc_gate=False,
        fusion_strategy=None,
        smoke=True,
        task="financebench",
        only_planned=False,
        log_level="INFO",
    )

    result = run_plan(plan, args)

    assert result.status == "completed"
    assert result.summary["smoke"] is True
    assert result.summary["metadata"]["seed"] == 77
    assert result.summary["metadata"]["source_repo"] == "https://github.com/czyssrs/FinanceBench"
    assert result.summary["smoke_rows"] == 2
    assert result.summary["smoke_trace"]["queries"] == 2
    assert (tmp_path / "financebench/financebench-smoke-test").exists()
    assert (tmp_path / "financebench/financebench-smoke-test" / "summary.json").exists()
