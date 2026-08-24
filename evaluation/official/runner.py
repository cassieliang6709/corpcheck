"""Official benchmark execution hub (FinanceBench / FinRank / AVeriTeC).

Includes a lightweight ``--smoke`` mode that emits deterministic synthetic
results for interview-ready dry-runs without requiring a live PostgreSQL
cluster.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from corpcheck.claims import normalize_claims
from corpcheck.claims.schema import ClaimEvaluationRequest
from corpcheck.claims.verifier import verify_claims
from corpcheck.db import close_pool, get_pool
from corpcheck.models import ChunkResult
from corpcheck.retrieval import load_known_tickers, retrieve
from evaluation.ir_eval import (
    QueryRecord,
    gate_diagnostics,
    run_retrieval,
    score_records,
    write_run,
)

from .manifest import BenchmarkPlan, BenchmarkResult, OfficialManifest, load_manifest

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_manifest.yaml")
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "runs"
DEFAULT_K = 10
DEFAULT_ALPHA = 0.7
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset missing: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(payload)
        return rows

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected JSON array")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path}: every record must be an object")
    return payload


def _smoke_fraction(*parts: object) -> float:
    seed_text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return (value % 10000) / 10000.0


def _smoke_record_id(plan: BenchmarkPlan, row: dict[str, Any], idx: int) -> str:
    return str(
        row.get("financebench_id")
        or row.get("id")
        or row.get("query_id")
        or f"{plan.name}-{idx + 1}"
    )


def _run_name(plan_name: str, label: Optional[str]) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    suffix = label.strip() if label else stamp
    return f"{plan_name}/{suffix}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(rendered, encoding="utf-8")


def _git_sha() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - environment specific
        return None


def _dataset_fingerprint(dataset: Path) -> str:
    if not dataset.exists():
        return "missing"
    try:
        stat = dataset.stat()
        return f"{dataset.name}|{stat.st_size}|{int(stat.st_mtime)}"
    except Exception:
        return f"{dataset.name}|unknown"


def _hardware_fingerprint() -> str:
    return "|".join(
        filter(
            bool,
            [
                platform.system(),
                platform.release(),
                platform.machine(),
                platform.processor() or "proc=unknown",
                f"cpu={os.cpu_count()}",
            ],
        )
    )


def _build_result_metadata(
    plan: BenchmarkPlan,
    dataset: Path | None,
    k: int,
    alpha: float,
    threshold: float,
    seed: int,
    hardware: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": str(dataset) if dataset is not None else None,
        "dataset_version": _dataset_fingerprint(dataset) if dataset is not None else "none",
        "commit": _git_sha(),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "hardware": hardware,
        "seed": seed,
        "params": {
            "k": k,
            "alpha": alpha,
            "threshold": threshold,
            **extra,
        },
        "source_repo": plan.source_repo,
    }


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        candidate = int(value)
        return candidate if candidate > 0 else default
    except Exception:
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_plan_ready(plan: BenchmarkPlan) -> bool:
    required = plan.required_files or []
    return all(Path(path).exists() for path in required)


def _normalize_smoke_label(raw: Any, default: str) -> str:
    if not isinstance(raw, str):
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    return {
        "entails": "verified",
        "supports": "verified",
        "support": "verified",
        "yes": "verified",
        "true": "verified",
        "refutes": "conflicting",
        "contradicts": "conflicting",
        "refute": "conflicting",
        "false": "conflicting",
        "no": "conflicting",
        "both": "conflicting",
        "uncertain": "insufficient_evidence",
        "insufficient": "insufficient_evidence",
        "not enough": "insufficient_evidence",
        "not_yet_decidable": "not_yet_decidable",
        "not yet decidable": "not_yet_decidable",
        "pending": "not_yet_decidable",
    }.get(normalized, default)


def _smoke_financebench_records(
    plan: BenchmarkPlan,
    rows: list[dict[str, Any]],
    *,
    k: int,
    threshold: float,
    seed: int,
    match_quarter: bool,
) -> tuple[list[QueryRecord], dict[str, Any], dict[str, Any]]:
    records: list[QueryRecord] = []
    output_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        record_id = _smoke_record_id(plan, row, idx)
        question = str(
            row.get("question")
            or row.get("query")
            or row.get("claim")
            or row.get("text")
            or ""
        ).strip()
        if not question:
            question = f"smoke financebench query {idx + 1}"

        score = round(0.45 + 0.55 * _smoke_fraction(seed, record_id, "financebench"), 4)
        overlaps = [
            [max(0.0, score - 0.16 * position) for position in range(k)]
            for _ in range(1)
        ]
        overlaps[0] = [min(1.0, value + 0.2) for value in overlaps[0]]

        overlaps_raw = [
            [min(1.0, value + 0.1) for value in overlaps[0]]
            for _ in range(1)
        ]

        quarter = row.get("doc_period")
        if isinstance(quarter, int):
            quarter = str(quarter)
        quarter_value = str(quarter).upper() if quarter else None

        doc_name = str(
            row.get("doc_name")
            or row.get("doc")
            or row.get("company", "UNKNOWN")
        )

        company = str(row.get("company") or row.get("ticker") or "UNK")
        filing_type = row.get("doc_type")
        if isinstance(filing_type, str) and filing_type.strip():
            filing_type = filing_type.strip().lower()
            if filing_type in {"10k", "10-k"}:
                filing_type = "10-K"
            elif filing_type in {"10q", "10-q"}:
                filing_type = "10-Q"
            elif filing_type in {"8k", "8-k"}:
                filing_type = "8-K"
        else:
            filing_type = None

        try:
            fiscal_year = int(row.get("doc_period")) if row.get("doc_period") else None
        except (TypeError, ValueError):
            fiscal_year = None

        gold_doc = {
            "ticker": company,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "quarter": (
                quarter_value
                if isinstance(quarter_value, str) and quarter_value.upper().startswith("Q")
                else None
            ),
            "doc_name": doc_name,
        }

        retrieved = [
            {
                "rank": position + 1,
                "chunk_id": f"{record_id}-chunk-{position + 1}",
                "score": float(overlaps[0][position] if position < k else 0.0),
                "company": company,
                "filing_type": filing_type,
                "fiscal_year": fiscal_year,
                "period_label": quarter_value if quarter_value else "FY",
                "source_type": "sec",
                "content_kind": "narrative",
                "text_preview": question[:80],
            }
            for position in range(k)
        ]

        gate_base = _smoke_fraction(seed, record_id, "financebench", idx, "gate")
        doc_gate_pass = [position < max(1, int(gate_base * k)) for position in range(k)]

        records.append(
            QueryRecord(
                financebench_id=record_id,
                question=question,
                doc_name=doc_name,
                gold_doc=gold_doc,
                n_gold_spans=1,
                latency_ms=7,
                retrieved=retrieved,
                overlaps={"clean": overlaps, "raw": overlaps_raw},
                doc_gate_pass=doc_gate_pass,
                error=None,
            )
        )

        output_rows.append(
            {
                "financebench_id": record_id,
                "question": question,
                "match_quarter": match_quarter,
                "doc_name": doc_name,
                "company": company,
                "fiscal_year": fiscal_year,
                "top1_score": overlaps[0][0] if overlaps else 0.0,
            }
        )

    gate_stats = {
        "queries": len(records),
        "smoke_seed": seed,
        "threshold": threshold,
        "match_quarter": match_quarter,
        "record_count": len(output_rows),
    }

    return records, gate_stats, output_rows


# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------


def run_financebench_smoke(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    threshold: float,
    seed: int,
    match_quarter: bool,
    no_doc_gate: bool,
    limit: Optional[int],
    output_root: Path,
    fusion_strategy: Optional[str],
    label: Optional[str],
) -> BenchmarkResult:
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    records, gate_stats, smoke_rows = _smoke_financebench_records(
        plan,
        rows,
        k=k,
        threshold=threshold,
        seed=seed,
        match_quarter=match_quarter,
    )
    run_dir = output_root / _run_name(plan.name, label)
    write_run(
        run_dir,
        records,
        k=k,
        alpha=alpha,
        threshold=threshold,
        dataset=dataset,
    )

    gated = score_records(
        records,
        threshold=threshold,
        variant="clean",
        apply_gate=not no_doc_gate,
        k=k,
    )
    ungated = score_records(
        records,
        threshold=threshold,
        variant="clean",
        apply_gate=False,
        k=k,
    )

    return BenchmarkResult(
        task=plan.name,
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "smoke": True,
            "records": len(records),
            "errors": sum(1 for row in records if row.error is not None),
            "primary_with_gate": gated,
            "primary_no_gate": ungated,
            "gate_diagnostics": gate_diagnostics(records),
            "smoke_rows": len(smoke_rows),
            "smoke_trace": gate_stats,
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "match_quarter": match_quarter,
                    "no_doc_gate": no_doc_gate,
                    "fusion_strategy": fusion_strategy,
                    "mode": "smoke",
                    "smoke_rows": smoke_rows[:3],
                },
            ),
        },
        config={
            "dataset": str(dataset),
            "k": k,
            "alpha": alpha,
            "match_quarter": match_quarter,
            "fusion_strategy": fusion_strategy,
            "no_doc_gate": no_doc_gate,
            "smoke": True,
        },
    )


def run_financebench(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    threshold: float,
    seed: int,
    match_quarter: bool,
    no_doc_gate: bool,
    limit: Optional[int],
    output_root: Path,
    fusion_strategy: Optional[str],
    label: Optional[str],
    smoke: bool = False,
) -> BenchmarkResult:
    """Run deterministic retrieval IR eval for FinanceBench-format rows."""
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    if smoke:
        smoke_rows = rows
        run_dir = output_root / _run_name(plan.name, label)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_json = run_dir / "summary.json"
        payload = {
            "task": plan.name,
            "smoke": True,
            "queries": len(smoke_rows),
            "trace": [
                {
                    "index": index,
                    "row_id": row.get("financebench_id") or row.get("id"),
                    "question": row.get("question"),
                    "company": row.get("company"),
                    "doc_type": row.get("doc_type"),
                    "doc_period": row.get("doc_period"),
                }
                for index, row in enumerate(smoke_rows)
            ],
        }
        summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return BenchmarkResult(
            task=plan.name,
            status="completed",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={
                "smoke": True,
                "smoke_rows": len(smoke_rows),
                "smoke_trace": {
                    "queries": len(smoke_rows),
                    "first_questions": [
                        row.get("question")
                        for row in smoke_rows[: min(len(smoke_rows), 2)]
                    ],
                },
                "metadata": _build_result_metadata(
                    plan,
                    dataset,
                    k=k,
                    alpha=alpha,
                    threshold=threshold,
                    seed=seed,
                    hardware=_hardware_fingerprint(),
                    extra={
                        "match_quarter": match_quarter,
                        "no_doc_gate": no_doc_gate,
                        "fusion_strategy": fusion_strategy,
                        "smoke": True,
                    },
                ),
            },
            config={
                "dataset": str(dataset),
                "k": k,
                "alpha": alpha,
                "match_quarter": match_quarter,
                "fusion_strategy": fusion_strategy,
                "no_doc_gate": no_doc_gate,
                "smoke": True,
            },
        )

    try:
        records = asyncio.run(
            run_retrieval(
                rows,
                k=k,
                alpha=alpha,
                match_quarter=match_quarter,
                fusion_strategy=fusion_strategy,
            )
        )
    except Exception as exc:  # pragma: no cover - infra dependent
        return BenchmarkResult(
            task=plan.name,
            status="failed",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={"error": str(exc)},
            config={"dataset": str(dataset), "k": k, "alpha": alpha},
            error=str(exc),
        )

    run_dir = output_root / _run_name(plan.name, label)
    write_run(
        run_dir,
        records,
        k=k,
        alpha=alpha,
        threshold=threshold,
        dataset=dataset,
    )

    gated = score_records(
        records,
        threshold=threshold,
        variant="clean",
        apply_gate=not no_doc_gate,
        k=k,
    )
    ungated = score_records(
        records,
        threshold=threshold,
        variant="clean",
        apply_gate=False,
        k=k,
    )

    errors = sum(1 for row in records if row.error is not None)

    return BenchmarkResult(
        task=plan.name,
        # A run that lost queries to infrastructure failures scored a different
        # denominator than the protocol asks for, so its metrics are not
        # comparable to a clean run. Reporting that as "completed" is how a
        # degraded run gets quoted as a result.
        status="completed" if errors == 0 else "degraded",
        error=(
            None
            if errors == 0
            else f"{errors} of {len(records)} queries failed to retrieve"
        ),
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "records": len(records),
            "errors": errors,
            "scored": len(records) - errors,
            "comparable": errors == 0,
            "primary_with_gate": gated,
            "primary_no_gate": ungated,
            "gate_diagnostics": gate_diagnostics(records),
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "match_quarter": match_quarter,
                    "no_doc_gate": no_doc_gate,
                    "fusion_strategy": fusion_strategy,
                },
            ),
        },
        config={
            "dataset": str(dataset),
            "k": k,
            "alpha": alpha,
            "match_quarter": match_quarter,
            "fusion_strategy": fusion_strategy,
            "no_doc_gate": no_doc_gate,
        },
    )


# ---------------------------------------------------------------------------
# FinRank
# ---------------------------------------------------------------------------


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> list[Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def run_finrank_smoke(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    threshold: float,
    seed: int,
    limit: Optional[int],
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    total = 0
    evaluated = 0
    pass_count = 0
    rows_out: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        question = row.get("question") or row.get("query") or row.get("claim")
        if not isinstance(question, str) or not question.strip():
            continue

        positives = _first_present(
            row,
            ["positive", "positives", "relevant", "supports", "label", "gold"],
        )
        negatives = _first_present(
            row,
            ["negative", "negatives", "hard_negative", "hard_negatives"],
        )
        if not positives and not negatives:
            rows_out.append(
                {
                    "id": row.get("id") or row.get("query_id"),
                    "question": question,
                    "status": "smoke_skipped_no_gold",
                }
            )
            continue

        total += 1
        score = _smoke_fraction(seed, plan.name, idx, _smoke_record_id(plan, row, idx))
        top_chunks = min(k, 4)
        positive_hit = bool(positives) and score > 0.42
        negative_hit = bool(negatives) and score < 0.58
        passed = positive_hit and not negative_hit
        evaluated += 1
        pass_count += int(passed)
        rows_out.append(
            {
                "id": row.get("id") or row.get("query_id"),
                "question": question,
                "positive_hit": positive_hit,
                "negative_hit": negative_hit,
                "pass": passed,
                "top_chunks": top_chunks,
            }
        )

    output_path = output_root / _run_name(plan.name, label) / "finrank_rows.jsonl"
    _write_jsonl(output_path, rows_out)

    return BenchmarkResult(
        task=plan.name,
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "smoke": True,
            "rows": len(rows),
            "evaluated": evaluated,
            "pass": pass_count,
            "acc": (pass_count / evaluated) if evaluated else 0.0,
            "output": str(output_path),
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "task": "finrank_smoke",
                    "mode": "smoke",
                    "total_input_rows": len(rows),
                },
            ),
        },
        config={
            "dataset": str(dataset),
            "k": k,
            "alpha": alpha,
            "smoke": True,
        },
    )


def _text_or_meta_match(chunk: ChunkResult, hint: Any) -> bool:
    if not isinstance(hint, str):
        return False

    haystack = " ".join(
        item
        for item in [
            chunk.text,
            chunk.chunk_id,
            chunk.company,
            chunk.source_url or "",
            chunk.filing_type or "",
            str(chunk.fiscal_year or ""),
        ]
        if item
    ).lower()
    return hint.lower() in haystack


def run_finrank(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    limit: Optional[int],
    threshold: float,
    seed: int,
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    try:
        pool = asyncio.run(get_pool())
        asyncio.run(load_known_tickers(pool))
    except Exception as exc:  # pragma: no cover - infra dependent
        return BenchmarkResult(
            task=plan.name,
            status="skipped",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={"error": str(exc)},
            config={"dataset": str(dataset), "k": k, "alpha": alpha},
            error=str(exc),
        )

    total = 0
    evaluated = 0
    pass_count = 0
    rows_out: list[dict[str, Any]] = []

    async def _run() -> None:
        nonlocal total, evaluated, pass_count
        for row in rows:
            question = row.get("question") or row.get("query")
            if not isinstance(question, str) or not question.strip():
                continue

            positives = _first_present(
                row,
                ["positive", "positives", "relevant", "supports", "label"],
            )
            negatives = _first_present(
                row,
                ["negative", "negatives", "hard_negative", "hard_negatives"],
            )
            if not positives:
                rows_out.append(
                    {
                        "id": row.get("id"),
                        "question": question,
                        "status": "skipped_no_positive",
                    }
                )
                continue

            total += 1
            chunks = await retrieve(
                pool=pool,
                query=question,
                k=k,
                alpha=alpha,
                sector=row.get("sector"),
                company=row.get("company"),
                filing_type=row.get("filing_type"),
                fiscal_year=row.get("year") if isinstance(row.get("year"), int) else None,
            )

            positive_hit = any(
                _text_or_meta_match(chunk, hint) for chunk in chunks for hint in positives
            )
            negative_hit = any(
                _text_or_meta_match(chunk, hint) for chunk in chunks for hint in negatives
            )

            evaluated += 1
            passed = positive_hit and not negative_hit
            pass_count += int(passed)
            rows_out.append(
                {
                    "id": row.get("id"),
                    "question": question,
                    "positive_hit": positive_hit,
                    "negative_hit": negative_hit,
                    "pass": passed,
                    "top_chunks": len(chunks),
                }
            )

        await close_pool()

    try:
        asyncio.run(_run())
    except Exception as exc:  # pragma: no cover - infra dependent
        return BenchmarkResult(
            task=plan.name,
            status="failed",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={"error": str(exc)},
            config={"dataset": str(dataset), "k": k, "alpha": alpha},
            error=str(exc),
        )

    output_path = output_root / _run_name(plan.name, label) / "finrank_rows.jsonl"
    _write_jsonl(output_path, rows_out)

    return BenchmarkResult(
        task=plan.name,
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "rows": len(rows),
            "evaluated": evaluated,
            "pass": pass_count,
            "acc": (pass_count / evaluated) if evaluated else 0.0,
            "output": str(output_path),
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "task": "finrank",
                },
            ),
        },
        config={"dataset": str(dataset), "k": k, "alpha": alpha},
    )


# ---------------------------------------------------------------------------
# AVeriTeC
# ---------------------------------------------------------------------------


def _normalize_gold_verdict(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    value = raw.strip().lower()
    aliases = {
        "entails": "verified",
        "support": "verified",
        "supported": "verified",
        "true": "verified",
        "yes": "verified",
        "contradicts": "conflicting",
        "contradict": "conflicting",
        "false": "conflicting",
        "no": "conflicting",
        "refuted": "conflicting",
        "refute": "conflicting",
        "both": "conflicting",
        "uncertain": "insufficient_evidence",
        "unknown": "insufficient_evidence",
        "undetermined": "insufficient_evidence",
    }
    return aliases.get(value, value)


def run_averitec_smoke(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    seed: int,
    threshold: float,
    limit: Optional[int],
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    evaluated = 0
    correct = 0
    rows_out: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        claim_text = row.get("claim") or row.get("claim_text") or row.get("statement")
        gold = _normalize_smoke_label(
            row.get("label") or row.get("verdict"), default="insufficient_evidence"
        )
        if not isinstance(claim_text, str) or not claim_text.strip():
            continue

        predicted = gold
        if _smoke_fraction(seed, plan.name, idx, "averitec") < 0.25:
            candidate = [
                "verified",
                "conflicting",
                "insufficient_evidence",
                "not_yet_decidable",
                "non_verifiable",
            ]
            choice = _smoke_fraction(seed, plan.name, idx, "flip")
            predicted = candidate[int(choice * len(candidate))]

        claim_id = str(row.get("id") or row.get("claim_id") or f"{plan.name}-{idx + 1}")
        matched = gold == predicted
        correct += int(matched)
        evaluated += 1

        rows_out.append(
            {
                "claim_id": claim_id,
                "gold": gold,
                "pred": predicted,
                "match": matched,
                "k": k,
            }
        )

    output_path = output_root / _run_name(plan.name, label) / "averitec_rows.jsonl"
    _write_jsonl(output_path, rows_out)

    return BenchmarkResult(
        task=plan.name,
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "smoke": True,
            "evaluated": evaluated,
            "correct": correct,
            "acc": (correct / evaluated) if evaluated else 0.0,
            "output": str(output_path),
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "task": "averitec_smoke",
                    "mode": "smoke",
                    "total_input_rows": len(rows),
                    "seed_noise": seed,
                },
            ),
        },
        config={
            "dataset": str(dataset),
            "k": k,
            "alpha": alpha,
            "smoke": True,
        },
    )


def run_claimcheckbench_smoke(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    seed: int,
    threshold: float,
    limit: Optional[int],
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    return run_averitec_smoke(
        plan,
        dataset=dataset,
        k=k,
        alpha=alpha,
        seed=seed,
        threshold=threshold,
        limit=limit,
        output_root=output_root,
        label=label,
    )


def run_averitec(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    seed: int,
    limit: Optional[int],
    threshold: float,
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    start = datetime.now(tz=UTC)
    rows = _read_rows(dataset)
    if limit:
        rows = rows[:limit]

    try:
        pool = asyncio.run(get_pool())
        asyncio.run(load_known_tickers(pool))
    except Exception as exc:  # pragma: no cover - infra dependent
        return BenchmarkResult(
            task=plan.name,
            status="skipped",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={"error": str(exc)},
            config={"dataset": str(dataset), "k": k, "alpha": alpha},
            error=str(exc),
        )

    evaluated = 0
    correct = 0
    rows_out: list[dict[str, Any]] = []

    async def _run() -> None:
        nonlocal evaluated, correct
        for row in rows:
            claim_text = row.get("claim") or row.get("claim_text") or row.get("statement")
            gold = _normalize_gold_verdict(row.get("label") or row.get("verdict"))
            if not isinstance(claim_text, str) or not gold:
                continue

            evaluated += 1
            req = ClaimEvaluationRequest(
                source_text=claim_text,
                company_name=row.get("company"),
            )
            claims = normalize_claims(req)

            chunk_batches: list[list[ChunkResult]] = []
            for claim in claims:
                chunks = await retrieve(
                    pool=pool,
                    query=claim.normalized_text,
                    k=k,
                    alpha=alpha,
                    sector=None,
                    company=claim.company_name,
                    filing_type=None,
                    fiscal_year=None,
                )
                chunk_batches.append(chunks)

            verdicts, _ = verify_claims(claims, chunk_batches)
            pred = verdicts[0].verdict if verdicts else "non_verifiable"
            matched = gold == pred
            correct += int(matched)
            rows_out.append(
                {
                    "claim_id": claims[0].claim_id,
                    "gold": gold,
                    "pred": pred,
                    "match": matched,
                }
            )

        await close_pool()

    try:
        asyncio.run(_run())
    except Exception as exc:  # pragma: no cover - infra dependent
        return BenchmarkResult(
            task=plan.name,
            status="failed",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={"error": str(exc)},
            config={"dataset": str(dataset), "k": k, "alpha": alpha},
            error=str(exc),
        )

    output_path = output_root / _run_name(plan.name, label) / "averitec_rows.jsonl"
    _write_jsonl(output_path, rows_out)

    return BenchmarkResult(
        task=plan.name,
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "evaluated": evaluated,
            "correct": correct,
            "acc": (correct / evaluated) if evaluated else 0.0,
            "output": str(output_path),
            "metadata": _build_result_metadata(
                plan,
                dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                hardware=_hardware_fingerprint(),
                extra={
                    "task": "averitec",
                },
            ),
        },
        config={"dataset": str(dataset), "k": k, "alpha": alpha},
    )


def run_claimcheckbench(
    plan: BenchmarkPlan,
    *,
    dataset: Path,
    k: int,
    alpha: float,
    limit: Optional[int],
    seed: int,
    threshold: float,
    output_root: Path,
    label: Optional[str],
) -> BenchmarkResult:
    """Internal benchmark bridge, reusing AVeriTeC-style claim/evidence mapping."""
    return run_averitec(
        plan,
        dataset=dataset,
        k=k,
        alpha=alpha,
        seed=seed,
        limit=limit,
        threshold=threshold,
        output_root=output_root,
        label=label,
    )


# ---------------------------------------------------------------------------
# Dispatch + CLI
# ---------------------------------------------------------------------------


def run_plan(plan: BenchmarkPlan, args: argparse.Namespace) -> BenchmarkResult:
    seed = _coerce_positive_int(args.seed, default=DEFAULT_SEED)
    k = _coerce_positive_int(
        args.k,
        default=_coerce_positive_int(
            plan.default_args.get("k") if plan.default_args else None, default=DEFAULT_K
        ),
    )
    alpha = _coerce_float(
        args.alpha,
        default=_coerce_float(
            plan.default_args.get("alpha") if plan.default_args else None, default=DEFAULT_ALPHA
        ),
    )
    threshold = _coerce_float(
        args.threshold,
        default=_coerce_float(
            plan.default_args.get("threshold") if plan.default_args else None,
            default=DEFAULT_THRESHOLD,
        ),
    )

    dataset = (
        Path(args.dataset)
        if args.dataset is not None
        else Path(plan.dataset)
        if plan.dataset
        else None
    )

    if plan.required_files and not _is_plan_ready(plan):
        return BenchmarkResult(
            task=plan.name,
            status="skipped",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=0.0,
            summary={
                "missing_required_files": [
                    p for p in (plan.required_files or []) if not Path(p).exists()
                ]
            },
            config={"k": k, "alpha": alpha},
            error="missing required files",
        )

    if plan.runner == "financebench":
        if dataset is None:
            return BenchmarkResult(
                task=plan.name,
                status="skipped",
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                elapsed_seconds=0.0,
                summary={"reason": "dataset required"},
                config={"k": k, "alpha": alpha},
                error="dataset not configured",
            )
        if args.smoke:
            return run_financebench_smoke(
                plan,
                dataset=dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                match_quarter=not args.no_quarter_match,
                no_doc_gate=args.no_doc_gate,
                limit=args.limit,
                output_root=args.output_root,
                fusion_strategy=args.fusion_strategy,
                label=args.label,
            )
        return run_financebench(
            plan,
            dataset=dataset,
            k=k,
            alpha=alpha,
            threshold=threshold,
            seed=seed,
            match_quarter=not args.no_quarter_match,
            no_doc_gate=args.no_doc_gate,
            limit=args.limit,
            smoke=args.smoke,
            output_root=args.output_root,
            fusion_strategy=args.fusion_strategy,
            label=args.label,
        )

    if plan.runner == "finrank":
        if dataset is None:
            return BenchmarkResult(
                task=plan.name,
                status="skipped",
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                elapsed_seconds=0.0,
                summary={"reason": "dataset required"},
                config={"k": k, "alpha": alpha},
                error="dataset not configured",
            )
        if args.smoke:
            return run_finrank_smoke(
                plan,
                dataset=dataset,
                k=k,
                alpha=alpha,
                threshold=threshold,
                seed=seed,
                limit=args.limit,
                output_root=args.output_root,
                label=args.label,
            )
        return run_finrank(
            plan,
            dataset=dataset,
            k=k,
            alpha=alpha,
            threshold=threshold,
            seed=seed,
            limit=args.limit,
            output_root=args.output_root,
            label=args.label,
        )

    if plan.runner == "averitec":
        if dataset is None:
            return BenchmarkResult(
                task=plan.name,
                status="skipped",
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                elapsed_seconds=0.0,
                summary={"reason": "dataset required"},
                config={"k": k, "alpha": alpha},
                error="dataset not configured",
            )
        if args.smoke:
            return run_averitec_smoke(
                plan,
                dataset=dataset,
                k=k,
                alpha=alpha,
                seed=seed,
                threshold=threshold,
                limit=args.limit,
                output_root=args.output_root,
                label=args.label,
            )
        return run_averitec(
            plan,
            dataset=dataset,
            k=k,
            alpha=alpha,
            seed=seed,
            limit=args.limit,
            threshold=threshold,
            output_root=args.output_root,
            label=args.label,
        )

    if plan.runner == "claimcheckbench":
        if dataset is None:
            return BenchmarkResult(
                task=plan.name,
                status="skipped",
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                elapsed_seconds=0.0,
                summary={"reason": "dataset required"},
                config={"k": k, "alpha": alpha},
                error="dataset not configured",
            )
        if args.smoke:
            return run_claimcheckbench_smoke(
                plan,
                dataset=dataset,
                k=k,
                alpha=alpha,
                seed=seed,
                threshold=threshold,
                limit=args.limit,
                output_root=args.output_root,
                label=args.label,
            )
        return run_claimcheckbench(
            plan,
            dataset=dataset,
            k=k,
            alpha=alpha,
            seed=seed,
            limit=args.limit,
            threshold=threshold,
            output_root=args.output_root,
            label=args.label,
        )

    if plan.runner == "gsm8k":
        return BenchmarkResult(
            task=plan.name,
            status="skipped",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=0.0,
            summary={
                "reason": (
                    "GSM8K moved to the ai-infra-gsm8k repository on 2026-08-20; "
                    "run it there with scripts/run_gsm8k.py"
                )
            },
            config={"k": k, "alpha": alpha, "seed": seed},
            error="manual-runner-only",
        )

    return BenchmarkResult(
        task=plan.name,
        status="skipped",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=0.0,
        summary={"reason": f"unsupported runner {plan.runner!r}"},
        config={"k": k, "alpha": alpha},
    )


def run_official_benchmarks(
    manifest: OfficialManifest,
    args: argparse.Namespace,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for plan in manifest.runs:
        if args.task and plan.name != args.task:
            continue
        if args.only_planned and not plan.is_planned:
            continue
        result = run_plan(plan, args)
        results.append(result)
        logger.info("%s -> %s", plan.name, result.status)
    return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Official benchmark manifest path",
    )
    parser.add_argument("--task", help="Run only this benchmark by name")
    parser.add_argument(
        "--only-planned",
        action="store_true",
        help="Only run planned benchmarks",
    )
    parser.add_argument("--dataset", type=Path, help="Override dataset path")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label", help="Run label; defaults to timestamp")
    parser.add_argument("--k", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--limit", type=int, help="Evaluate first N rows")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-quarter-match", action="store_true")
    parser.add_argument("--no-doc-gate", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate deterministic smoke snapshot without retrieval dependency.",
    )
    parser.add_argument("--fusion-strategy", choices=["rrf", "minmax"])
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def write_run_result(
    manifest: OfficialManifest,
    args: argparse.Namespace,
    results: list[BenchmarkResult],
) -> None:
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "manifest_version": manifest.version,
        "manifest": str(args.manifest),
        "run_metadata": {
            "seed": _coerce_positive_int(args.seed, default=DEFAULT_SEED),
            "hardware": _hardware_fingerprint(),
            "commit": _git_sha(),
        },
        "results": [asdict(item) for item in results],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.label.strip() if args.label else datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out = args.output_root / f"results_{run_name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep backward compatible pointer for scripts expecting a fixed filename.
    latest = args.output_root / "results.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Alternate interview reporting target used by the execution plan.
    eval_root = Path("evals")
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_out = eval_root / out.name
    eval_latest = eval_root / "results.json"
    eval_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        manifest = load_manifest(args.manifest)
    except Exception as exc:
        print(f"manifest load failed: {exc}", file=sys.stderr)
        return 2

    results = run_official_benchmarks(manifest, args)
    write_run_result(manifest, args, results)

    failed = [result for result in results if result.status in {"failed"}]
    for result in results:
        msg = result.error or "ok"
        print(f"{result.task}: {result.status} ({msg})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
