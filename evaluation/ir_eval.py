#!/usr/bin/env python3
"""Deterministic IR evaluation of the retrieval stage against FinanceBench.

Calls :func:`corpcheck.retrieval.retrieve` in-process — no HTTP server, no LLM.
What is measured is purely whether the retriever surfaces the gold evidence, so
a change in the numbers is attributable to retrieval and nothing else.

Two deliberate choices about fairness:

* **No gold metadata is leaked into the query.** The retriever is given the
  question text only, never the benchmark's ``company`` / ``doc_period`` /
  ``doc_type``. Passing those as filters would measure a system that does not
  exist at serving time.
* **Provenance is checked before content.** A chunk must come from the filing
  the question is about before its token overlap counts. Disable with
  ``--no-doc-gate`` to see how much the gate costs.

The overlap matrix is computed once per query and re-scored at every threshold,
so the sweep is free.

Usage::

    python -m evaluation.ir_eval --label baseline
    python -m evaluation.ir_eval --label rrf --alpha 0.5 --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from corpcheck.db import close_pool, get_pool
from corpcheck.models import ChunkResult
from corpcheck.retrieval import load_known_tickers, retrieve
from corpcheck.retrieval.search import get_model
from evaluation.financebench import GoldDoc, chunk_matches_gold_doc, gold_spans, parse_gold_doc
from evaluation.metrics import (
    THRESHOLD_SWEEP_DEFAULT,
    first_hit_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    token_overlap,
)

logger = logging.getLogger("ir_eval")

DEFAULT_DATASET = Path("evaluation/datasets/financebench_filtered.json")
DEFAULT_OUTPUT_ROOT = Path("evaluation/runs")
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SWEEP = THRESHOLD_SWEEP_DEFAULT
RECALL_CUTOFFS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Per-query evaluation record
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    """Everything needed to re-score a single query without re-retrieving it."""

    financebench_id: str
    question: str
    doc_name: str
    gold_doc: dict[str, Any]
    n_gold_spans: int
    latency_ms: int
    retrieved: list[dict[str, Any]]
    # overlaps[variant][gold_index][rank_index]; variant is "clean" or "raw"
    overlaps: dict[str, list[list[float]]]
    doc_gate_pass: list[bool]
    error: Optional[str] = None


def _chunk_summary(rank: int, chunk: ChunkResult, passes_gate: bool) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "company": chunk.company,
        "filing_type": chunk.filing_type,
        "fiscal_year": chunk.fiscal_year,
        "period_label": chunk.period_label,
        "source_type": chunk.source_type,
        "content_kind": chunk.content_kind,
        "passes_doc_gate": passes_gate,
        "text_preview": chunk.text[:200],
    }


def build_query_record(
    row: dict[str, Any],
    chunks: list[ChunkResult],
    latency_ms: int,
    *,
    match_quarter: bool,
) -> QueryRecord:
    """Score one retrieved list against one benchmark row."""
    gold = parse_gold_doc(row)
    spans = gold_spans(row)

    gate = [chunk_matches_gold_doc(c, gold, match_quarter=match_quarter) for c in chunks]

    overlaps: dict[str, list[list[float]]] = {"clean": [], "raw": []}
    for span in spans:
        overlaps["clean"].append(
            [token_overlap(c.text, span, remove_boilerplate=True) for c in chunks]
        )
        overlaps["raw"].append(
            [
                token_overlap(c.text, span, remove_boilerplate=False, remove_stopwords=False)
                for c in chunks
            ]
        )

    return QueryRecord(
        financebench_id=str(row.get("financebench_id") or ""),
        question=str(row.get("question") or ""),
        doc_name=gold.doc_name,
        gold_doc=asdict(gold),
        n_gold_spans=len(spans),
        latency_ms=latency_ms,
        retrieved=[_chunk_summary(i, c, gate[i - 1]) for i, c in enumerate(chunks, start=1)],
        overlaps=overlaps,
        doc_gate_pass=gate,
    )


# ---------------------------------------------------------------------------
# Scoring (pure — operates on recorded overlaps, never re-retrieves)
# ---------------------------------------------------------------------------


def _masked_overlaps(record: QueryRecord, variant: str, apply_gate: bool) -> list[list[float]]:
    """Return the overlap matrix, zeroing chunks that fail the provenance gate."""
    matrix = record.overlaps[variant]
    if not apply_gate:
        return matrix
    return [
        [value if record.doc_gate_pass[i] else 0.0 for i, value in enumerate(row)]
        for row in matrix
    ]


def score_records(
    records: list[QueryRecord],
    *,
    threshold: float,
    variant: str = "clean",
    apply_gate: bool = True,
    k: int = 10,
) -> dict[str, Any]:
    """Aggregate Recall@k / Hit@k / MRR over all queries at one configuration."""
    scored = [r for r in records if r.error is None]
    per_query: list[dict[str, Any]] = []

    for record in scored:
        matrix = _masked_overlaps(record, variant, apply_gate)
        hit_ranks = [first_hit_rank(row, threshold) for row in matrix]
        per_query.append(
            {
                "financebench_id": record.financebench_id,
                "hit_ranks": hit_ranks,
                "recall": {c: recall_at_k(hit_ranks, c) for c in RECALL_CUTOFFS},
                "hit": {c: hit_at_k(hit_ranks, c) for c in RECALL_CUTOFFS},
                "rr": reciprocal_rank(hit_ranks, k=k),
            }
        )

    summary: dict[str, Any] = {
        "config": {
            "threshold": threshold,
            "variant": variant,
            "doc_gate": apply_gate,
            "k": k,
        },
        "queries_scored": len(per_query),
        "mrr_at_k": round(mean(p["rr"] for p in per_query), 4),
    }
    for cutoff in RECALL_CUTOFFS:
        summary[f"recall_at_{cutoff}"] = round(
            mean(p["recall"][cutoff] for p in per_query), 4
        )
        summary[f"hit_at_{cutoff}"] = round(mean(p["hit"][cutoff] for p in per_query), 4)
    return summary


def gate_diagnostics(records: list[QueryRecord]) -> dict[str, Any]:
    """How often the provenance gate admits anything at all.

    If this is near zero the metric is measuring filing-metadata alignment, not
    retrieval quality — worth knowing before drawing conclusions.
    """
    scored = [r for r in records if r.error is None]
    if not scored:
        return {}
    passed_counts = [sum(r.doc_gate_pass) for r in scored]
    return {
        "queries": len(scored),
        "mean_chunks_passing_gate": round(mean(float(c) for c in passed_counts), 2),
        "queries_with_zero_passing": sum(1 for c in passed_counts if c == 0),
        "queries_with_unresolvable_gold_doc": sum(
            1 for r in scored if not GoldDoc(**r.gold_doc).is_resolvable
        ),
    }


# ---------------------------------------------------------------------------
# Retrieval driver
# ---------------------------------------------------------------------------


async def run_retrieval(
    rows: list[dict[str, Any]],
    *,
    k: int,
    alpha: float,
    match_quarter: bool,
    fusion_strategy: Optional[str] = None,
) -> list[QueryRecord]:
    """Retrieve for every benchmark row and build its evaluation record."""
    pool = await get_pool()
    await load_known_tickers(pool)

    # Load the embedding model up front. It is loaded lazily inside the retrieval
    # path, so a transient failure -- a flaky HuggingFace Hub call, say -- would
    # otherwise be caught by the per-query handler and turn into a silent block of
    # errored queries that are quietly excluded from every metric. Failing here
    # instead makes a broken run impossible to mistake for a bad result.
    get_model()

    records: list[QueryRecord] = []
    try:
        for index, row in enumerate(rows, start=1):
            question = str(row.get("question") or "").strip()
            started = time.perf_counter()
            try:
                # Question text only — no benchmark metadata is passed as a filter.
                chunks = await retrieve(
                    pool=pool,
                    query=question,
                    k=k,
                    alpha=alpha,
                    sector=None,
                    company=None,
                    filing_type=None,
                    fiscal_year=None,
                    fusion_strategy=fusion_strategy,
                )
                latency_ms = round((time.perf_counter() - started) * 1000)
                record = build_query_record(
                    row, chunks, latency_ms, match_quarter=match_quarter
                )
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                latency_ms = round((time.perf_counter() - started) * 1000)
                logger.exception("Retrieval failed for %s", row.get("financebench_id"))
                record = build_query_record(row, [], latency_ms, match_quarter=match_quarter)
                record.error = f"{type(exc).__name__}: {exc}"

            records.append(record)
            logger.info(
                "[%03d/%03d] %s  %d chunks  gate_pass=%d  %dms",
                index,
                len(rows),
                record.financebench_id or "?",
                len(record.retrieved),
                sum(record.doc_gate_pass),
                record.latency_ms,
            )
    finally:
        await close_pool()

    return records



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()


def print_report(records: list[QueryRecord], *, k: int, threshold: float) -> None:
    scored = [r for r in records if r.error is None]
    errored = [r for r in records if r.error is not None]
    latencies = [r.latency_ms for r in scored]

    print()
    print("=" * 78)
    print(f"IR EVALUATION  —  {len(records)} queries, k={k}")
    print("=" * 78)
    if errored:
        print(f"!! {len(errored)} queries errored and are excluded from all metrics")
    if latencies:
        print(
            f"latency ms: mean={statistics.mean(latencies):.0f} "
            f"median={statistics.median(latencies):.0f} max={max(latencies)}"
        )

    diag = gate_diagnostics(records)
    if diag:
        print(
            f"provenance gate: {diag['mean_chunks_passing_gate']}/{k} chunks pass on average; "
            f"{diag['queries_with_zero_passing']} queries admit none; "
            f"{diag['queries_with_unresolvable_gold_doc']} gold docs unresolvable"
        )

    primary = score_records(records, threshold=threshold, variant="clean", apply_gate=True, k=k)
    print()
    print(f"-- PRIMARY  (overlap>={threshold}, boilerplate stripped, provenance gate ON) --")
    print(
        f"  MRR@{k}      {primary['mrr_at_k']:.4f}\n"
        + "\n".join(
            f"  Recall@{c:<2}  {primary[f'recall_at_{c}']:.4f}"
            f"      Hit@{c:<2}  {primary[f'hit_at_{c}']:.4f}"
            for c in RECALL_CUTOFFS
        )
    )

    print()
    print("-- ABLATIONS (Recall@10 / MRR@10) --")
    widths = [34, 12, 12]
    print(_fmt_row(["configuration", f"Recall@{k}", f"MRR@{k}"], widths))
    print(_fmt_row(["-" * 34, "-" * 12, "-" * 12], widths))
    for label, variant, gate in (
        ("clean tokens + gate", "clean", True),
        ("clean tokens, no gate", "clean", False),
        ("raw tokens + gate", "raw", True),
        ("raw tokens, no gate", "raw", False),
    ):
        s = score_records(records, threshold=threshold, variant=variant, apply_gate=gate, k=k)
        print(
            _fmt_row(
                [label, f"{s[f'recall_at_{k}']:.4f}", f"{s['mrr_at_k']:.4f}"],
                widths,
            )
        )

    print()
    print("-- THRESHOLD SENSITIVITY (clean tokens + gate) --")
    widths = [12, 12, 12, 12]
    print(_fmt_row(["threshold", f"Recall@{k}", f"MRR@{k}", f"Hit@{k}"], widths))
    print(_fmt_row(["-" * 12] * 4, widths))
    for t in THRESHOLD_SWEEP:
        s = score_records(records, threshold=t, variant="clean", apply_gate=True, k=k)
        print(
            _fmt_row(
                [
                    f"{t:.2f}",
                    f"{s[f'recall_at_{k}']:.4f}",
                    f"{s['mrr_at_k']:.4f}",
                    f"{s[f'hit_at_{k}']:.4f}",
                ],
                widths,
            )
        )
    print()


def write_run(
    output_dir: Path,
    records: list[QueryRecord],
    *,
    k: int,
    alpha: float,
    threshold: float,
    dataset: Path,
) -> None:
    """Persist per-query detail and the full metric grid for later diffing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    grid = [
        score_records(records, threshold=t, variant=v, apply_gate=g, k=k)
        for t in THRESHOLD_SWEEP
        for v in ("clean", "raw")
        for g in (True, False)
    ]
    summary = {
        "dataset": str(dataset),
        "queries": len(records),
        "errors": sum(1 for r in records if r.error is not None),
        "retrieval_config": {"k": k, "alpha": alpha},
        "primary_threshold": threshold,
        "primary": score_records(
            records, threshold=threshold, variant="clean", apply_gate=True, k=k
        ),
        "gate_diagnostics": gate_diagnostics(records),
        "grid": grid,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {output_dir/'queries.jsonl'} and {output_dir/'summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--label", default="baseline", help="Run name; output goes to evaluation/runs/<label>"
    )
    p.add_argument("--k", type=int, default=10, help="Chunks to retrieve per query")
    p.add_argument("--alpha", type=float, default=0.7, help="Dense/sparse blend weight")
    p.add_argument(
        "--fusion-strategy",
        choices=["rrf", "minmax"],
        default=None,
        help="Fusion strategy to use ('rrf' or 'minmax')",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Token-overlap fraction at which a chunk counts as covering a gold span",
    )
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows")
    p.add_argument(
        "--no-doc-gate",
        action="store_true",
        help="Report the primary metric without the provenance gate",
    )
    p.add_argument(
        "--no-quarter-match",
        action="store_true",
        help="Do not require the quarter to match for 10-Q questions",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print(f"Expected a JSON list in {args.dataset}", file=sys.stderr)
        return 2
    if args.limit is not None:
        rows = rows[: args.limit]

    try:
        records = asyncio.run(
            run_retrieval(
                rows,
                k=args.k,
                alpha=args.alpha,
                match_quarter=not args.no_quarter_match,
                fusion_strategy=args.fusion_strategy,
            )
        )
    except OSError as exc:

        print(
            f"Could not reach Postgres ({exc}). Start the database and retry.",
            file=sys.stderr,
        )
        return 1

    print_report(records, k=args.k, threshold=args.threshold)
    write_run(
        args.output_root / args.label,
        records,
        k=args.k,
        alpha=args.alpha,
        threshold=args.threshold,
        dataset=args.dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
