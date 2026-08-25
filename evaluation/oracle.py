#!/usr/bin/env python3
"""Compute the achievable ceiling for the IR metric — the suite's own sanity check.

中文：先计算指标在当前切块下可达到的理论上限，用来区分“检索失败”和“指标本身
不可达”；它不调用在线模型，也不修改语料。

A weak-supervision metric can fail for two very different reasons: the retriever
is bad, or the metric is unachievable by construction. If our chunker splits a
gold evidence page into five pieces, no single chunk can ever cover 50% of it and
Recall@10 is pinned at zero no matter how good retrieval gets.

This tool distinguishes those cases. For each gold span it takes the best token
overlap achievable by *any* chunk of the correct filing — perfect oracle
retrieval — and reports how many spans clear each threshold. That number is the
upper bound :mod:`evaluation.ir_eval` can possibly report.

Run it whenever the corpus is rebuilt or the chunking strategy changes.

Usage::

    python -m evaluation.oracle
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

from corpcheck.db import close_pool, get_pool
from evaluation.financebench import GoldDoc, gold_spans, parse_gold_doc
from evaluation.metrics import THRESHOLD_SWEEP_DEFAULT, token_overlap

logger = logging.getLogger("oracle")

DEFAULT_DATASET = Path("evaluation/datasets/financebench_filtered.json")


async def _fetch_filing_chunks(conn: Any, gold: GoldDoc) -> list[str]:
    """Every SEC chunk belonging to the gold filing."""
    sql = """
        SELECT content
        FROM v_retrieval_chunks
        WHERE source_type = 'sec'
          AND ticker = $1
          AND filing_type = $2
          AND fiscal_year = $3
    """
    args: list[Any] = [gold.ticker, gold.filing_type, gold.fiscal_year]
    if gold.quarter:
        sql += " AND period_label = $4"
        args.append(gold.quarter)
    rows = await conn.fetch(sql, *args)
    return [r["content"] for r in rows]


async def compute_ceiling(
    rows: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Best achievable overlap per gold span under oracle filing selection."""
    pool = await get_pool()
    best_overlaps: list[float] = []
    absent_docs: list[str] = []
    unresolvable: list[str] = []

    try:
        async with pool.acquire() as conn:
            for row in rows:
                gold = parse_gold_doc(row)
                if not gold.is_resolvable:
                    unresolvable.append(gold.doc_name)
                    continue

                chunks = await _fetch_filing_chunks(conn, gold)
                if not chunks:
                    absent_docs.append(gold.doc_name)
                    logger.warning(
                        "Gold filing absent from corpus: %s (%s %s FY%s Q%s)",
                        gold.doc_name,
                        gold.ticker,
                        gold.filing_type,
                        gold.fiscal_year,
                        gold.quarter,
                    )
                    continue

                for span in gold_spans(row):
                    best_overlaps.append(max(token_overlap(c, span) for c in chunks))
    finally:
        await close_pool()

    reachable = {
        f"{t:.2f}": sum(1 for b in best_overlaps if b >= t) for t in thresholds
    }
    return {
        "gold_spans_measured": len(best_overlaps),
        "gold_docs_absent_from_corpus": sorted(set(absent_docs)),
        "gold_docs_unresolvable": sorted(set(unresolvable)),
        "best_overlap": {
            "mean": round(statistics.mean(best_overlaps), 4) if best_overlaps else 0.0,
            "median": round(statistics.median(best_overlaps), 4) if best_overlaps else 0.0,
            "min": round(min(best_overlaps), 4) if best_overlaps else 0.0,
            "max": round(max(best_overlaps), 4) if best_overlaps else 0.0,
        },
        "spans_reachable": reachable,
        "ceiling_recall": {
            t: round(c / len(best_overlaps), 4) if best_overlaps else 0.0
            for t, c in reachable.items()
        },
    }


def print_ceiling(result: dict[str, Any]) -> None:
    total = result["gold_spans_measured"]
    print()
    print("=" * 78)
    print(f"METRIC CEILING  —  oracle filing selection, {total} gold spans")
    print("=" * 78)
    bo = result["best_overlap"]
    print(
        f"best achievable overlap: mean={bo['mean']:.3f} median={bo['median']:.3f} "
        f"min={bo['min']:.3f} max={bo['max']:.3f}"
    )
    if result["gold_docs_absent_from_corpus"]:
        print(f"!! gold filings missing from corpus: {result['gold_docs_absent_from_corpus']}")
    if result["gold_docs_unresolvable"]:
        print(f"!! gold docs we cannot map to a ticker: {result['gold_docs_unresolvable']}")
    print()
    print("threshold     reachable     ceiling Recall")
    print("-" * 44)
    for t, count in result["spans_reachable"].items():
        print(f"{t:<12}  {count:>3}/{total:<8}  {result['ceiling_recall'][t]:.4f}")
    print()
    print("Any Recall reported by ir_eval above these values indicates a scoring bug.")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    """Run the asynchronous metric-ceiling check from the command line.

    中文：入口计算当前语料的可达上限；数据库或语料不满足前提时不应把结果当作有效基线。
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    try:
        result = asyncio.run(compute_ceiling(rows, thresholds=THRESHOLD_SWEEP_DEFAULT))
    except OSError as exc:
        print(f"Could not reach Postgres ({exc}). Start the database and retry.", file=sys.stderr)
        return 1

    print_ceiling(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
