"""Collect CorpCheck answer-evaluation predictions sequentially as JSONL.

中文：按固定输入顺序收集预测并写成可审计 JSONL，顺序执行能让失败记录及其
上下文清晰可复现；检索基线不会伪造 LLM 答案。

Retrieval-only mode records the confidence gate without contacting an LLM. Its
``answer`` is therefore empty when the gate passes; this is intentional so the
output remains honest and can be inspected or scored as a retrieval baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Optional

from corpcheck import settings as config
from corpcheck.db import close_pool, get_pool
from corpcheck.llm.chat import chat_once
from corpcheck.models import ChunkResult
from corpcheck.retrieval import load_known_tickers, retrieve
from corpcheck.retrieval.abstain import AbstainDecision, evaluate_answerability

DEFAULT_SEED = Path(__file__).parent / "datasets" / "answer_eval_seed.json"

RetrieveFn = Callable[..., Awaitable[list[ChunkResult]]]
ChatFn = Callable[..., Awaitable[dict[str, Any]]]
AbstainFn = Callable[..., AbstainDecision]


class CollectionError(ValueError):
    """Invalid collector input or configuration."""


def read_seed(path: Path) -> list[dict[str, Any]]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CollectionError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CollectionError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(records, list):
        raise CollectionError(f"{path}: seed must be a JSON list")

    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise CollectionError(f"{path}: record {index} must be an object")
        record_id = record.get("id")
        question = record.get("question")
        if not isinstance(record_id, str) or not record_id.strip():
            raise CollectionError(f"{path}: record {index} has invalid 'id'")
        if record_id in seen:
            raise CollectionError(f"{path}: duplicate id {record_id!r}")
        if not isinstance(question, str) or not question.strip():
            raise CollectionError(f"{path}: record {index} has invalid 'question'")
        seen.add(record_id)
    return records


def _serialize_chunks(chunks: Sequence[ChunkResult]) -> list[dict[str, Any]]:
    return [chunk.model_dump(mode="json") for chunk in chunks]


async def collect_predictions(
    records: Sequence[dict[str, Any]],
    pool: Any,
    *,
    retrieval_only: bool,
    use_seed_filters: bool = False,
    k: int = 10,
    alpha: float = 0.7,
    retrieve_fn: RetrieveFn = retrieve,
    abstain_fn: AbstainFn = evaluate_answerability,
    chat_fn: ChatFn = chat_once,
) -> list[dict[str, Any]]:
    """Collect one prediction at a time, preserving seed order."""
    predictions: list[dict[str, Any]] = []
    for record in records:
        question = record["question"]
        chunks = await retrieve_fn(
            pool=pool,
            query=question,
            k=k,
            alpha=alpha,
            sector=record.get("sector") if use_seed_filters else None,
            company=record.get("company") if use_seed_filters else None,
            filing_type=record.get("filing_type") if use_seed_filters else None,
            fiscal_year=record.get("year") if use_seed_filters else None,
        )
        if use_seed_filters and record.get("company") is not None:
            decision = abstain_fn(
                question,
                chunks,
                expected_company=record["company"],
            )
        else:
            decision = abstain_fn(question, chunks)
        answer = decision.reason if decision.abstain else ""
        thinking: Optional[str] = None

        if not retrieval_only and not decision.abstain:
            result = await chat_fn(question, chunks)
            answer = str(result.get("answer") or "")
            thinking_value = result.get("thinking")
            thinking = str(thinking_value) if thinking_value is not None else None

        predictions.append(
            {
                "id": record["id"],
                "answer": answer,
                "thinking": thinking,
                "abstained": decision.abstain,
                "abstain_reason": decision.reason or None,
                "chunks": _serialize_chunks(chunks),
                "diagnostics": {
                    "mode": "retrieval-only" if retrieval_only else "answer",
                    "top1_cosine": decision.top1,
                    "mean_top3_cosine": decision.mean_top3,
                    "gate_status": decision.status,
                },
            }
        )
    return predictions


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(rendered, encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse sequential prediction-collection settings without contacting services.

    中文：这里不初始化数据库或 LLM；连接、输入和输出问题由执行阶段明确报告。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--use-seed-filters",
        action="store_true",
        help="Diagnostic oracle mode: pass seed metadata as retrieval filters",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    pool = await get_pool()
    try:
        await load_known_tickers(pool)
        predictions = await collect_predictions(
            records,
            pool,
            retrieval_only=args.retrieval_only,
            use_seed_filters=args.use_seed_filters,
            k=args.k,
            alpha=args.alpha,
        )
        write_jsonl(predictions, args.output)
    finally:
        await close_pool()


def main(argv: Optional[list[str]] = None) -> int:
    """Run asynchronous collection behind a synchronous CLI exit-code boundary.

    中文：入口保留 JSONL 收集器的失败信息，避免在评测数据不完整时静默写出结果。
    """
    args = parse_args(argv)
    try:
        records = read_seed(args.seed)
        if not args.retrieval_only and not config.SGLANG_BASE_URL:
            raise CollectionError(
                "SGLANG_BASE_URL is unset; configure the LLM endpoint or use --retrieval-only"
            )
        asyncio.run(_run(args, records))
    except CollectionError as exc:
        print(f"answer prediction collection error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
