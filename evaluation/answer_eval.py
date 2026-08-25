"""Deterministic end-to-end answer evaluation with no model or database calls.

中文：这是一个纯离线的答案评分器。它将答案正确性、拒答和引文支持拆开
测量，避免模型或数据库状态影响可复现的评测结论。

Predictions are JSONL records with ``id``, ``answer``, ``abstained``, and a
``chunks`` list, matching CorpCheck's non-streaming chat response.  Citations in
``answer`` use 1-based indices (for example ``[1]`` or ``[1, 2]``) into that
list. Items may be chunk-id strings or objects containing ``chunk_id``.

Gold is a JSON list or JSONL.  Each record has ``id`` and ``should_abstain``.
Answerable records must also supply at least one of:

* ``accepted_answers``: exact matches after case/punctuation/space normalisation.
* ``expected_values``: strings that must occur after normalisation, or numbers
  that must occur as numeric tokens.  All expected values must be present.

Optional ``supporting_chunk_ids`` enable a conservative supported-claim proxy:
the proxy passes only when at least one citation is present and every valid
citation points to a gold supporting chunk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from evaluation.metrics import mean, normalize_text

_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_NUMBER_BODY = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_RE = re.compile(
    rf"""(?x)
    (?<![\w.])
    (?:
        \(\s*\$?\s*(?P<accounting_number>{_NUMBER_BODY})\s*\)
        |
        (?P<leading_sign>[-+]?)\s*\$?\s*(?P<currency_sign>[-+]?)
        (?P<plain_number>{_NUMBER_BODY})
    )
    (?![\w.])
    """
)
_FINAL_SECTION_RES = (
    re.compile(
        r"(?im)^[ \t]{0,3}#{1,6}[ \t]+"
        r"(?:final[ \t]+answer|answer|conclusion)\b[ \t]*:?[ \t]*"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:\*\*)?final[ \t]+answer(?:\*\*)?[ \t]*:[ \t]*"
    ),
)
_DIRECT_ANSWER_MAX_CHARS = 280


class RecordError(ValueError):
    """A malformed or inconsistent evaluation input."""


def _require(record: dict[str, Any], key: str, expected_type: type, *, source: str) -> Any:
    value = record.get(key)
    if not isinstance(value, expected_type):
        raise RecordError(f"{source}: field {key!r} must be {expected_type.__name__}")
    return value


def _read_records(path: Path, *, jsonl_only: bool = False) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordError(f"cannot read {path}: {exc}") from exc

    records: Any
    if not jsonl_only and path.suffix.lower() == ".json":
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RecordError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(records, list):
            raise RecordError(f"{path}: JSON input must be a list of records")
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RecordError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise RecordError(f"{path}: record {index} must be an object")
    return records


def _index_by_id(records: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise RecordError(f"{source}: record {position} field 'id' must be a non-empty string")
        if record_id in indexed:
            raise RecordError(f"{source}: duplicate id {record_id!r}")
        indexed[record_id] = record
    return indexed


def extract_citation_indices(answer: str) -> list[int]:
    """Extract bracketed 1-based citation indices in appearance order."""
    return [
        int(index)
        for match in _CITATION_RE.finditer(answer)
        for index in re.split(r"\s*,\s*", match.group(1))
    ]


def _chunk_ids(chunks: list[Any], *, source: str) -> list[str]:
    chunk_ids: list[str] = []
    for index, item in enumerate(chunks, start=1):
        if isinstance(item, str):
            chunk_id = item
        elif isinstance(item, dict):
            chunk_id = item.get("chunk_id")
        else:
            chunk_id = None
        if not isinstance(chunk_id, str) or not chunk_id:
            raise RecordError(
                f"{source}: chunk item {index} must be a chunk-id string "
                "or object with string 'chunk_id'"
            )
        chunk_ids.append(chunk_id)
    return chunk_ids


def _number_tokens(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    for match in _NUMBER_RE.finditer(text):
        accounting_number = match.group("accounting_number")
        token = accounting_number or match.group("plain_number")
        is_negative = accounting_number is not None or "-" in (
            (match.group("leading_sign") or "") + (match.group("currency_sign") or "")
        )
        try:
            value = Decimal(token.replace(",", ""))
            values.add(-value if is_negative else value)
        except InvalidOperation:  # pragma: no cover - regex only emits decimal forms
            pass
    return values


def _expected_value_scope(answer: str) -> str:
    """Return the explicit or inferred final-answer portion of a response."""
    markers = [match for pattern in _FINAL_SECTION_RES for match in pattern.finditer(answer)]
    if markers:
        stripped = answer[max(markers, key=lambda match: match.start()).end() :].strip()
    else:
        stripped = answer.strip()
        if len(stripped) <= _DIRECT_ANSWER_MAX_CHARS:
            return stripped

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs[-1]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    return sentences[-1] if sentences else stripped


def answer_matches(answer: str, gold: dict[str, Any], *, source: str) -> bool:
    """Score explicit accepted answers and expected values deterministically."""
    accepted = gold.get("accepted_answers", [])
    expected = gold.get("expected_values", [])
    if not isinstance(accepted, list) or not all(isinstance(value, str) for value in accepted):
        raise RecordError(f"{source}: 'accepted_answers' must be a list of strings")
    if not isinstance(expected, list) or any(
        isinstance(value, bool) or not isinstance(value, (str, int, float)) for value in expected
    ):
        raise RecordError(f"{source}: 'expected_values' must contain only strings or numbers")
    if not accepted and not expected:
        raise RecordError(
            f"{source}: answerable gold must supply 'accepted_answers' or 'expected_values'"
        )

    # The model-facing answer necessarily includes citation markers, while the
    # accepted answer is semantic text.  Do not make gold labels repeat them.
    semantic_answer = _CITATION_RE.sub("", answer)
    normalized = normalize_text(semantic_answer)
    accepted_match = any(normalized == normalize_text(candidate) for candidate in accepted)
    expected_scope = _CITATION_RE.sub("", _expected_value_scope(answer))
    normalized_expected_scope = normalize_text(expected_scope)
    numeric_tokens = _number_tokens(expected_scope)
    expected_match = bool(expected) and all(
        Decimal(str(value)) in numeric_tokens
        if isinstance(value, (int, float))
        else normalize_text(value) in normalized_expected_scope
        for value in expected
    )
    return accepted_match or expected_match


def score_record(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Return per-record booleans plus citation diagnostics."""
    record_id = gold["id"]
    gold_source = f"gold id {record_id!r}"
    pred_source = f"prediction id {record_id!r}"
    should_abstain = _require(gold, "should_abstain", bool, source=gold_source)
    answer = _require(prediction, "answer", str, source=pred_source)
    abstained = _require(prediction, "abstained", bool, source=pred_source)
    chunks = _require(prediction, "chunks", list, source=pred_source)
    chunk_ids = _chunk_ids(chunks, source=pred_source)

    cited_indices = extract_citation_indices(answer)
    unique_indices = list(dict.fromkeys(cited_indices))
    invalid_indices = [index for index in unique_indices if index < 1 or index > len(chunk_ids)]
    citation_present = bool(unique_indices)
    citation_indices_valid = citation_present and not invalid_indices

    support_proxy: Optional[bool] = None
    support_precision: Optional[float] = None
    if "supporting_chunk_ids" in gold:
        supporting = gold["supporting_chunk_ids"]
        if not isinstance(supporting, list) or not all(
            isinstance(value, str) for value in supporting
        ):
            raise RecordError(f"{gold_source}: 'supporting_chunk_ids' must be a list of strings")
        supporting_set = set(supporting)
        valid_cited_ids = [
            chunk_ids[index - 1] for index in unique_indices if 1 <= index <= len(chunk_ids)
        ]
        support_precision = (
            sum(chunk_id in supporting_set for chunk_id in valid_cited_ids) / len(valid_cited_ids)
            if valid_cited_ids
            else 0.0
        )
        support_proxy = bool(valid_cited_ids) and not invalid_indices and support_precision == 1.0

    correct_abstention = abstained == should_abstain
    answer_correct: Optional[bool] = None
    if not should_abstain:
        answer_correct = False if abstained else answer_matches(answer, gold, source=gold_source)

    if should_abstain:
        end_to_end_pass = correct_abstention
    else:
        end_to_end_pass = bool(
            correct_abstention
            and answer_correct
            and citation_present
            and citation_indices_valid
            and (support_proxy is not False)
        )

    return {
        "id": record_id,
        "should_abstain": should_abstain,
        "abstained": abstained,
        "correct_abstention": correct_abstention,
        "answer_correct": answer_correct,
        "citation_present": citation_present,
        "citation_indices_valid": citation_indices_valid,
        "cited_indices": unique_indices,
        "invalid_citation_indices": invalid_indices,
        "support_proxy": support_proxy,
        "support_precision": support_precision,
        "end_to_end_pass": end_to_end_pass,
    }


def evaluate(
    predictions: list[dict[str, Any]], gold_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate aligned datasets and return per-record and aggregate scores."""
    prediction_by_id = _index_by_id(predictions, source="predictions")
    gold_by_id = _index_by_id(gold_records, source="gold")
    missing = sorted(set(gold_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(gold_by_id))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing prediction ids: {missing}")
        if extra:
            parts.append(f"unexpected prediction ids: {extra}")
        raise RecordError("; ".join(parts))

    records = [
        score_record(prediction_by_id[record_id], gold)
        for record_id, gold in gold_by_id.items()
    ]
    answerable = [record for record in records if not record["should_abstain"]]
    support_scored = [record for record in answerable if record["support_proxy"] is not None]

    summary = {
        "records": len(records),
        "answerable_records": len(answerable),
        "abstention_accuracy": mean(float(record["correct_abstention"]) for record in records),
        "answer_accuracy": mean(float(record["answer_correct"]) for record in answerable),
        "citation_presence_rate": mean(float(record["citation_present"]) for record in answerable),
        "citation_index_validity_rate": mean(
            float(record["citation_indices_valid"]) for record in answerable
        ),
        "support_proxy_records": len(support_scored),
        "supported_claim_proxy_rate": mean(
            float(record["support_proxy"]) for record in support_scored
        ),
        "end_to_end_pass_rate": mean(float(record["end_to_end_pass"]) for record in records),
    }
    return {"summary": summary, "records": records}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse offline answer-evaluation inputs without loading predictions.

    中文：仅构造评测配置，不读取或改写结果文件；参数不完整时由 argparse 在执行前退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL prediction file")
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSON list or JSONL file")
    parser.add_argument("--output", type=Path, help="Also write the full JSON result to this path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run deterministic answer scoring and return a process-style status code.

    中文：CLI 入口只协调读取、评分和报告；无效记录或文件错误会保持可见而非产出部分分数。
    """
    args = parse_args(argv)
    try:
        result = evaluate(
            _read_records(args.predictions, jsonl_only=True),
            _read_records(args.gold),
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except RecordError as exc:
        print(f"answer-eval input error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
