#!/usr/bin/env python3
"""Build inspectable row-level search children from structured table chunks.

中文：这是可观察的表格行级检索表示实验，输出用于检查切分效果而不进入生产索引；
输入格式不合法会显式失败，避免生成看似有效的行级样本。

This is an evaluation-only representation prototype. It does not write to the
database or change production retrieval.

Usage::

    python -m evaluation.table_child_experiment parents.jsonl
    python -m evaluation.table_child_experiment parents.jsonl --output children.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\(?[-+$€£]?\d[\d,]*(?:\.\d+)?%?\)?)")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_YEAR_LABEL_RE = re.compile(r"\b(?:fiscal|year|years|ended)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TableChild:
    """One searchable table row plus the context needed to interpret it."""

    parent_chunk_id: str
    child_index: int
    table_title: str
    header_rows: tuple[str, ...]
    data_row: str
    text: str


def _payload(line: str, marker: str) -> str:
    return line[len(marker) :].strip()


def _is_year_header(row: str) -> bool:
    """Return whether every pipe-delimited value is a year or empty decoration."""
    cells = [cell.strip().strip("()[]") for cell in row.split("|")]
    populated = [cell for cell in cells if cell]
    if populated and _YEAR_LABEL_RE.search(populated[0]):
        populated = populated[1:]
    return len(populated) >= 2 and all(_YEAR_RE.fullmatch(cell) for cell in populated)


def _is_data_row(row: str) -> bool:
    # A searchable data row needs a label/value or value/value relationship.
    # Standalone dated captions such as "December 31," are context, not facts.
    return "|" in row and bool(_NUMBER_RE.search(row)) and not _is_year_header(row)


def parse_table_children(table_text: str, parent_chunk_id: str) -> list[TableChild]:
    """Parse one ``[TABLE]`` chunk into deduplicated row-level children.

    Malformed blocks, untitled tables, and tables without a numeric data row
    return an empty list. Non-data rows are inherited as rolling context, which
    supports repeated headers in continuation chunks.
    """
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if (
        len(lines) < 3
        or not lines[0].startswith("[TABLE]")
        or lines[-1] != "[/TABLE]"
        or sum(line.startswith("[TABLE]") for line in lines) != 1
    ):
        return []

    title = _payload(lines[0], "[TABLE]")
    if not title:
        return []

    context: list[str] = []
    children: list[TableChild] = []
    seen_texts: set[str] = set()

    for line in lines[1:-1]:
        if line.startswith("[HEADER]"):
            header = _payload(line, "[HEADER]")
            if header:
                context.append(f"[HEADER] {header}")
            continue
        if not line.startswith("[ROW]"):
            return []

        row = _payload(line, "[ROW]")
        if not row:
            continue
        if not _is_data_row(row):
            context_line = f"[ROW] {row}"
            if context_line not in context:
                context.append(context_line)
            continue

        data_line = f"[ROW] {row}"
        child_text = "\n".join([f"[TABLE] {title}", *context, data_line, "[/TABLE]"])
        if child_text in seen_texts:
            continue
        seen_texts.add(child_text)
        children.append(
            TableChild(
                parent_chunk_id=str(parent_chunk_id),
                child_index=len(children),
                table_title=title,
                header_rows=tuple(context),
                data_row=data_line,
                text=child_text,
            )
        )

    return children


def children_from_parent_records(records: Iterable[dict[str, Any]]) -> list[TableChild]:
    """Convert JSON-like parent records, skipping records without usable fields."""
    children: list[TableChild] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        parent_id = record.get("chunk_id", record.get("parent_chunk_id"))
        table_text = record.get("text", record.get("content"))
        if parent_id is None or not isinstance(table_text, str):
            continue
        for child in parse_table_children(table_text, str(parent_id)):
            key = (child.parent_chunk_id, child.text)
            if key not in seen:
                seen.add(key)
                children.append(child)
    return children


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _write_children(children: Iterable[TableChild], stream: TextIO) -> None:
    for child in children:
        stream.write(json.dumps(asdict(child), ensure_ascii=False) + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse JSONL prototype input/output paths without consuming the stream.

    中文：只定义行级实验的 I/O；无效 JSONL 在实际读取时显式失败，避免部分静默转换。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL parent records")
    parser.add_argument("--output", type=Path, help="Output JSONL; defaults to stdout")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Convert parent records to inspectable child rows and return CLI status.

    中文：入口仅写明确指定的实验输出，不修改数据库或生产检索索引。
    """
    args = parse_args(argv)
    try:
        children = children_from_parent_records(_read_jsonl(args.input))
        if args.output:
            with args.output.open("w", encoding="utf-8") as stream:
                _write_children(children, stream)
        else:
            _write_children(children, sys.stdout)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
