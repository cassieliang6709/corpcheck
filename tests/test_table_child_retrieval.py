from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

import asyncpg
import pytest

from corpcheck import settings
from corpcheck.retrieval import pipeline
from corpcheck.retrieval.search import table_child_vector_search


def _row(chunk_id: str, score: float, text: str = "parent") -> dict:
    return {
        "chunk_id": chunk_id,
        "company": "TEST",
        "sector": "Technology",
        "filing_type": "10-K",
        "filed_date": date(2025, 1, 1),
        "source_url": "https://example.com/filing",
        "text": text,
        "company_name": "Test Corp",
        "fiscal_year": 2024,
        "period_label": "FY",
        "section_name": "FINANCIAL STATEMENTS",
        "source_type": "sec",
        "content_kind": "table",
        "chunk_strategy": "table",
        "display_title": "Revenue",
        "data_signal_score": 1.0,
        "is_quantitative": True,
        "cos_sim": score,
        "score_v": score,
    }


async def _retrieve(monkeypatch: pytest.MonkeyPatch, query: str):
    calls = {"vector": 0, "bm25": 0, "child": 0}

    async def vector(*args, **kwargs):
        calls["vector"] += 1
        return [_row("base", 0.6, "base parent")]

    async def bm25(*args, **kwargs):
        calls["bm25"] += 1
        return []

    async def child(*args, **kwargs):
        calls["child"] += 1
        return [_row("child-parent", 0.9, "child parent")]

    monkeypatch.setattr(pipeline, "embed_query", lambda _: [0.1, 0.2])
    monkeypatch.setattr(pipeline, "vector_search", vector)
    monkeypatch.setattr(pipeline, "bm25_search", bm25)
    monkeypatch.setattr(pipeline, "table_child_vector_search", child)
    monkeypatch.setattr(pipeline, "REVISION_FILTER_ENABLED", False)
    monkeypatch.setattr(pipeline, "COMPANY_SCOPE_ENABLED", False)

    results = await pipeline.retrieve(None, query, 5, 0.7, None, None, None)
    return calls, results


def test_table_child_flag_defaults_off():
    assert settings.TABLE_CHILD_RETRIEVAL_ENABLED is False


async def test_disabled_path_keeps_existing_search_call_count(monkeypatch):
    monkeypatch.setattr(pipeline, "TABLE_CHILD_RETRIEVAL_ENABLED", False)

    calls, results = await _retrieve(monkeypatch, "What was 2024 revenue?")

    assert calls == {"vector": 1, "bm25": 1, "child": 0}
    assert [result.chunk_id for result in results] == ["base"]


async def test_non_quantitative_query_skips_child_arm(monkeypatch):
    monkeypatch.setattr(pipeline, "TABLE_CHILD_RETRIEVAL_ENABLED", True)

    calls, results = await _retrieve(monkeypatch, "Describe the company strategy")

    assert calls == {"vector": 1, "bm25": 1, "child": 0}
    assert [result.chunk_id for result in results] == ["base"]


async def test_child_parent_enters_dense_candidates(monkeypatch):
    monkeypatch.setattr(pipeline, "TABLE_CHILD_RETRIEVAL_ENABLED", True)

    calls, results = await _retrieve(monkeypatch, "What was 2024 revenue?")

    assert calls == {"vector": 1, "bm25": 1, "child": 1}
    assert [result.chunk_id for result in results] == ["child-parent", "base"]
    assert results[0].text == "child parent"


class _Connection:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.sql = ""
        self.args = ()

    async def fetch(self, sql, *args):
        self.sql = sql
        self.args = args
        if self.error:
            raise self.error
        return self.rows


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


async def test_child_sql_dedupes_parents_and_returns_fixed_limit():
    connection = _Connection([_row("42", 0.91)])

    rows = await table_child_vector_search(_Pool(connection), [0.1], "", {})

    assert rows == [_row("42", 0.91)]
    assert "PARTITION BY chunk_id" in connection.sql
    assert "WHERE parent_rank = 1" in connection.sql
    assert "parent.chunk_id = child.parent_chunk_id::text" in connection.sql
    assert connection.args[1] == 30


async def test_missing_experimental_table_degrades_safely(caplog):
    connection = _Connection(error=asyncpg.UndefinedTableError("missing relation"))

    with caplog.at_level(logging.WARNING):
        rows = await table_child_vector_search(_Pool(connection), [0.1], "", {})

    assert rows == []
    assert "eval_table_child_chunks is missing" in caplog.text
