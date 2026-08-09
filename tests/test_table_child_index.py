from __future__ import annotations

import numpy as np
import pytest
from evaluation.table_child_index import (
    SCHEMA_STATEMENTS,
    TRACKER_UPSERT_SQL,
    UPSERT_SQL,
    IndexStats,
    build_index,
    build_parent_query,
    parse_args,
    prepare_child_rows,
    prepare_tracker_rows,
    rebuild_scope,
)


def _table(label: str = "Revenue") -> str:
    return f"""[TABLE] Statement
[ROW] 2023 | 2022
[ROW] {label} | 100 | 90
[/TABLE]"""


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.ones((len(texts), 384), dtype=np.float32)


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self.rowcount = 0
        self._rows: list[tuple[int, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=()):
        self.conn.executions.append((sql, params))
        if "SELECT c.id, c.content" in sql:
            after_id = params[0]
            fetch_size = params[-1]
            self._rows = [row for row in self.conn.parents if row[0] > after_id][:fetch_size]
        elif sql.lstrip().startswith("DELETE"):
            self.rowcount = self.conn.deleted_rows

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, parents=(), deleted_rows=0) -> None:
        self.parents = list(parents)
        self.deleted_rows = deleted_rows
        self.executions: list[tuple[str, tuple]] = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_schema_has_exact_contract_and_search_indexes() -> None:
    schema = "\n".join(SCHEMA_STATEMENTS)

    assert "CREATE TABLE IF NOT EXISTS eval_table_child_chunks" in schema
    assert "parent_chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE" in schema
    assert "child_index INTEGER NOT NULL" in schema
    assert "content TEXT NOT NULL" in schema
    assert "embedding vector(384)" in schema
    assert "content_tsv TSVECTOR" in schema
    assert "UNIQUE (parent_chunk_id, child_index)" in schema
    assert "USING hnsw (embedding vector_cosine_ops)" in schema
    assert "USING GIN (content_tsv)" in schema
    assert "(parent_chunk_id)" in schema
    assert "CREATE TABLE IF NOT EXISTS eval_table_child_indexed_parents" in schema
    assert "parent_chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE" in schema
    assert "child_count INTEGER NOT NULL" in schema
    assert "indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in schema


def test_parent_query_is_resumable_and_scoped() -> None:
    sql, params = build_parent_query(["AAPL"], [2022, 2023], after_id=41, fetch_size=8)

    assert "c.content_kind = 'table'" in sql
    assert "c.id > %s" in sql
    assert "NOT EXISTS" in sql
    assert "eval_table_child_indexed_parents indexed" in sql
    assert "indexed.parent_chunk_id = c.id" in sql
    assert "c.ticker = ANY(%s)" in sql
    assert "c.fiscal_year = ANY(%s)" in sql
    assert "ORDER BY c.id" in sql
    assert params == (41, ["AAPL"], [2022, 2023], 8)


def test_cli_requires_an_explicit_scope_or_all() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--all", "--tickers", "AAPL"])

    scoped = parse_args(["--tickers", "aapl", "msft", "--years", "2023"])
    assert scoped.tickers == ["AAPL", "MSFT"]
    assert scoped.years == [2023]
    assert parse_args(["--all"]).all is True


def test_prepare_child_rows_batches_embedding_and_counts_empty_parents() -> None:
    embedder = FakeEmbedder()

    rows, with_children, without_children = prepare_child_rows(
        [(10, _table()), (11, "not a table")], embedder
    )

    assert (with_children, without_children) == (1, 1)
    assert len(rows) == 1
    assert rows[0]["parent_chunk_id"] == 10
    assert rows[0]["child_index"] == 0
    assert rows[0]["embedding"].shape == (384,)
    assert embedder.calls == [[rows[0]["content"]]]

    tracker_rows = prepare_tracker_rows([(10, _table()), (11, "not a table")], rows)
    assert tracker_rows == [
        {"parent_chunk_id": 10, "child_count": 1},
        {"parent_chunk_id": 11, "child_count": 0},
    ]


def test_build_index_uses_parent_batches_upserts_and_limit() -> None:
    conn = FakeConnection([(1, _table("Revenue")), (2, "malformed"), (3, _table("Cash"))])
    embedder = FakeEmbedder()
    writes: list[tuple[str, list[dict], int]] = []

    def fake_execute_batch(cursor, sql, rows, page_size):
        writes.append((sql, list(rows), page_size))

    stats = build_index(
        conn,
        embedder,
        tickers=["AAPL"],
        batch_size=2,
        limit=3,
        batch_writer=fake_execute_batch,
    )

    assert stats == IndexStats(3, 2, 1, 2)
    assert [len(call) for call in embedder.calls] == [1, 1]
    assert [(sql, len(rows)) for sql, rows, _ in writes] == [
        (UPSERT_SQL, 1),
        (TRACKER_UPSERT_SQL, 2),
        (UPSERT_SQL, 1),
        (TRACKER_UPSERT_SQL, 1),
    ]
    assert all(page_size == 2 for _, _, page_size in writes)
    select_params = [params for sql, params in conn.executions if "SELECT c.id" in sql]
    assert select_params == [(0, ["AAPL"], 2), (2, ["AAPL"], 1)]
    assert conn.commits == 3  # schema plus two completed parent batches


def test_scoped_limited_rebuild_clears_children_and_tracker_for_same_parents() -> None:
    conn = FakeConnection(deleted_rows=2)

    deleted = rebuild_scope(conn, ["AAPL"], [2023], limit=5)

    delete_calls = [(sql, params) for sql, params in conn.executions if "DELETE FROM" in sql]
    assert len(delete_calls) == 2
    assert "eval_table_child_chunks" in delete_calls[0][0]
    assert "eval_table_child_indexed_parents" in delete_calls[1][0]
    assert all("c.ticker = ANY(%s)" in sql for sql, _ in delete_calls)
    assert all("c.fiscal_year = ANY(%s)" in sql for sql, _ in delete_calls)
    assert all(params == (["AAPL"], [2023], 5) for _, params in delete_calls)
    assert deleted == 4
    assert conn.commits == 1


def test_direct_build_rejects_an_accidental_unscoped_run() -> None:
    with pytest.raises(ValueError, match="scope is required"):
        build_index(FakeConnection(), FakeEmbedder())

    stats = build_index(
        FakeConnection(),
        FakeEmbedder(),
        all_scope=True,
        batch_writer=lambda *args, **kwargs: None,
    )
    assert stats == IndexStats()
