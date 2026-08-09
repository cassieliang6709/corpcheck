from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from corpcheck.ingestion.config import EMBEDDING_DIM
from corpcheck.ingestion.loaders import db_loader
from corpcheck.ingestion.loaders.db_loader import DBLoader


def _chunk_row(*, filing_id: int = 17, chunk_index: int = 0) -> dict[str, Any]:
    return {
        "filing_id": filing_id,
        "ticker": "TEST",
        "sector": "Technology",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "period": "annual",
        "section_name": "MD&A",
        "chunk_index": chunk_index,
        "content": f"chunk {chunk_index}",
        "char_count": 7,
        "token_count": 2,
        "embedding": [0.0] * EMBEDDING_DIM,
        "source_url": "https://example.test/filing",
    }


class FakeCursor:
    def __init__(self, events: list[Any], *, fail_delete: bool = False, deleted: int = 0) -> None:
        self.events = events
        self.fail_delete = fail_delete
        self.rowcount = deleted

    def execute(self, sql: str, params: Any) -> None:
        self.events.append(("delete", self, sql, params))
        if self.fail_delete:
            raise RuntimeError("delete failed")

    def close(self) -> None:
        self.events.append(("close", self))


class FakeConnection:
    closed = False

    def __init__(self, *, fail_delete: bool = False, deleted: int = 0) -> None:
        self.events: list[Any] = []
        self.cursor_instance = FakeCursor(
            self.events,
            fail_delete=fail_delete,
            deleted=deleted,
        )

    def cursor(self) -> FakeCursor:
        self.events.append(("cursor", self.cursor_instance))
        return self.cursor_instance

    def commit(self) -> None:
        self.events.append(("commit",))

    def rollback(self) -> None:
        self.events.append(("rollback",))


def _loader_with_connection(connection: FakeConnection) -> DBLoader:
    loader = DBLoader("postgresql://unused")
    loader._conn = connection
    return loader


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one row"),
        ([_chunk_row(filing_id=18)], "filing_id must match"),
        ([_chunk_row(chunk_index=0), _chunk_row(chunk_index=0)], "must be unique"),
        ([_chunk_row(chunk_index=0), _chunk_row(chunk_index=2)], "must be contiguous"),
        ([{**_chunk_row(), "embedding": None}], "embedding is required"),
        ([{**_chunk_row(), "embedding": [0.0] * (EMBEDDING_DIM - 1)}], "exactly 384"),
    ],
)
def test_replace_filing_chunks_atomic_validates_before_opening_cursor(
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    connection = FakeConnection()
    loader = _loader_with_connection(connection)

    with pytest.raises(ValueError, match=message):
        loader.replace_filing_chunks_atomic(17, rows)

    assert connection.events == []


def test_replace_filing_chunks_atomic_rejects_invalid_filing_id_before_cursor() -> None:
    connection = FakeConnection()
    loader = _loader_with_connection(connection)

    with pytest.raises(ValueError, match="positive filing_id"):
        loader.replace_filing_chunks_atomic(0, [_chunk_row(filing_id=0)])

    assert connection.events == []


def test_replace_filing_chunks_atomic_upserts_and_prunes_before_one_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(deleted=4)
    loader = _loader_with_connection(connection)
    rows = [_chunk_row(chunk_index=1), _chunk_row(chunk_index=0)]

    def fake_execute_batch(
        cursor: FakeCursor,
        sql: str,
        received_rows: list[dict[str, Any]],
        *,
        page_size: int,
    ) -> None:
        connection.events.append(("upsert", cursor, sql, received_rows, page_size))

    monkeypatch.setattr(db_loader.psycopg2.extras, "execute_batch", fake_execute_batch)

    assert loader.replace_filing_chunks_atomic(17, rows) == 2

    assert [event[0] for event in connection.events] == [
        "cursor",
        "upsert",
        "delete",
        "commit",
        "close",
    ]
    upsert_event = connection.events[1]
    delete_event = connection.events[2]
    assert upsert_event[1] is connection.cursor_instance
    assert delete_event[1] is connection.cursor_instance
    assert delete_event[3] == (17, [1, 0])
    assert "ON CONFLICT (filing_id, chunk_index) DO UPDATE" in upsert_event[2]
    assert "DELETE FROM chunks" in delete_event[2]
    assert all(row["embedding"].shape == (EMBEDDING_DIM,) for row in upsert_event[3])
    assert all(row["embedding"].dtype == np.float32 for row in upsert_event[3])


def test_replace_filing_chunks_atomic_rolls_back_when_upsert_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()
    loader = _loader_with_connection(connection)

    def failing_execute_batch(*args: Any, **kwargs: Any) -> None:
        connection.events.append(("upsert",))
        raise RuntimeError("upsert failed")

    monkeypatch.setattr(db_loader.psycopg2.extras, "execute_batch", failing_execute_batch)

    with pytest.raises(RuntimeError, match="upsert failed"):
        loader.replace_filing_chunks_atomic(17, [_chunk_row()])

    assert [event[0] for event in connection.events] == ["cursor", "upsert", "rollback", "close"]


def test_replace_filing_chunks_atomic_rolls_back_upsert_when_prune_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(fail_delete=True)
    loader = _loader_with_connection(connection)

    def fake_execute_batch(*args: Any, **kwargs: Any) -> None:
        connection.events.append(("upsert",))

    monkeypatch.setattr(db_loader.psycopg2.extras, "execute_batch", fake_execute_batch)

    with pytest.raises(RuntimeError, match="delete failed"):
        loader.replace_filing_chunks_atomic(17, [_chunk_row()])

    assert [event[0] for event in connection.events] == [
        "cursor",
        "upsert",
        "delete",
        "rollback",
        "close",
    ]
