from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest
from evaluation import snapshot_sec_corpus_manifest as snapshotter


def filing(
    accession: str = "0000320193-23-000106",
    cik: str = "0000320193",
    ticker: str = "AAPL",
) -> dict:
    return {
        "ticker": ticker,
        "form": "10-K",
        "fiscal_year": 2023,
        "period": "annual",
        "filed_date": dt.date(2023, 11, 3),
        "period_of_report": dt.date(2023, 9, 30),
        "accession": accession,
        "cik": cik,
        "source_url": "https://www.sec.gov/Archives/example-index.htm",
    }


class AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.transaction_options = None
        self.queries = []

    def transaction(self, **options):
        self.transaction_options = options
        return AsyncContext()

    async def fetch(self, sql):
        self.queries.append(sql)
        return self.rows

    async def fetchrow(self, sql):
        self.queries.append(sql)
        return {"chunk_count": 12, "embedded_chunk_count": 11}


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return AsyncContext(self.connection)


def test_build_envelope_is_sorted_deterministic_and_has_raw_submission_urls() -> None:
    later = filing()
    earlier = filing("0001018724-20-000004", "1018724", "AMZN")

    first = snapshotter.build_envelope(
        [later, earlier], chunk_count=20, embedded_chunk_count=19
    )
    second = snapshotter.build_envelope(
        [earlier, later], chunk_count=20, embedded_chunk_count=19
    )

    assert first == second
    assert [row["ticker"] for row in first["filings"]] == ["AAPL", "AMZN"]
    assert first["filings"][1]["full_submission_url"] == (
        "https://www.sec.gov/Archives/edgar/data/1018724/"
        "000101872420000004/0001018724-20-000004.txt"
    )
    assert first["corpus_counts"] == {
        "ticker_count": 2,
        "filing_count": 2,
        "chunk_count": 20,
        "embedded_chunk_count": 19,
    }
    payload = {key: value for key, value in first.items() if key != "digest"}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert first["digest"] == {
        "algorithm": "sha256",
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
    }


@pytest.mark.parametrize(
    ("accession", "cik", "message"),
    [
        ("", "320193", "accession"),
        ("000032019323000106", "320193", "accession"),
        ("0000320193-23-000106", "", "(?i)CIK"),
        ("0000320193-23-000106", "not-a-cik", "CIK"),
    ],
)
def test_invalid_or_missing_accession_and_cik_fail_closed(
    accession, cik, message
) -> None:
    with pytest.raises(snapshotter.ManifestError, match=message):
        snapshotter.build_envelope(
            [filing(accession, cik)], chunk_count=1, embedded_chunk_count=1
        )


def test_accession_prefix_may_be_a_filing_agent_cik() -> None:
    manifest = snapshotter.build_envelope(
        [filing("0000950170-21-000046", "1652044", "GOOGL")],
        chunk_count=1,
        embedded_chunk_count=1,
    )
    assert manifest["filings"][0]["full_submission_url"].startswith(
        "https://www.sec.gov/Archives/edgar/data/1652044/"
    )


def test_duplicate_accession_fails_closed() -> None:
    with pytest.raises(snapshotter.ManifestError, match="duplicate accession"):
        snapshotter.build_envelope(
            [filing(), filing(ticker="OTHER")],
            chunk_count=2,
            embedded_chunk_count=2,
        )


def test_empty_corpus_and_source_url_credentials_fail_closed() -> None:
    with pytest.raises(snapshotter.ManifestError, match="no filings"):
        snapshotter.build_envelope([], chunk_count=0, embedded_chunk_count=0)

    row = filing()
    row["source_url"] = "https://api-user:api-secret@example.invalid/filing"
    with pytest.raises(snapshotter.ManifestError, match="contains credentials"):
        snapshotter.build_envelope([row], chunk_count=1, embedded_chunk_count=1)

    row = filing()
    row["source_url"] = "https://example.invalid/filing"
    with pytest.raises(snapshotter.ManifestError, match="non-SEC source_url"):
        snapshotter.build_envelope([row], chunk_count=1, embedded_chunk_count=1)


@pytest.mark.parametrize("field", ["filed_date", "period_of_report"])
def test_missing_filing_dates_fail_closed(field) -> None:
    row = filing()
    row[field] = None
    with pytest.raises(snapshotter.ManifestError, match=f"invalid {field}"):
        snapshotter.build_envelope([row], chunk_count=1, embedded_chunk_count=1)


def test_invalid_chunk_counts_fail_closed() -> None:
    with pytest.raises(snapshotter.ManifestError, match="chunk counts"):
        snapshotter.build_envelope(
            [filing()], chunk_count=1, embedded_chunk_count=2
        )


async def test_snapshot_uses_one_readonly_repeatable_read_transaction() -> None:
    connection = FakeConnection([filing()])

    manifest = await snapshotter.snapshot_corpus(FakePool(connection))

    assert connection.transaction_options == {
        "isolation": "repeatable_read",
        "readonly": True,
    }
    assert len(connection.queries) == 2
    assert all("SELECT" in query and "INSERT" not in query for query in connection.queries)
    assert manifest["corpus_counts"]["filing_count"] == 1


def test_write_once_is_idempotent_and_refuses_collision(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    snapshotter.write_once(output, "same\n")
    snapshotter.write_once(output, "same\n")

    with pytest.raises(snapshotter.ManifestError, match="refusing to overwrite"):
        snapshotter.write_once(output, "different\n")


async def test_run_configures_read_only_pool_and_never_exposes_dsn(
    monkeypatch,
) -> None:
    captured = {}

    class Pool:
        closed = False

        async def close(self):
            self.closed = True

    pool = Pool()

    async def create_pool(**kwargs):
        captured.update(kwargs)
        return pool

    async def snapshot_corpus(fake_pool):
        assert fake_pool is pool
        return {"ok": True}

    monkeypatch.setattr(snapshotter.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(snapshotter, "snapshot_corpus", snapshot_corpus)
    dsn = "postgresql://secret-user:secret-password@localhost/corpcheck"

    assert await snapshotter.run(dsn) == {"ok": True}
    assert captured == {
        "dsn": dsn,
        "min_size": 1,
        "max_size": 1,
        "server_settings": {"default_transaction_read_only": "on"},
    }
    assert pool.closed


@pytest.mark.parametrize(
    "database_url",
    ["corpcheck", "sqlite:///corpcheck", "postgresql://localhost"],
)
def test_database_url_must_be_explicit(database_url) -> None:
    with pytest.raises(snapshotter.ManifestError, match="database"):
        snapshotter.validate_database_url(database_url)
