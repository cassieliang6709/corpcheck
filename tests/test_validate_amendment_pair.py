from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation import validate_amendment_pair as runner

TARGET_URL = "postgresql://postgres:postgres@localhost:5432/gme_amendment_validation"
BENCHMARK_URL = "postgresql://postgres:postgres@localhost:5432/financial_rag"
REAL_USER_AGENT = "CorpCheck research@corpcheck.org"


def fixture() -> dict:
    return runner.load_fixture()


def discovered_meta(accession: str, path: str = "/tmp/filing.html") -> tuple:
    return (
        "GME",
        "10-K",
        2024,
        "annual",
        Path(path),
        "",
        None,
        None,
        accession,
        "0001326380",
    )


def test_cli_requires_explicit_target_database_and_download_directory() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([])
    with pytest.raises(SystemExit):
        runner.parse_args(["--database-url", TARGET_URL])
    with pytest.raises(SystemExit):
        runner.parse_args(["--download-dir", "/tmp/gme-validation"])


@pytest.mark.parametrize(
    ("target", "download_dir", "user_agent", "benchmark", "message"),
    [
        (BENCHMARK_URL, "/tmp/gme", REAL_USER_AGENT, BENCHMARK_URL, "refusing benchmark"),
        (TARGET_URL, "data/sec_filings", REAL_USER_AGENT, BENCHMARK_URL, "default"),
        (
            TARGET_URL,
            "/tmp/gme",
            "corpcheck-pipeline your-email@example.com",
            BENCHMARK_URL,
            "SEC_USER_AGENT",
        ),
        (TARGET_URL, "/tmp/gme", REAL_USER_AGENT, TARGET_URL, "must differ"),
        (
            TARGET_URL,
            "/tmp/gme",
            REAL_USER_AGENT,
            "postgresql://localhost/not_the_benchmark",
            "must point",
        ),
        (
            TARGET_URL,
            "/tmp/gme",
            "CorpCheck Validation research@corpcheck.org",
            BENCHMARK_URL,
            "exactly",
        ),
    ],
)
def test_isolation_guards_fail_closed(target, download_dir, user_agent, benchmark, message) -> None:
    with pytest.raises(runner.ValidationError, match=message):
        runner.validate_isolation(
            target,
            Path(download_dir),
            user_agent,
            benchmark_database_url=benchmark,
        )


def test_isolation_accepts_distinct_db_temp_dir_and_real_user_agent(tmp_path) -> None:
    runner.validate_isolation(
        TARGET_URL,
        tmp_path / "gme-download",
        REAL_USER_AGENT,
        benchmark_database_url=BENCHMARK_URL,
    )


def test_pristine_target_inspects_catalog_and_rejects_existing_relations(monkeypatch) -> None:
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql):
            assert "pg_catalog.pg_class" in sql
            assert "pg_catalog.pg_namespace" in sql

        def fetchall(self):
            return [("public", "filings", "r")]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_session(self, *, readonly):
            assert readonly is True

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(runner.psycopg2, "connect", lambda url: FakeConnection())

    with pytest.raises(runner.ValidationError, match="pristine database"):
        runner.assert_pristine_target(TARGET_URL)


def test_fixture_contract_rejects_non_distinct_accessions(tmp_path) -> None:
    bad = fixture()
    bad["amendment"]["accession_number"] = bad["original"]["accession_number"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(runner.ValidationError, match="exactly two distinct"):
        runner.load_fixture(path)


def test_exact_pair_metadata_applies_fixture_authoritative_types_and_period() -> None:
    expected = fixture()
    original = expected["original"]["accession_number"]
    amendment = expected["amendment"]["accession_number"]

    pair = runner.exact_pair_metadata(
        [discovered_meta(amendment, "/tmp/amendment.html"), discovered_meta(original)],
        expected,
    )

    assert [(meta[1], meta[2], meta[3], meta[8]) for meta in pair] == [
        ("10-K", 2024, "annual", original),
        ("10-K/A", 2024, "annual", amendment),
    ]
    assert all(meta[7].isoformat() == "2024-02-03" for meta in pair)


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_exact_pair_metadata_rejects_missing_extra_and_duplicate_rows(mode) -> None:
    expected = fixture()
    original = discovered_meta(expected["original"]["accession_number"])
    amendment = discovered_meta(expected["amendment"]["accession_number"])
    rows = {
        "missing": [original],
        "extra": [original, amendment, discovered_meta("unexpected-accession")],
        "duplicate": [original, original],
    }[mode]

    with pytest.raises(runner.ValidationError, match="match the fixture exactly"):
        runner.exact_pair_metadata(rows, expected)


class FakeLibraryDownloader:
    def __init__(self) -> None:
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeSECDownloader:
    def __init__(self, rows) -> None:
        self._downloader = FakeLibraryDownloader()
        self.rows = rows
        self.rate_limited = False
        self.collect_args = None

    def _rate_limit(self):
        self.rate_limited = True

    def _collect_metadata(self, tickers, filing_types, years):
        self.collect_args = (tickers, filing_types, years)
        return self.rows


def test_prepare_pair_requests_base_form_with_amendments_and_exact_filters(tmp_path) -> None:
    expected = fixture()
    rows = [
        discovered_meta(expected["original"]["accession_number"]),
        discovered_meta(expected["amendment"]["accession_number"]),
    ]
    downloader = FakeSECDownloader(rows)

    pair = runner.prepare_pair(tmp_path, expected, downloader=downloader)

    assert len(pair) == 2
    assert downloader.rate_limited is True
    args, kwargs = downloader._downloader.calls[0]
    assert args == ("10-K", "GME")
    assert kwargs["include_amends"] is True
    assert kwargs["download_details"] is True
    assert kwargs["limit"] == 2
    assert downloader.collect_args == (["GME"], ["10-K"], [2024])


def test_prepare_only_never_initializes_or_imports_a_database(monkeypatch, tmp_path) -> None:
    expected = fixture()
    pair = [
        discovered_meta(expected["original"]["accession_number"]),
        discovered_meta(expected["amendment"]["accession_number"]),
    ]
    imported = False

    monkeypatch.setenv("SEC_USER_AGENT", REAL_USER_AGENT)
    monkeypatch.setattr(runner, "load_fixture", lambda path: expected)
    monkeypatch.setattr(runner, "prepare_pair", lambda download_dir, fixture: pair)

    def forbidden_import(*args, **kwargs):
        nonlocal imported
        imported = True

    monkeypatch.setattr(runner, "import_and_validate", forbidden_import)

    result = runner.main(
        [
            "--database-url",
            TARGET_URL,
            "--benchmark-database-url",
            BENCHMARK_URL,
            "--download-dir",
            str(tmp_path / "download"),
            "--prepare-only",
        ]
    )

    assert result == 0
    assert imported is False


def test_import_path_initializes_only_target_and_checks_benchmark_stability(monkeypatch) -> None:
    expected = fixture()
    pair = [discovered_meta(expected["original"]["accession_number"])]
    events = []
    snapshot = runner.BenchmarkSnapshot(50, 1662, 469874, 0, 0)

    class FakeLoader:
        def __init__(self, dsn):
            events.append(("loader", dsn))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def init_schema(self, path):
            events.append(("schema", path))

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: snapshot)
    monkeypatch.setattr(
        runner, "assert_pristine_target", lambda url: events.append(("pristine", url))
    )
    monkeypatch.setattr(runner, "DBLoader", FakeLoader)
    monkeypatch.setattr(
        runner, "assert_empty_target", lambda url: events.append(("empty", url))
    )
    monkeypatch.setattr(runner, "Embedder", lambda batch_size: object())
    monkeypatch.setattr(runner, "_process_filings", lambda *args, **kwargs: 12)
    monkeypatch.setattr(
        runner,
        "assert_imported_pair",
        lambda *args: events.append(("sql", args[0])),
    )
    monkeypatch.setattr(
        runner.asyncio,
        "run",
        lambda coro: (coro.close(), events.append(("revision", TARGET_URL))),
    )

    loaded = runner.import_and_validate(
        TARGET_URL,
        BENCHMARK_URL,
        pair,
        expected,
        batch_size=2,
    )

    assert loaded == 12
    assert events.index(("pristine", TARGET_URL)) < events.index(("loader", TARGET_URL))
    assert ("loader", TARGET_URL) in events
    assert ("empty", TARGET_URL) in events
    assert ("sql", TARGET_URL) in events
    assert ("revision", TARGET_URL) in events


def test_import_fails_if_public_benchmark_snapshot_changes(monkeypatch) -> None:
    before = runner.BenchmarkSnapshot(50, 1662, 469874, 0, 0)
    after = runner.BenchmarkSnapshot(51, 1662, 469900, 0, 0)
    snapshots = iter([before, after])

    class FakeLoader:
        def __init__(self, dsn):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def init_schema(self, path):
            pass

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: next(snapshots))
    monkeypatch.setattr(runner, "assert_pristine_target", lambda url: None)
    monkeypatch.setattr(runner, "DBLoader", FakeLoader)
    monkeypatch.setattr(runner, "assert_empty_target", lambda url: None)
    monkeypatch.setattr(runner, "Embedder", lambda batch_size: object())
    monkeypatch.setattr(runner, "_process_filings", lambda *args, **kwargs: 1)
    monkeypatch.setattr(runner, "assert_imported_pair", lambda *args: None)
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: coro.close())

    with pytest.raises(runner.ValidationError, match="benchmark database changed"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            [],
            fixture(),
            batch_size=2,
        )


def test_import_rejects_an_already_contaminated_public_snapshot(monkeypatch) -> None:
    contaminated = runner.BenchmarkSnapshot(51, 1664, 469900, 2, 1)
    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: contaminated)

    with pytest.raises(runner.ValidationError, match="already contaminated"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            [],
            fixture(),
            batch_size=2,
        )


def test_import_rejects_populated_target_before_constructing_loader(monkeypatch) -> None:
    snapshot = runner.BenchmarkSnapshot(50, 1662, 469874, 0, 0)
    loader_constructed = False

    class ForbiddenLoader:
        def __init__(self, dsn):
            nonlocal loader_constructed
            loader_constructed = True

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: snapshot)
    monkeypatch.setattr(
        runner,
        "assert_pristine_target",
        lambda url: (_ for _ in ()).throw(runner.ValidationError("not pristine")),
    )
    monkeypatch.setattr(runner, "DBLoader", ForbiddenLoader)

    with pytest.raises(runner.ValidationError, match="not pristine"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            [],
            fixture(),
            batch_size=2,
        )

    assert loader_constructed is False
