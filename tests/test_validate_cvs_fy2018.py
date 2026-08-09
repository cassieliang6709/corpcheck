from __future__ import annotations

from pathlib import Path

import pytest
from evaluation import validate_cvs_fy2018 as runner
from evaluation.financebench import gold_spans

from corpcheck.models import ChunkResult

TARGET_URL = "postgresql://postgres:postgres@localhost:5432/cvs_fy2018_validation"
BENCHMARK_URL = "postgresql://postgres:postgres@localhost:5432/financial_rag"
REAL_USER_AGENT = "CorpCheck research@corpcheck.org"


def discovered_meta(**changes) -> tuple:
    values = [
        runner.TICKER,
        runner.FILING_TYPE,
        runner.FISCAL_YEAR,
        runner.PERIOD,
        Path("/tmp/cvs.html"),
        "https://www.sec.gov/example",
        runner.FILED_DATE,
        runner.PERIOD_OF_REPORT,
        runner.ACCESSION,
        runner.CIK,
    ]
    indexes = {
        "ticker": 0,
        "filing_type": 1,
        "fiscal_year": 2,
        "period": 3,
        "filed_date": 6,
        "period_of_report": 7,
        "accession": 8,
        "cik": 9,
    }
    for name, value in changes.items():
        values[indexes[name]] = value
    return tuple(values)


def chunk(text: str, *, year: int = runner.FISCAL_YEAR) -> ChunkResult:
    return ChunkResult(
        chunk_id="chunk",
        text=text,
        score=1.0,
        company=runner.TICKER,
        filing_type=runner.FILING_TYPE,
        fiscal_year=year,
        period_label=runner.PERIOD,
    )


def test_cli_requires_explicit_target_database_and_download_directory() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([])
    with pytest.raises(SystemExit):
        runner.parse_args(["--database-url", TARGET_URL])
    with pytest.raises(SystemExit):
        runner.parse_args(["--download-dir", "/tmp/cvs-validation"])


@pytest.mark.parametrize(
    ("target", "download_dir", "message"),
    [
        (BENCHMARK_URL, "/tmp/cvs", "refusing benchmark"),
        (TARGET_URL, "data/sec_filings", "default"),
    ],
)
def test_reused_isolation_guards_reject_benchmark_and_default_dir(
    target, download_dir, message
) -> None:
    with pytest.raises(runner.ValidationError, match=message):
        runner.validate_isolation(
            target,
            Path(download_dir),
            REAL_USER_AGENT,
            benchmark_database_url=BENCHMARK_URL,
        )


def test_download_directory_must_be_pristine(tmp_path) -> None:
    runner.assert_pristine_download_dir(tmp_path / "absent")
    runner.assert_pristine_download_dir(tmp_path)
    (tmp_path / "old-filing.txt").write_text("old", encoding="utf-8")

    with pytest.raises(runner.ValidationError, match="absent or empty"):
        runner.assert_pristine_download_dir(tmp_path)


def test_exact_filing_metadata_accepts_only_locked_accession() -> None:
    selected = runner.exact_filing_metadata(
        [discovered_meta(accession="unrelated"), discovered_meta(cik="64803")]
    )

    assert selected[8] == runner.ACCESSION
    assert selected[2:4] == (2018, "annual")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"accession": "wrong"}, "exactly one"),
        ({"filing_type": "10-Q"}, "locked CVS"),
        ({"fiscal_year": 2019}, "locked CVS"),
        ({"filed_date": None}, "locked CVS"),
        ({"period_of_report": None}, "locked CVS"),
        ({"cik": "123"}, "locked CVS"),
    ],
)
def test_exact_filing_metadata_rejects_wrong_fixture(changes, message) -> None:
    with pytest.raises(runner.ValidationError, match=message):
        runner.exact_filing_metadata([discovered_meta(**changes)])


class FakeLibraryDownloader:
    def __init__(self) -> None:
        self.calls = []

    def get(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class FakeSECDownloader:
    def __init__(self, rows) -> None:
        self._downloader = FakeLibraryDownloader()
        self.rows = rows
        self.rate_limited = False
        self.collect_args = None

    def _rate_limit(self) -> None:
        self.rate_limited = True

    def _collect_metadata(self, tickers, filing_types, years):
        self.collect_args = (tickers, filing_types, years)
        return self.rows


def test_prepare_filing_uses_narrow_window_and_exact_selection(tmp_path) -> None:
    downloader = FakeSECDownloader([discovered_meta()])

    filing = runner.prepare_filing(tmp_path, downloader=downloader)

    assert filing[8] == runner.ACCESSION
    assert downloader.rate_limited is True
    args, kwargs = downloader._downloader.calls[0]
    assert args == ("10-K", "CVS")
    assert kwargs == {
        "after": "2019-02-01",
        "before": "2019-03-31",
        "limit": 1,
        "include_amends": False,
        "download_details": True,
    }
    assert downloader.collect_args == (["CVS"], ["10-K"], [2018])


def test_main_rejects_non_two_token_user_agent_before_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "CorpCheck Validation research@corpcheck.org")
    monkeypatch.setattr(
        runner,
        "prepare_filing",
        lambda *args, **kwargs: pytest.fail("download must not start"),
    )

    with pytest.raises(runner.ValidationError, match="exactly"):
        runner.main(
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


def test_prepare_only_never_imports_or_connects_to_database(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", REAL_USER_AGENT)
    monkeypatch.setattr(runner, "prepare_filing", lambda path: discovered_meta())
    monkeypatch.setattr(
        runner,
        "import_and_validate",
        lambda *args, **kwargs: pytest.fail("database import must not start"),
    )

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


def test_imported_filing_requires_financial_statement_values(monkeypatch) -> None:
    filings = [
        (
            runner.TICKER,
            runner.FILING_TYPE,
            runner.FISCAL_YEAR,
            runner.PERIOD,
            runner.FILED_DATE,
            runner.PERIOD_OF_REPORT,
            runner.ACCESSION,
            "64803",
        )
    ]
    result_sets = iter(
        [
            filings,
            [
                ("Total revenues 194,579", True),
                ("Property and equipment, net 11,349 10,292", True),
            ],
        ]
    )

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            if params is not None:
                assert params == (runner.ACCESSION,)
                assert "Financial Statements" in sql

        def fetchall(self):
            return next(result_sets)

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

    runner.assert_imported_filing(TARGET_URL)


def test_imported_filing_rejects_missing_value(monkeypatch) -> None:
    filing = (
        runner.TICKER,
        runner.FILING_TYPE,
        runner.FISCAL_YEAR,
        runner.PERIOD,
        runner.FILED_DATE,
        runner.PERIOD_OF_REPORT,
        runner.ACCESSION,
        runner.CIK,
    )
    result_sets = iter([[filing], [("Total revenues 194,579", True)]])

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            pass

        def fetchall(self):
            return next(result_sets)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_session(self, *, readonly):
            pass

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(runner.psycopg2, "connect", lambda url: FakeConnection())

    with pytest.raises(runner.ValidationError, match="11,349"):
        runner.assert_imported_filing(TARGET_URL)


def test_strict_overlap_requires_both_gold_spans_with_provenance() -> None:
    benchmark_case = runner.load_benchmark_case()
    spans = gold_spans(benchmark_case)

    assert runner.assert_strict_overlap([chunk(spans[0]), chunk(spans[1])], benchmark_case) == [
        1,
        2,
    ]

    with pytest.raises(runner.ValidationError, match="Recall@10 failed"):
        runner.assert_strict_overlap(
            [chunk(spans[0], year=2019), chunk(spans[1], year=2019)],
            benchmark_case,
        )


def test_import_path_checks_benchmark_before_and_after(monkeypatch) -> None:
    snapshot = runner.BenchmarkSnapshot(50, 1662, 469874, 0, 0)
    snapshots = iter([snapshot, snapshot])
    events = []

    class FakeLoader:
        def __init__(self, dsn):
            events.append(("loader", dsn))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def init_schema(self, path):
            events.append(("schema", path))

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: next(snapshots))
    monkeypatch.setattr(
        runner, "assert_pristine_target", lambda url: events.append(("pristine", url))
    )
    monkeypatch.setattr(runner, "DBLoader", FakeLoader)
    monkeypatch.setattr(runner, "assert_empty_target", lambda url: None)
    monkeypatch.setattr(runner, "Embedder", lambda batch_size: object())
    monkeypatch.setattr(runner, "_process_filings", lambda *args, **kwargs: 17)
    monkeypatch.setattr(runner, "assert_imported_filing", lambda url: None)
    monkeypatch.setattr(runner, "load_benchmark_case", lambda: {"question": "question"})

    async def fake_retrieval(database_url, benchmark_case):
        return [1, 2]

    monkeypatch.setattr(runner, "_retrieval_smoke", fake_retrieval)

    loaded, ranks = runner.import_and_validate(
        TARGET_URL,
        BENCHMARK_URL,
        discovered_meta(),
        batch_size=2,
    )

    assert (loaded, ranks) == (17, [1, 2])
    assert events.index(("pristine", TARGET_URL)) < events.index(("loader", TARGET_URL))
    assert next(snapshots, None) is None


def test_import_checks_benchmark_after_target_validation_error(monkeypatch) -> None:
    before = runner.BenchmarkSnapshot(50, 1662, 469874, 0, 0)
    after = runner.BenchmarkSnapshot(50, 1663, 469900, 0, 0)
    snapshots = iter([before, after])

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: next(snapshots))
    monkeypatch.setattr(runner, "assert_pristine_target", lambda url: None)

    class FailingLoader:
        def __init__(self, dsn):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def init_schema(self, path):
            raise runner.ValidationError("target validation failed")

    monkeypatch.setattr(runner, "DBLoader", FailingLoader)

    with pytest.raises(runner.ValidationError, match="benchmark database changed"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            discovered_meta(),
            batch_size=2,
        )


def test_import_rejects_contaminated_benchmark_before_target_write(monkeypatch) -> None:
    contaminated = runner.BenchmarkSnapshot(50, 1663, 469900, 1, 1)
    target_checked = False

    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: contaminated)

    def forbidden_target_check(url):
        nonlocal target_checked
        target_checked = True

    monkeypatch.setattr(runner, "assert_pristine_target", forbidden_target_check)

    with pytest.raises(runner.ValidationError, match="already contaminated"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            discovered_meta(),
            batch_size=2,
        )

    assert target_checked is False
