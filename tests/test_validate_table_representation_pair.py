from __future__ import annotations

from pathlib import Path

import pytest
from evaluation import validate_table_representation_pair as runner
from evaluation.financebench import gold_spans, parse_gold_doc

from corpcheck.models import ChunkResult

TARGET_URL = "postgresql://postgres:postgres@localhost:5432/table_pair_validation"
BENCHMARK_URL = "postgresql://postgres:postgres@localhost:5432/financial_rag"
REAL_USER_AGENT = "CorpCheck research@corpcheck.org"


def meta(spec: runner.FilingSpec, **changes) -> tuple:
    values = [
        spec.ticker,
        runner.FILING_TYPE,
        spec.fiscal_year,
        runner.PERIOD,
        Path(f"/tmp/{spec.ticker}.html"),
        "https://www.sec.gov/example",
        spec.filed_date,
        spec.period_of_report,
        spec.accession,
        spec.cik,
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


def filing_row(spec: runner.FilingSpec) -> tuple:
    return (
        spec.ticker,
        runner.FILING_TYPE,
        spec.fiscal_year,
        runner.PERIOD,
        spec.filed_date,
        spec.period_of_report,
        spec.accession,
        spec.cik,
    )


def evidence_chunk(case: dict, text: str) -> ChunkResult:
    gold = parse_gold_doc(case)
    return ChunkResult(
        chunk_id=str(case["financebench_id"]),
        text=text,
        score=1.0,
        company=str(gold.ticker),
        filing_type=gold.filing_type,
        fiscal_year=gold.fiscal_year,
        period_label="annual",
    )


def test_cli_requires_explicit_target_and_download_directory() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([])
    with pytest.raises(SystemExit):
        runner.parse_args(["--database-url", TARGET_URL])
    with pytest.raises(SystemExit):
        runner.parse_args(["--download-dir", "/tmp/table-pair"])


def test_exact_pair_metadata_accepts_only_locked_pair() -> None:
    pair = runner.exact_pair_metadata(
        [meta(runner.FILING_SPECS[1]), meta(runner.FILING_SPECS[0], cik="1018724")]
    )

    assert [(row[0], row[8]) for row in pair] == [
        ("AMZN", "0001018724-20-000004"),
        ("NKE", "0000320187-18-000142"),
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [meta(runner.FILING_SPECS[0])],
        [meta(runner.FILING_SPECS[0]), meta(runner.FILING_SPECS[0])],
        [
            meta(runner.FILING_SPECS[0]),
            meta(runner.FILING_SPECS[1]),
            meta(runner.FILING_SPECS[0], accession="extra"),
        ],
    ],
)
def test_exact_pair_metadata_rejects_missing_duplicate_or_extra(rows) -> None:
    with pytest.raises(runner.ValidationError, match="pair exactly"):
        runner.exact_pair_metadata(rows)


def test_exact_pair_metadata_rejects_wrong_period() -> None:
    rows = [
        meta(runner.FILING_SPECS[0], period_of_report=None),
        meta(runner.FILING_SPECS[1]),
    ]
    with pytest.raises(runner.ValidationError, match="locked filing"):
        runner.exact_pair_metadata(rows)


class FakeLibraryDownloader:
    def __init__(self) -> None:
        self.calls = []

    def get(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class FakeSECDownloader:
    def __init__(self) -> None:
        self._downloader = FakeLibraryDownloader()
        self.rate_limits = 0
        self.collect_calls = []

    def _rate_limit(self) -> None:
        self.rate_limits += 1

    def _collect_metadata(self, tickers, filing_types, years):
        self.collect_calls.append((tickers, filing_types, years))
        spec = next(spec for spec in runner.FILING_SPECS if spec.ticker == tickers[0])
        return [meta(spec)]


def test_prepare_pair_uses_two_narrow_exact_downloads(tmp_path) -> None:
    downloader = FakeSECDownloader()

    pair = runner.prepare_pair(tmp_path, downloader=downloader)

    assert len(pair) == 2
    assert downloader.rate_limits == 2
    assert [call[0] for call in downloader._downloader.calls] == [
        ("10-K", "AMZN"),
        ("10-K", "NKE"),
    ]
    for (_, kwargs), spec in zip(
        downloader._downloader.calls, runner.FILING_SPECS, strict=True
    ):
        assert kwargs == {
            "after": spec.after,
            "before": spec.before,
            "limit": 1,
            "include_amends": False,
            "download_details": True,
        }


def test_main_rejects_invalid_user_agent_before_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "CorpCheck Validation research@corpcheck.org")
    monkeypatch.setattr(
        runner,
        "prepare_pair",
        lambda *args, **kwargs: pytest.fail("download must not begin"),
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


def test_prepare_only_does_not_touch_database(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", REAL_USER_AGENT)
    monkeypatch.setattr(
        runner,
        "prepare_pair",
        lambda path: [meta(spec) for spec in runner.FILING_SPECS],
    )
    monkeypatch.setattr(
        runner,
        "import_and_validate",
        lambda *args, **kwargs: pytest.fail("database import must not begin"),
    )
    monkeypatch.setattr(
        runner,
        "benchmark_snapshot",
        lambda *args, **kwargs: pytest.fail("benchmark DB must not be read"),
    )

    assert (
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
        == 0
    )


def test_semantic_table_requires_values_title_header_and_embedding() -> None:
    spec = runner.FILING_SPECS[0]
    valid = [
        (
            "Net income 11,588",
            True,
            "Amazon.com, Inc. Consolidated Statements of Operations",
            "Year Ended December 31 | 2017 | 2018 | 2019",
        )
    ]
    runner._assert_semantic_table(spec, valid)

    invalid_rows = [
        ("Net income 11,588", True, "Table 119", "2017 | 2018 | 2019"),
        (
            "Net income 11,588",
            True,
            "Consolidated Statements of Operations",
            None,
        ),
        (
            "Net income 11,588",
            False,
            "Consolidated Statements of Operations",
            "Year Ended December 31 | 2019",
        ),
    ]
    for row in invalid_rows:
        with pytest.raises(runner.ValidationError, match="semantic title"):
            runner._assert_semantic_table(spec, [row])


def test_imported_pair_asserts_exact_filings_and_table_metadata(monkeypatch) -> None:
    amzn, nke = runner.FILING_SPECS
    result_sets = iter(
        [
            [filing_row(amzn), filing_row(nke)],
            [
                (
                    "Net income 11,588",
                    True,
                    "Amazon.com, Inc. Consolidated Statements of Operations",
                    "Year Ended December 31 | 2017 | 2018 | 2019",
                )
            ],
            [
                (
                    "Revenues 36,397 Cost of sales 20,441",
                    True,
                    "NIKE, Inc. Consolidated Statements of Income",
                    "Year Ended May 31 | 2018 | 2017 | 2016",
                )
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
                assert params in {(amzn.accession,), (nke.accession,)}
                assert "structure_meta->>'header_text'" in sql
                assert "content_kind = 'table'" in sql

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

    runner.assert_imported_pair(TARGET_URL)


def test_strict_overlap_is_provenance_gated_for_both_questions() -> None:
    for case in runner.load_benchmark_cases():
        span = gold_spans(case)[0]
        assert runner.assert_strict_overlap([evidence_chunk(case, span)], case) == [1]

        wrong = evidence_chunk(case, span)
        wrong.fiscal_year = int(wrong.fiscal_year or 0) + 1
        with pytest.raises(runner.ValidationError, match="Recall@10 failed"):
            runner.assert_strict_overlap([wrong], case)


@pytest.mark.asyncio
async def test_retrieval_smoke_passes_question_only_without_gold_filters(monkeypatch) -> None:
    cases = runner.load_benchmark_cases()
    by_question = {str(case["question"]): case for case in cases}
    calls = []

    class FakePool:
        async def close(self):
            pass

    async def fake_create_pool(**kwargs):
        return FakePool()

    async def fake_load_known_tickers(pool):
        pass

    async def fake_retrieve(**kwargs):
        calls.append(kwargs)
        case = by_question[str(kwargs["query"])]
        return [evidence_chunk(case, gold_spans(case)[0])]

    monkeypatch.setattr(runner.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(runner, "load_known_tickers", fake_load_known_tickers)
    monkeypatch.setattr(runner, "retrieve", fake_retrieve)

    results = await runner._retrieval_smoke(TARGET_URL, cases)

    assert set(results) == {spec.financebench_id for spec in runner.FILING_SPECS}
    assert len(calls) == 2
    for call in calls:
        assert call["company"] is None
        assert call["filing_type"] is None
        assert call["fiscal_year"] is None
        assert call["sector"] is None
        assert call["k"] == 10


def test_import_path_is_pristine_and_checks_benchmark_before_after(monkeypatch) -> None:
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
    monkeypatch.setattr(runner, "_process_filings", lambda *args, **kwargs: 23)
    monkeypatch.setattr(runner, "assert_imported_pair", lambda url: None)
    monkeypatch.setattr(runner, "load_benchmark_cases", lambda: [])

    async def fake_retrieval(database_url, cases):
        return {"amzn": [1], "nke": [2]}

    monkeypatch.setattr(runner, "_retrieval_smoke", fake_retrieval)

    loaded, ranks = runner.import_and_validate(
        TARGET_URL,
        BENCHMARK_URL,
        [meta(spec) for spec in runner.FILING_SPECS],
        batch_size=2,
    )

    assert loaded == 23
    assert ranks == {"amzn": [1], "nke": [2]}
    assert events.index(("pristine", TARGET_URL)) < events.index(("loader", TARGET_URL))
    assert next(snapshots, None) is None


def test_import_path_rejects_non_locked_input_before_database_access(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "benchmark_snapshot",
        lambda url: pytest.fail("benchmark DB must not be read for invalid input"),
    )

    with pytest.raises(runner.ValidationError, match="pair exactly"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            [meta(runner.FILING_SPECS[0])],
            batch_size=2,
        )


def test_import_path_rejects_contaminated_benchmark_before_target_access(
    monkeypatch,
) -> None:
    contaminated = runner.BenchmarkSnapshot(51, 1663, 469875, 1, 0)
    monkeypatch.setattr(runner, "benchmark_snapshot", lambda url: contaminated)
    monkeypatch.setattr(
        runner,
        "assert_pristine_target",
        lambda url: pytest.fail("target must not be inspected after dirty benchmark"),
    )

    with pytest.raises(runner.ValidationError, match="contaminated"):
        runner.import_and_validate(
            TARGET_URL,
            BENCHMARK_URL,
            [meta(spec) for spec in runner.FILING_SPECS],
            batch_size=2,
        )
