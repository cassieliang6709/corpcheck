from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from evaluation import recover_sec_submissions as recovery
from evaluation import reprocess_sec_corpus as reprocessor
from evaluation.snapshot_sec_corpus_manifest import build_envelope, render_manifest


def filing(
    *,
    ticker: str,
    accession: str,
    cik: str,
    fiscal_year: int,
) -> dict:
    return {
        "ticker": ticker,
        "form": "10-K",
        "fiscal_year": fiscal_year,
        "period": "annual",
        "filed_date": dt.date(fiscal_year + 1, 2, 1),
        "period_of_report": dt.date(fiscal_year, 12, 31),
        "accession": accession,
        "cik": cik,
        "source_url": "https://www.sec.gov/Archives/example-index.htm",
    }


def valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    rows = [
        filing(
            ticker="AMZN",
            accession="0001018724-20-000004",
            cik="0001018724",
            fiscal_year=2019,
        ),
        filing(
            ticker="AAPL",
            accession="0000320193-23-000106",
            cik="0000320193",
            fiscal_year=2022,
        ),
    ]
    envelope = build_envelope(rows, chunk_count=2, embedded_chunk_count=2)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(render_manifest(envelope), encoding="utf-8")
    _, filings = recovery.load_manifest(manifest)

    root = tmp_path / "recovery"
    records = []
    for index, spec in enumerate(filings):
        path = recovery.destination_for(root, spec)
        path.parent.mkdir(parents=True)
        body = f"submission-{index}-{spec.accession}".encode()
        path.write_bytes(body)
        records.append(
            recovery.FileRecord(
                relative_path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(body).hexdigest(),
                size=len(body),
            )
        )

    report_data = recovery.build_report(envelope["digest"]["payload_sha256"], records)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_data), encoding="utf-8")
    return manifest, report, root, report_data


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def test_valid_inputs_return_typed_verified_metadata(tmp_path) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)

    result = reprocessor.load_reprocessing_inputs(manifest, report, root)

    assert result.manifest_payload_sha256 == report_data["manifest_payload_sha256"]
    assert result.recovery_root == root.resolve()
    assert [item.filing.accession for item in result.submissions] == [
        "0000320193-23-000106",
        "0001018724-20-000004",
    ]
    assert result.submissions[0].filing.fiscal_year == 2022
    assert result.submissions[0].filing.period == "annual"
    assert result.submissions[0].filing.source_url.startswith("https://www.sec.gov/")
    assert [item.relative_path for item in result.submissions] == [
        row["relative_path"] for row in report_data["files"]
    ]
    assert all(item.path.is_absolute() for item in result.submissions)


def test_report_manifest_digest_must_match(tmp_path) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    report_data["manifest_payload_sha256"] = "a" * 64
    write_report(report, report_data)

    with pytest.raises(reprocessor.ReprocessInputError, match="digest does not match"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_report_must_have_one_ordered_file_per_manifest_accession(
    tmp_path, mutation
) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    if mutation == "missing":
        report_data["files"].pop()
    elif mutation == "extra":
        report_data["files"].append(
            {
                "relative_path": "sec-edgar-filings/EXTRA/10-K/x/full-submission.txt",
                "sha256": "b" * 64,
                "size": 1,
            }
        )
    elif mutation == "duplicate":
        report_data["files"][1] = dict(report_data["files"][0])
    else:
        report_data["files"].reverse()
    report_data["file_count"] = len(report_data["files"])
    write_report(report, report_data)

    message = "duplicate" if mutation == "duplicate" else "manifest|accession order"
    with pytest.raises(reprocessor.ReprocessInputError, match=message):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


@pytest.mark.parametrize(
    "relative_path",
    [
        "/tmp/full-submission.txt",
        "../full-submission.txt",
        "sec-edgar-filings\\AAPL\\full-submission.txt",
        "sec-edgar-filings//AAPL/full-submission.txt",
    ],
)
def test_report_rejects_non_deterministic_relative_paths(tmp_path, relative_path) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    report_data["files"][0]["relative_path"] = relative_path
    write_report(report, report_data)

    with pytest.raises(reprocessor.ReprocessInputError, match="relative_path"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


@pytest.mark.parametrize("field", ["sha256", "size"])
def test_recovered_bytes_must_match_report(tmp_path, field) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    if field == "sha256":
        report_data["files"][0][field] = "c" * 64
    else:
        report_data["files"][0][field] += 1
    write_report(report, report_data)

    with pytest.raises(reprocessor.ReprocessInputError, match=f"{field} mismatch"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


def test_recovered_file_symlink_is_rejected_even_when_target_is_inside_root(tmp_path) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    relative_path = report_data["files"][0]["relative_path"]
    path = root / relative_path
    target = root / "target.txt"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(reprocessor.ReprocessInputError, match="symlinks"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


def test_symlinked_parent_and_non_regular_submission_are_rejected(tmp_path) -> None:
    manifest, report, root, report_data = valid_inputs(tmp_path)
    relative_path = Path(report_data["files"][0]["relative_path"])
    ticker_dir = root / relative_path.parts[0] / relative_path.parts[1]
    real_ticker_dir = root / "real-ticker"
    ticker_dir.rename(real_ticker_dir)
    ticker_dir.symlink_to(real_ticker_dir, target_is_directory=True)

    with pytest.raises(reprocessor.ReprocessInputError, match="symlinks"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)

    ticker_dir.unlink()
    real_ticker_dir.rename(ticker_dir)
    submission = root / relative_path
    submission.unlink()
    submission.mkdir()
    with pytest.raises(reprocessor.ReprocessInputError, match="not a regular file"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


@pytest.mark.parametrize("input_name", ["manifest", "report", "root"])
def test_top_level_input_symlinks_are_rejected(tmp_path, input_name) -> None:
    manifest, report, root, _ = valid_inputs(tmp_path)
    original = {"manifest": manifest, "report": report, "root": root}[input_name]
    target = tmp_path / f"real-{input_name}"
    original.rename(target)
    original.symlink_to(target, target_is_directory=input_name == "root")

    with pytest.raises(reprocessor.ReprocessInputError, match="non-symlink"):
        reprocessor.load_reprocessing_inputs(manifest, report, root)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self.rows = []

    def execute(self, sql):
        if "current_database" in sql:
            columns = ["database_name", "server_address", "server_port"]
            rows = [(self.connection.database_name, "127.0.0.1", 5432)]
        elif "FROM companies" in sql:
            columns = [
                "ticker", "name", "sector", "industry", "market_cap", "description"
            ]
            rows = self.connection.companies
        elif "FROM filings" in sql:
            columns = [
                "id", "ticker", "filing_type", "fiscal_year", "period", "filed_date",
                "period_of_report", "accession_number", "cik", "source_url", "local_path",
            ]
            rows = self.connection.filings
        else:
            raise AssertionError(f"unexpected SQL: {sql}")
        self.description = [(column,) for column in columns]
        self.rows = rows

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, database_name, *, companies=None, filings=None):
        self.database_name = database_name
        self.companies = companies or [
            ("AMZN", "Amazon.com, Inc.", "tech", None, None, None)
        ]
        self.filings = filings or [
            (
                7,
                "AMZN",
                "10-K",
                2019,
                "annual",
                dt.date(2020, 2, 1),
                dt.date(2019, 12, 31),
                "0001018724-20-000004",
                "0001018724",
                "https://www.sec.gov/Archives/example-index.htm",
                "/old/raw.txt",
            )
        ]

    def cursor(self):
        return FakeCursor(self)


def test_database_pair_requires_same_server_and_distinct_names() -> None:
    old, new = reprocessor.validate_database_pair(
        "postgresql://user:secret@db.internal:5432/corpcheck_old",
        "postgresql://user:other@db.internal/corpcheck_new",
    )
    assert (old.host, old.port, old.database_name) == (
        "db.internal", 5432, "corpcheck_old"
    )
    assert new.database_name == "corpcheck_new"

    with pytest.raises(reprocessor.ReprocessInputError, match="same server"):
        reprocessor.validate_database_pair(
            "postgresql://db-a/old", "postgresql://db-b/new"
        )
    with pytest.raises(reprocessor.ReprocessInputError, match="must differ"):
        reprocessor.validate_database_pair(
            "postgresql://db/old", "postgresql://db/old"
        )


def test_preflight_requires_exact_company_and_filing_identity() -> None:
    old_endpoint, new_endpoint = reprocessor.validate_database_pair(
        "postgresql://db/corpcheck_old", "postgresql://db/corpcheck_new"
    )
    result = reprocessor.preflight_databases(
        FakeConnection("corpcheck_old"),
        FakeConnection("corpcheck_new"),
        old_endpoint=old_endpoint,
        new_endpoint=new_endpoint,
    )
    assert result.old.identity_sha256 == result.new.identity_sha256
    assert result.old.company_count == 1
    assert result.old.filing_count == 1

    changed_companies = [
        ("AMZN", "Amazon.com, Inc.", "consumer", None, None, None)
    ]
    with pytest.raises(reprocessor.ReprocessInputError, match="company identities"):
        reprocessor.preflight_databases(
            FakeConnection("corpcheck_old"),
            FakeConnection("corpcheck_new", companies=changed_companies),
            old_endpoint=old_endpoint,
            new_endpoint=new_endpoint,
        )

    changed_filing = list(FakeConnection("x").filings[0])
    changed_filing[8] = "1018724"
    with pytest.raises(reprocessor.ReprocessInputError, match="filing identities"):
        reprocessor.preflight_databases(
            FakeConnection("corpcheck_old"),
            FakeConnection("corpcheck_new", filings=[tuple(changed_filing)]),
            old_endpoint=old_endpoint,
            new_endpoint=new_endpoint,
        )


def _submission_and_target(tmp_path):
    manifest, report, root, _ = valid_inputs(tmp_path)
    submission = reprocessor.load_reprocessing_inputs(manifest, report, root).submissions[0]
    filing = submission.filing
    target = reprocessor.FilingTarget(
        filing_id=41,
        ticker=filing.ticker,
        sector="tech",
        form=filing.form,
        fiscal_year=filing.fiscal_year,
        period=filing.period,
        filed_date=filing.filed_date,
        period_of_report=filing.period_of_report,
        accession=filing.accession,
        cik=filing.cik,
        source_url=filing.source_url,
    )
    return submission, target


class FakeCleaner:
    def __init__(self, segments):
        self.segments = segments
        self.paths = []

    def clean_segments(self, path):
        self.paths.append(path)
        return self.segments


class FakeChunker:
    def __init__(self, payloads):
        self.payloads = payloads

    def chunk_segments(self, segments):
        assert segments
        return self.payloads


class FakeEmbedder:
    def __init__(self, result):
        self.result = result
        self.texts = None

    def encode(self, texts):
        self.texts = texts
        return self.result


def test_process_one_submission_builds_contiguous_fully_embedded_rows(tmp_path) -> None:
    submission, target = _submission_and_target(tmp_path)
    segment = SimpleNamespace(text="segment")
    payloads = [
        SimpleNamespace(
            section_name="Financial Statements",
            text="Revenue was $100 million in the year ended 2022.",
            token_count=11,
            content_kind="table",
            chunk_strategy="table_rows",
            display_title="Revenue",
            chunk_group_key="table:1",
            structure_meta={"row_start": 0},
        ),
        SimpleNamespace(
            section_name="MD&A",
            text="Operating income increased to $20 million during 2022.",
            token_count=10,
            content_kind="narrative",
            chunk_strategy="sentence_pack",
            display_title=None,
            chunk_group_key=None,
            structure_meta={},
        ),
    ]
    cleaner = FakeCleaner([segment])
    embedder = FakeEmbedder(np.ones((2, 384), dtype=np.float32))

    rows = reprocessor.process_one_submission(
        submission,
        target,
        cleaner_factory=lambda form: cleaner,
        chunker=FakeChunker(payloads),
        embedder=embedder,
    )

    assert cleaner.paths == [submission.path]
    assert embedder.texts == [payload.text for payload in payloads]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert all(row["filing_id"] == target.filing_id for row in rows)
    assert all(row["ticker"] == target.ticker for row in rows)
    assert all(row["sector"] == target.sector for row in rows)
    assert all(row["embedding"].shape == (384,) for row in rows)
    assert rows[0]["content_kind"] == "table"
    assert rows[0]["numeric_token_count"] > 0


@pytest.mark.parametrize(
    "failure, message",
    [
        ("metadata", "metadata does not match"),
        ("segments", "no segments"),
        ("empty-segment", "empty segment"),
        ("chunks", "no chunks"),
        ("embedding-shape", "embeddings have shape"),
        ("embedding-nan", "non-finite"),
    ],
)
def test_process_one_submission_fails_closed(tmp_path, failure, message) -> None:
    submission, target = _submission_and_target(tmp_path)
    if failure == "metadata":
        target = replace(target, fiscal_year=1999)
    segments = (
        []
        if failure == "segments"
        else [SimpleNamespace(text="" if failure == "empty-segment" else "segment")]
    )
    payloads = [] if failure == "chunks" else [
        SimpleNamespace(
            section_name="MD&A",
            text="Revenue increased to $100 million in 2022.",
            token_count=9,
            content_kind="narrative",
            chunk_strategy="sentence_pack",
            display_title=None,
            chunk_group_key=None,
            structure_meta={},
        )
    ]
    embeddings = np.ones((len(payloads), 384), dtype=np.float32)
    if failure == "embedding-shape":
        embeddings = np.ones((1, 383), dtype=np.float32)
    elif failure == "embedding-nan":
        embeddings[0, 0] = np.nan

    with pytest.raises(reprocessor.ReprocessInputError, match=message):
        reprocessor.process_one_submission(
            submission,
            target,
            cleaner_factory=lambda form: FakeCleaner(segments),
            chunker=FakeChunker(payloads),
            embedder=FakeEmbedder(embeddings),
        )


def test_process_one_submission_rehashes_raw_file_before_work(tmp_path) -> None:
    submission, target = _submission_and_target(tmp_path)
    submission.path.write_bytes(b"changed after initial verification")

    with pytest.raises(reprocessor.ReprocessInputError, match="changed"):
        reprocessor.process_one_submission(
            submission,
            target,
            cleaner_factory=lambda form: pytest.fail("cleaner must not run"),
        )


def test_processing_source_fingerprint_is_stable_sha256() -> None:
    first = reprocessor.processing_source_fingerprint()
    assert first == reprocessor.processing_source_fingerprint()
    assert len(first) == 64
    int(first, 16)
