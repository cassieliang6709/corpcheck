from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import urllib.error
from pathlib import Path

import pytest
from evaluation import recover_sec_submissions as recovery
from evaluation.snapshot_sec_corpus_manifest import build_envelope, render_manifest

REAL_USER_AGENT = "CorpCheck research@corpcheck.org"
ACCESSION = "0000320193-23-000106"
CIK = "0000320193"


def filing_row() -> dict:
    return {
        "ticker": "AAPL",
        "form": "10-K",
        "fiscal_year": 2023,
        "period": "annual",
        "filed_date": dt.date(2023, 11, 3),
        "period_of_report": dt.date(2023, 9, 30),
        "accession": ACCESSION,
        "cik": CIK,
        "source_url": "https://www.sec.gov/Archives/example-index.htm",
    }


def manifest_file(tmp_path: Path) -> tuple[Path, dict]:
    envelope = build_envelope([filing_row()], chunk_count=4, embedded_chunk_count=4)
    path = tmp_path / "manifest.json"
    path.write_text(render_manifest(envelope), encoding="utf-8")
    return path, envelope


def submission(**changes: str) -> bytes:
    fields = {
        "accession": ACCESSION,
        "form": "10-K",
        "cik": CIK,
        "filed_date": "20231103",
        "period_of_report": "20230930",
        **changes,
    }
    return (
        "<SEC-DOCUMENT>example.txt\n"
        "<SEC-HEADER>\n"
        f"ACCESSION NUMBER: {fields['accession']}\n"
        f"CONFORMED SUBMISSION TYPE: {fields['form']}\n"
        "FILER:\n"
        f"CENTRAL INDEX KEY: {fields['cik']}\n"
        f"FILED AS OF DATE: {fields['filed_date']}\n"
        f"CONFORMED PERIOD OF REPORT: {fields['period_of_report']}\n"
        "</SEC-HEADER>\n"
        "<DOCUMENT>body</DOCUMENT>\n"
    ).encode()


def spec() -> recovery.FilingSpec:
    return recovery.FilingSpec(
        ticker="AAPL",
        form="10-K",
        fiscal_year=2023,
        period="annual",
        filed_date="2023-11-03",
        period_of_report="2023-09-30",
        accession=ACCESSION,
        cik=CIK,
        source_url="https://www.sec.gov/Archives/example-index.htm",
        full_submission_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019323000106/0000320193-23-000106.txt"
        ),
    )


class FakeResponse:
    def __init__(self, body: bytes, *, on_read=None):
        self.body = body
        self.on_read = on_read
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return 200

    def read(self, _size):
        if self.on_read is not None:
            self.on_read()
        if self.sent:
            return b""
        self.sent = True
        return self.body


def test_manifest_tamper_is_rejected(tmp_path) -> None:
    path, envelope = manifest_file(tmp_path)
    envelope["filings"][0]["ticker"] = "MSFT"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recovery.RecoveryError, match="digest mismatch"):
        recovery.load_manifest(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"accession": "0000320193-23-999999"}, "accession mismatch"),
        ({"form": "10-K/A"}, "form mismatch"),
        ({"cik": "123456"}, "cik mismatch"),
        ({"filed_date": "20231104"}, "filed_date mismatch"),
        ({"period_of_report": "20230929"}, "period_of_report mismatch"),
    ],
)
def test_sec_header_is_exact(change, message) -> None:
    recovery.validate_submission_header(submission(), spec())

    with pytest.raises(recovery.RecoveryError, match=message):
        recovery.validate_submission_header(submission(**change), spec())


def test_existing_valid_file_is_resumed_without_http(tmp_path) -> None:
    manifest, envelope = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    destination = recovery.destination_for(download_dir.resolve(), spec())
    destination.parent.mkdir(parents=True)
    body = submission()
    destination.write_bytes(body)

    def unexpected_http(*_args, **_kwargs):
        raise AssertionError("valid resumed file must not make an HTTP request")

    report, resumed = recovery.recover_manifest(
        manifest,
        download_dir,
        tmp_path / "report.json",
        user_agent=REAL_USER_AGENT,
        opener=unexpected_http,
    )

    assert resumed == 1
    assert report["manifest_payload_sha256"] == envelope["digest"]["payload_sha256"]
    assert report["files"] == [
        {
            "relative_path": (
                f"sec-edgar-filings/AAPL/10-K/{ACCESSION}/full-submission.txt"
            ),
            "sha256": hashlib.sha256(body).hexdigest(),
            "size": len(body),
        }
    ]


def test_invalid_existing_file_is_a_collision(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    destination = recovery.destination_for(download_dir.resolve(), spec())
    destination.parent.mkdir(parents=True)
    destination.write_bytes(submission(form="10-Q"))

    with pytest.raises(recovery.RecoveryError, match="invalid existing-file collision"):
        recovery.recover_manifest(
            manifest,
            download_dir,
            tmp_path / "report.json",
            user_agent=REAL_USER_AGENT,
            opener=lambda *_args, **_kwargs: pytest.fail("must not download"),
        )


def test_successful_download_is_atomic_and_writes_report(tmp_path) -> None:
    manifest, envelope = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    destination = recovery.destination_for(download_dir.resolve(), spec())
    part = destination.with_name("full-submission.txt.part")
    body = submission()
    requests = []

    def during_read():
        assert part.exists()
        assert not destination.exists()

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(body, on_read=during_read)

    report_path = tmp_path / "report.json"
    report, resumed = recovery.recover_manifest(
        manifest,
        download_dir,
        report_path,
        user_agent=REAL_USER_AGENT,
        opener=opener,
    )

    assert resumed == 0
    assert destination.read_bytes() == body
    assert not part.exists()
    assert requests[0][0].full_url == envelope["filings"][0]["full_submission_url"]
    assert requests[0][0].get_header("User-agent") == REAL_USER_AGENT
    assert requests[0][1] == 30
    assert json.loads(report_path.read_text()) == report


def test_transient_error_retries_with_rate_limit_then_succeeds(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    attempts = []
    sleeps = []

    def opener(request, *, timeout):
        attempts.append((request, timeout))
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "busy", None, None)
        return FakeResponse(submission())

    report, _ = recovery.recover_manifest(
        manifest,
        tmp_path / "isolated",
        tmp_path / "report.json",
        user_agent=REAL_USER_AGENT,
        opener=opener,
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )

    assert report["file_count"] == 1
    assert len(attempts) == 2
    assert sleeps == [pytest.approx(0.1)]


def test_curl_transport_is_atomic_https_only_and_validated(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    destination = recovery.destination_for(download_dir.resolve(), spec())
    commands = []

    def runner(command, **options):
        commands.append((command, options))
        part_path = Path(command[command.index("--output") + 1])
        assert part_path.name.endswith(".part")
        assert not destination.exists()
        part_path.write_bytes(submission())
        return subprocess.CompletedProcess(command, 0, "", "")

    report, resumed = recovery.recover_manifest(
        manifest,
        download_dir,
        tmp_path / "report.json",
        user_agent=REAL_USER_AGENT,
        transport="curl",
        curl_runner=runner,
        opener=lambda *_args, **_kwargs: pytest.fail("urllib must not run"),
    )

    assert resumed == 0
    assert report["file_count"] == 1
    assert destination.read_bytes() == submission()
    command, options = commands[0]
    assert command[0] == "curl"
    assert command[1] == "--disable"
    assert "--insecure" not in command
    assert command[command.index("--proto") + 1] == "=https"
    assert command[command.index("--proto-redir") + 1] == "=https"
    assert command[command.index("--user-agent") + 1] == REAL_USER_AGENT
    assert options == {"capture_output": True, "text": True, "check": False}


def test_curl_transport_retries_and_removes_partial_file(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    attempts = []
    sleeps = []

    def runner(command, **_options):
        part_path = Path(command[command.index("--output") + 1])
        attempts.append(part_path)
        if len(attempts) == 1:
            part_path.write_bytes(b"partial")
            return subprocess.CompletedProcess(command, 35, "", "tls failed")
        assert not part_path.exists()
        part_path.write_bytes(submission())
        return subprocess.CompletedProcess(command, 0, "", "")

    report, _ = recovery.recover_manifest(
        manifest,
        download_dir,
        tmp_path / "report.json",
        user_agent=REAL_USER_AGENT,
        transport="curl",
        curl_runner=runner,
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )

    assert report["file_count"] == 1
    assert len(attempts) == 2
    assert sleeps == [pytest.approx(0.1)]


def test_unknown_transport_fails_before_writing(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"

    with pytest.raises(recovery.RecoveryError, match="--transport"):
        recovery.recover_manifest(
            manifest,
            download_dir,
            tmp_path / "report.json",
            user_agent=REAL_USER_AGENT,
            transport="other",
        )

    assert not download_dir.exists()


def test_permanent_error_does_not_retry_or_leave_partial_output(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    download_dir = tmp_path / "isolated"
    calls = 0

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, 404, "missing", None, None)

    with pytest.raises(recovery.RecoveryError, match="after 1 attempt"):
        recovery.recover_manifest(
            manifest,
            download_dir,
            tmp_path / "report.json",
            user_agent=REAL_USER_AGENT,
            opener=opener,
        )

    assert calls == 1
    destination = recovery.destination_for(download_dir.resolve(), spec())
    assert not destination.exists()
    assert not destination.with_name("full-submission.txt.part").exists()
    assert not (tmp_path / "report.json").exists()


def test_strict_user_agent_and_default_directory_are_rejected(tmp_path) -> None:
    manifest, _ = manifest_file(tmp_path)
    with pytest.raises(recovery.RecoveryError, match="non-placeholder"):
        recovery.recover_manifest(
            manifest,
            tmp_path / "isolated",
            tmp_path / "report.json",
            user_agent="CorpCheck your-email@example.com",
        )

    with pytest.raises(recovery.RecoveryError, match="default production"):
        recovery.validate_download_dir(Path("data/sec_filings"))


def test_report_is_write_once_and_collision_safe(tmp_path) -> None:
    path = tmp_path / "report.json"
    report = recovery.build_report(
        "a" * 64,
        [recovery.FileRecord("sec-edgar-filings/A/10-K/a/full-submission.txt", "b" * 64, 1)],
    )
    recovery.write_report_once(path, report)
    recovery.write_report_once(path, report)

    changed = {**report, "file_count": 2}
    with pytest.raises(recovery.RecoveryError, match="refusing to overwrite"):
        recovery.write_report_once(path, changed)


def test_main_returns_nonzero_without_a_real_user_agent(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "CorpCheck your-email@example.com")
    exit_code = recovery.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--download-dir",
            str(tmp_path / "isolated"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert exit_code == 1
    assert "non-placeholder" in capsys.readouterr().err
    assert not (tmp_path / "isolated").exists()
