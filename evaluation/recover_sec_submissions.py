#!/usr/bin/env python3
"""Recover exact raw SEC submissions from a signed corpus snapshot.

This utility only writes an isolated SEC download tree and a deterministic
recovery report. It never connects to or mutates a database.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from corpcheck.ingestion.config import SEC_DOWNLOAD_DIR, SEC_USER_AGENT
from evaluation.snapshot_sec_corpus_manifest import (
    ACCESSION_RE,
    CIK_RE,
    SCHEMA_VERSION,
    full_submission_url,
)
from evaluation.validate_amendment_pair import (
    ValidationError,
    validate_real_sec_user_agent,
)

MANIFEST_KEYS = {"schema_version", "corpus_counts", "filings", "digest"}
COUNT_KEYS = {"ticker_count", "filing_count", "chunk_count", "embedded_chunk_count"}
FILING_KEYS = {
    "ticker",
    "form",
    "fiscal_year",
    "period",
    "filed_date",
    "period_of_report",
    "accession",
    "cik",
    "source_url",
    "full_submission_url",
}
DIGEST_KEYS = {"algorithm", "payload_sha256"}
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_HEADER_BYTES = 1_048_576
CHUNK_BYTES = 64 * 1024
MAX_ATTEMPTS_LIMIT = 5
REPORT_SCHEMA_VERSION = 1
SAFE_TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_FORM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*(?:/[A-Za-z0-9][A-Za-z0-9-]*)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RecoveryError(RuntimeError):
    """The recovery contract was violated; no unsafe fallback is allowed."""


class _HTTPStatusError(RuntimeError):
    def __init__(self, status: int):
        super().__init__(f"SEC returned HTTP {status}")
        self.status = status


@dataclass(frozen=True)
class FilingSpec:
    ticker: str
    form: str
    filed_date: str
    period_of_report: str
    accession: str
    cik: str
    full_submission_url: str


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    sha256: str
    size: int


class RateLimiter:
    """A process-local limiter that spaces every request attempt."""

    def __init__(
        self,
        max_rps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < max_rps <= 10:
            raise RecoveryError("--max-rps must be greater than 0 and no more than 10")
        self._interval = 1.0 / max_rps
        self._clock = clock
        self._sleep = sleep
        self._last_request: Optional[float] = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_request is not None:
            delay = self._last_request + self._interval - now
            if delay > 0:
                self._sleep(delay)
        self._last_request = self._clock()


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RecoveryError(f"{label} must contain exactly: {', '.join(sorted(expected))}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecoveryError(f"{label} must be a non-negative integer")
    return value


def _required_text(row: dict[str, Any], field: str, accession: str = "filing") -> str:
    value = row[field]
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecoveryError(f"{accession} has invalid {field}")
    return value


def _iso_date(row: dict[str, Any], field: str, accession: str) -> str:
    value = _required_text(row, field, accession)
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise RecoveryError(f"{accession} has invalid {field}") from exc
    if parsed.isoformat() != value:
        raise RecoveryError(f"{accession} has invalid {field}")
    return value


def _validate_sec_url(url: str, accession: str, field: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"sec.gov", "www.sec.gov"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise RecoveryError(f"{accession} has invalid {field}")


def _validate_filing(raw: Any) -> FilingSpec:
    row = _exact_keys(raw, FILING_KEYS, "filing row")
    accession = _required_text(row, "accession")
    if ACCESSION_RE.fullmatch(accession) is None:
        raise RecoveryError(f"filing has invalid accession: {accession!r}")

    ticker = _required_text(row, "ticker", accession)
    form = _required_text(row, "form", accession)
    cik = _required_text(row, "cik", accession)
    if SAFE_TICKER_RE.fullmatch(ticker) is None or ticker in {".", ".."}:
        raise RecoveryError(f"{accession} has unsafe ticker path component")
    if SAFE_FORM_RE.fullmatch(form) is None:
        raise RecoveryError(f"{accession} has unsafe form path component")
    if CIK_RE.fullmatch(cik) is None or int(cik) <= 0:
        raise RecoveryError(f"{accession} has invalid CIK")

    fiscal_year = row["fiscal_year"]
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
        raise RecoveryError(f"{accession} has invalid fiscal_year")
    _required_text(row, "period", accession)
    filed_date = _iso_date(row, "filed_date", accession)
    period_of_report = _iso_date(row, "period_of_report", accession)

    source_url = _required_text(row, "source_url", accession)
    _validate_sec_url(source_url, accession, "source_url")
    submission_url = _required_text(row, "full_submission_url", accession)
    _validate_sec_url(submission_url, accession, "full_submission_url")
    if submission_url != full_submission_url(cik, accession):
        raise RecoveryError(f"{accession} full_submission_url does not match its CIK/accession")

    return FilingSpec(
        ticker=ticker,
        form=form,
        filed_date=filed_date,
        period_of_report=period_of_report,
        accession=accession,
        cik=cik,
        full_submission_url=submission_url,
    )


def load_manifest(path: Path) -> tuple[str, list[FilingSpec]]:
    """Validate the complete snapshot envelope and return its exact filing rows."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read manifest {path}: {exc}") from exc

    envelope = _exact_keys(raw, MANIFEST_KEYS, "manifest")
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise RecoveryError(f"unsupported manifest schema_version: {envelope['schema_version']!r}")
    counts = _exact_keys(envelope["corpus_counts"], COUNT_KEYS, "corpus_counts")
    ticker_count = _nonnegative_int(counts["ticker_count"], "ticker_count")
    filing_count = _nonnegative_int(counts["filing_count"], "filing_count")
    chunk_count = _nonnegative_int(counts["chunk_count"], "chunk_count")
    embedded_count = _nonnegative_int(
        counts["embedded_chunk_count"], "embedded_chunk_count"
    )
    if embedded_count > chunk_count:
        raise RecoveryError("embedded_chunk_count cannot exceed chunk_count")

    filings_raw = envelope["filings"]
    if not isinstance(filings_raw, list) or not filings_raw:
        raise RecoveryError("manifest filings must be a non-empty list")
    filings = [_validate_filing(row) for row in filings_raw]
    accessions = [filing.accession for filing in filings]
    if accessions != sorted(accessions):
        raise RecoveryError("manifest filings must be sorted by accession")
    if len(accessions) != len(set(accessions)):
        raise RecoveryError("manifest contains duplicate accession numbers")
    if filing_count != len(filings):
        raise RecoveryError("filing_count does not match the exact filing rows")
    if ticker_count != len({filing.ticker for filing in filings}):
        raise RecoveryError("ticker_count does not match the exact filing rows")

    digest = _exact_keys(envelope["digest"], DIGEST_KEYS, "digest")
    expected_digest = digest["payload_sha256"]
    if digest["algorithm"] != "sha256" or not isinstance(expected_digest, str):
        raise RecoveryError("manifest digest must use sha256")
    if SHA256_RE.fullmatch(expected_digest) is None:
        raise RecoveryError("manifest payload_sha256 is invalid")
    payload = {key: value for key, value in envelope.items() if key != "digest"}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if actual_digest != expected_digest:
        raise RecoveryError("manifest payload digest mismatch")
    return expected_digest, filings


def destination_for(download_dir: Path, filing: FilingSpec) -> Path:
    return (
        download_dir
        / "sec-edgar-filings"
        / filing.ticker
        / filing.form
        / filing.accession
        / "full-submission.txt"
    )


def validate_download_dir(download_dir: Path) -> Path:
    resolved = download_dir.expanduser().resolve()
    default = Path(SEC_DOWNLOAD_DIR).expanduser().resolve()
    if resolved == default or tuple(resolved.parts[-2:]) == ("data", "sec_filings"):
        raise RecoveryError("refusing the default production data/sec_filings directory")
    return resolved


def _header_field(header: str, label: str) -> str:
    match = re.search(rf"(?mi)^[ \t]*{re.escape(label)}:[ \t]*([^\r\n]+)", header)
    if match is None:
        raise RecoveryError(f"SEC header is missing {label}")
    return match.group(1).strip()


def validate_submission_header(header_bytes: bytes, filing: FilingSpec) -> None:
    text = header_bytes.decode("latin-1")
    start = text.find("<SEC-HEADER>")
    end = text.find("</SEC-HEADER>", start + 12)
    if start < 0 or end < 0:
        raise RecoveryError("downloaded submission has no complete SEC header")
    header = text[start:end]

    actual = {
        "accession": _header_field(header, "ACCESSION NUMBER"),
        "form": _header_field(header, "CONFORMED SUBMISSION TYPE"),
        "cik": _header_field(header, "CENTRAL INDEX KEY"),
        "filed_date": _header_field(header, "FILED AS OF DATE"),
        "period_of_report": _header_field(header, "CONFORMED PERIOD OF REPORT"),
    }
    expected = {
        "accession": filing.accession,
        "form": filing.form,
        "cik": str(int(filing.cik)),
        "filed_date": filing.filed_date.replace("-", ""),
        "period_of_report": filing.period_of_report.replace("-", ""),
    }
    actual["cik"] = str(int(actual["cik"])) if actual["cik"].isdigit() else actual["cik"]
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise RecoveryError(
                f"SEC header {field} mismatch for {filing.accession}: "
                f"expected {expected_value!r}, got {actual[field]!r}"
            )


def _inspect_file(path: Path, filing: FilingSpec) -> FileRecord:
    digest = hashlib.sha256()
    size = 0
    header = bytearray()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
                if len(header) < MAX_HEADER_BYTES:
                    header.extend(chunk[: MAX_HEADER_BYTES - len(header)])
    except OSError as exc:
        raise RecoveryError(f"cannot read submission {path}: {exc}") from exc
    validate_submission_header(bytes(header), filing)
    return FileRecord(relative_path="", sha256=digest.hexdigest(), size=size)


def _response_status(response: Any) -> int:
    status = response.getcode() if hasattr(response, "getcode") else response.status
    return int(status)


def _download_attempt(
    filing: FilingSpec,
    part_path: Path,
    *,
    user_agent: str,
    timeout: float,
    opener: Callable[..., Any],
) -> FileRecord:
    request = urllib.request.Request(
        filing.full_submission_url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
    )
    digest = hashlib.sha256()
    size = 0
    header = bytearray()
    with opener(request, timeout=timeout) as response:
        status = _response_status(response)
        if status != 200:
            raise _HTTPStatusError(status)
        with part_path.open("wb") as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if len(header) < MAX_HEADER_BYTES:
                    header.extend(chunk[: MAX_HEADER_BYTES - len(header)])
            handle.flush()
            os.fsync(handle.fileno())
    validate_submission_header(bytes(header), filing)
    return FileRecord(relative_path="", sha256=digest.hexdigest(), size=size)


def _transient_status(exc: Exception) -> Optional[int]:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code if exc.code in TRANSIENT_HTTP_STATUS else None
    if isinstance(exc, _HTTPStatusError):
        return exc.status if exc.status in TRANSIENT_HTTP_STATUS else None
    if isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException),
    ):
        return 0
    return None


def recover_one(
    download_dir: Path,
    filing: FilingSpec,
    *,
    user_agent: str,
    limiter: RateLimiter,
    max_attempts: int,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[FileRecord, bool]:
    """Return the verified file record and whether an existing file was resumed."""
    destination = destination_for(download_dir, filing)
    relative_path = destination.relative_to(download_dir).as_posix()
    if destination.is_symlink():
        raise RecoveryError(f"refusing symlink collision: {destination}")
    if destination.exists():
        try:
            record = _inspect_file(destination, filing)
        except RecoveryError as exc:
            raise RecoveryError(f"invalid existing-file collision: {destination}: {exc}") from exc
        return FileRecord(relative_path, record.sha256, record.size), True

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(download_dir):
        raise RecoveryError(f"destination escapes isolated download directory: {destination}")
    part_path = destination.with_name(destination.name + ".part")
    if part_path.is_symlink():
        raise RecoveryError(f"refusing symlink collision: {part_path}")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            record = _download_attempt(
                filing,
                part_path,
                user_agent=user_agent,
                timeout=timeout,
                opener=opener,
            )
            if destination.exists() or destination.is_symlink():
                raise RecoveryError(f"destination appeared during download: {destination}")
            os.replace(part_path, destination)
            return FileRecord(relative_path, record.sha256, record.size), False
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            transient = _transient_status(exc)
            if transient is None or attempt == max_attempts:
                if isinstance(exc, RecoveryError):
                    raise
                raise RecoveryError(
                    f"failed to recover {filing.accession} after {attempt} attempt(s): {exc}"
                ) from exc
            last_error = exc
    raise RecoveryError(f"failed to recover {filing.accession}: {last_error}")


def build_report(manifest_digest: str, records: list[FileRecord]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_payload_sha256": manifest_digest,
        "file_count": len(records),
        "files": [
            {
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "size": record.size,
            }
            for record in records
        ],
    }


def write_report_once(path: Path, report: dict[str, Any]) -> None:
    content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
            return
        raise RecoveryError(f"refusing to overwrite different recovery report: {path}") from exc


def recover_manifest(
    manifest_path: Path,
    download_dir: Path,
    report_path: Path,
    *,
    user_agent: str,
    max_rps: float = 10,
    max_attempts: int = 3,
    timeout: float = 30,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> tuple[dict[str, Any], int]:
    try:
        validate_real_sec_user_agent(user_agent)
    except ValidationError as exc:
        raise RecoveryError(str(exc)) from exc
    if not 1 <= max_attempts <= MAX_ATTEMPTS_LIMIT:
        raise RecoveryError(f"--max-attempts must be between 1 and {MAX_ATTEMPTS_LIMIT}")
    if timeout <= 0:
        raise RecoveryError("--timeout must be greater than 0")
    resolved_download = validate_download_dir(download_dir)
    manifest_digest, filings = load_manifest(manifest_path)
    limiter = RateLimiter(max_rps, clock=clock, sleep=sleep)

    records: list[FileRecord] = []
    resumed = 0
    for filing in filings:
        record, was_resumed = recover_one(
            resolved_download,
            filing,
            user_agent=user_agent,
            limiter=limiter,
            max_attempts=max_attempts,
            timeout=timeout,
            opener=opener,
        )
        records.append(record)
        resumed += int(was_resumed)
        if progress is not None:
            progress(len(records), len(filings), resumed)
    report = build_report(manifest_digest, records)
    write_report_once(report_path, report)
    return report, resumed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-rps", type=float, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    def show_progress(completed: int, total: int, resumed: int) -> None:
        if completed == total or completed % 25 == 0:
            print(
                f"Verified {completed}/{total} submissions ({resumed} resumed)",
                file=sys.stderr,
                flush=True,
            )

    try:
        report, resumed = recover_manifest(
            args.manifest,
            args.download_dir,
            args.report,
            user_agent=os.getenv("SEC_USER_AGENT", SEC_USER_AGENT),
            max_rps=args.max_rps,
            max_attempts=args.max_attempts,
            timeout=args.timeout,
            progress=show_progress,
        )
    except (RecoveryError, OSError) as exc:
        print(f"SEC submission recovery failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Verified {report['file_count']} raw SEC submissions "
        f"({resumed} resumed): {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
