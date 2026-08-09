#!/usr/bin/env python3
"""Validate immutable SEC recovery inputs before corpus reprocessing.

This module is deliberately read-only. Database mutation and checkpointing belong
to a later reprocessing stage and must only consume the verified records returned
here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import numpy as np

# Reprocessing consumes only the already verified recovery tree.  Model loading
# must use the local cache and fail instead of reaching an external service.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from corpcheck.ingestion.config import EMBEDDING_DIM
from evaluation.recover_sec_submissions import (
    CHUNK_BYTES,
    REPORT_SCHEMA_VERSION,
    SHA256_RE,
    FilingSpec,
    RecoveryError,
    destination_for,
    load_manifest,
)

REPORT_KEYS = {
    "schema_version",
    "manifest_payload_sha256",
    "file_count",
    "files",
}
FILE_KEYS = {"relative_path", "sha256", "size"}
EMBEDDING_DIMENSION = EMBEDDING_DIM

_COMPANY_IDENTITY_SQL = """
    SELECT ticker, name, sector, industry, market_cap, description
    FROM companies
    ORDER BY ticker
"""
_FILING_IDENTITY_SQL = """
    SELECT id, ticker, filing_type, fiscal_year, period, filed_date,
           period_of_report, accession_number, cik, source_url, local_path
    FROM filings
    ORDER BY accession_number, id
"""
_RUNTIME_IDENTITY_SQL = """
    SELECT current_database() AS database_name,
           COALESCE(inet_server_addr()::text, '') AS server_address,
           inet_server_port() AS server_port
"""


class ReprocessInputError(RuntimeError):
    """The manifest, recovery report, or recovered files are inconsistent."""


@dataclass(frozen=True)
class DatabaseEndpoint:
    host: str
    port: int
    database_name: str


@dataclass(frozen=True)
class CorpusIdentity:
    database_name: str
    company_count: int
    filing_count: int
    identity_sha256: str


@dataclass(frozen=True)
class DatabasePreflight:
    old: CorpusIdentity
    new: CorpusIdentity


@dataclass(frozen=True)
class FilingTarget:
    filing_id: int
    ticker: str
    sector: str
    form: str
    fiscal_year: int
    period: str
    filed_date: str
    period_of_report: str
    accession: str
    cik: str
    source_url: str


@dataclass(frozen=True)
class VerifiedSubmission:
    filing: FilingSpec
    path: Path
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ReprocessingInputs:
    manifest_path: Path
    manifest_payload_sha256: str
    recovery_report_path: Path
    recovery_root: Path
    submissions: tuple[VerifiedSubmission, ...]


@dataclass(frozen=True)
class _ReportFile:
    relative_path: str
    sha256: str
    size: int


def parse_database_endpoint(database_url: str) -> DatabaseEndpoint:
    """Parse an explicit PostgreSQL URL without retaining credentials."""
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ReprocessInputError("database URLs must be explicit PostgreSQL URLs")
    database_name = unquote(parsed.path.lstrip("/"))
    if not database_name or "/" in database_name:
        raise ReprocessInputError("database URLs must each name exactly one database")
    try:
        port = parsed.port or 5432
    except ValueError as exc:
        raise ReprocessInputError("database URL has an invalid port") from exc
    return DatabaseEndpoint(parsed.hostname.lower(), port, database_name)


def validate_database_pair(
    old_database_url: str,
    new_database_url: str,
) -> tuple[DatabaseEndpoint, DatabaseEndpoint]:
    """Require two distinct database names on the same explicitly named server."""
    old = parse_database_endpoint(old_database_url)
    new = parse_database_endpoint(new_database_url)
    if (old.host, old.port) != (new.host, new.port):
        raise ReprocessInputError("old and new databases must be on the same server")
    if old.database_name == new.database_name:
        raise ReprocessInputError("old and new database names must differ")
    return old, new


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_type = type(value).__name__
    raise ReprocessInputError(
        f"database identity contains unsupported value type: {value_type}"
    )


def _row_mapping(row: Any, columns: Sequence[str]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {column: row[column] for column in columns}
    if len(row) != len(columns):
        raise ReprocessInputError("database identity query returned an invalid row shape")
    return dict(zip(columns, row, strict=True))


def _fetch_rows(connection: Any, sql: str) -> tuple[list[str], list[dict[str, Any]]]:
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        if cursor.description is None:
            raise ReprocessInputError("database identity query returned no columns")
        columns = [item[0] for item in cursor.description]
        return columns, [_row_mapping(row, columns) for row in cursor.fetchall()]
    finally:
        cursor.close()


def _runtime_identity(connection: Any) -> DatabaseEndpoint:
    columns, rows = _fetch_rows(connection, _RUNTIME_IDENTITY_SQL)
    if len(rows) != 1:
        raise ReprocessInputError("database runtime identity query returned an invalid result")
    row = rows[0]
    if set(columns) != {"database_name", "server_address", "server_port"}:
        raise ReprocessInputError("database runtime identity query returned invalid columns")
    database_name = row["database_name"]
    address = row["server_address"]
    port = row["server_port"]
    if not isinstance(database_name, str) or not database_name:
        raise ReprocessInputError("database runtime identity has an invalid database name")
    if not isinstance(address, str) or isinstance(port, bool) or not isinstance(port, int):
        raise ReprocessInputError("database runtime identity has an invalid server")
    return DatabaseEndpoint(address, port, database_name)


def _corpus_identity(
    connection: Any,
    database_name: str,
) -> tuple[CorpusIdentity, list[dict[str, Any]], list[dict[str, Any]]]:
    company_columns, companies = _fetch_rows(connection, _COMPANY_IDENTITY_SQL)
    filing_columns, filings = _fetch_rows(connection, _FILING_IDENTITY_SQL)
    expected_company_columns = [
        "ticker", "name", "sector", "industry", "market_cap", "description"
    ]
    expected_filing_columns = [
        "id", "ticker", "filing_type", "fiscal_year", "period", "filed_date",
        "period_of_report", "accession_number", "cik", "source_url", "local_path",
    ]
    if company_columns != expected_company_columns or filing_columns != expected_filing_columns:
        raise ReprocessInputError(
            "database identity schema does not match the reprocessor contract"
        )
    if not companies or not filings:
        raise ReprocessInputError("database corpus identity must not be empty")
    accessions = [row["accession_number"] for row in filings]
    if any(not isinstance(value, str) or not value for value in accessions):
        raise ReprocessInputError("database corpus contains a missing accession")
    if len(accessions) != len(set(accessions)):
        raise ReprocessInputError("database corpus contains duplicate accessions")
    payload = {"companies": _json_value(companies), "filings": _json_value(filings)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return (
        CorpusIdentity(database_name, len(companies), len(filings), digest),
        companies,
        filings,
    )


def preflight_databases(
    old_connection: Any,
    new_connection: Any,
    *,
    old_endpoint: DatabaseEndpoint,
    new_endpoint: DatabaseEndpoint,
) -> DatabasePreflight:
    """Fail unless both connections are the requested matched corpus pair."""
    old_runtime = _runtime_identity(old_connection)
    new_runtime = _runtime_identity(new_connection)
    if old_runtime.database_name != old_endpoint.database_name:
        raise ReprocessInputError("old connection opened a different database than requested")
    if new_runtime.database_name != new_endpoint.database_name:
        raise ReprocessInputError("new connection opened a different database than requested")
    if (old_runtime.host, old_runtime.port) != (new_runtime.host, new_runtime.port):
        raise ReprocessInputError("runtime databases are not on the same server")
    if old_runtime.database_name == new_runtime.database_name:
        raise ReprocessInputError("runtime old and new databases must differ")

    old_identity, old_companies, old_filings = _corpus_identity(
        old_connection, old_runtime.database_name
    )
    new_identity, new_companies, new_filings = _corpus_identity(
        new_connection, new_runtime.database_name
    )
    if _json_value(old_companies) != _json_value(new_companies):
        raise ReprocessInputError("old and new company identities do not match exactly")
    if _json_value(old_filings) != _json_value(new_filings):
        raise ReprocessInputError("old and new filing identities do not match exactly")
    if old_identity.identity_sha256 != new_identity.identity_sha256:
        raise ReprocessInputError("old and new corpus identity digests do not match")
    return DatabasePreflight(old_identity, new_identity)


def _regular_input_path(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ReprocessInputError(f"cannot inspect {label} {candidate}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ReprocessInputError(f"{label} must be a non-symlink regular file: {candidate}")
    return candidate


def _recovery_root(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ReprocessInputError(f"cannot inspect recovery root {candidate}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ReprocessInputError(
            f"recovery root must be a non-symlink directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReprocessInputError(
            f"{label} must contain exactly: {', '.join(sorted(keys))}"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReprocessInputError(f"{label} must be a non-negative integer")
    return value


def _relative_path(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReprocessInputError(f"recovery file {index} has invalid relative_path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ReprocessInputError(
            f"recovery file {index} has non-deterministic relative_path: {value!r}"
        )
    return value


def _load_report(path: Path) -> tuple[str, list[_ReportFile]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReprocessInputError(f"cannot read recovery report {path}: {exc}") from exc

    report = _exact_mapping(raw, REPORT_KEYS, "recovery report")
    if (
        isinstance(report["schema_version"], bool)
        or report["schema_version"] != REPORT_SCHEMA_VERSION
    ):
        raise ReprocessInputError(
            f"unsupported recovery report schema_version: {report['schema_version']!r}"
        )
    manifest_digest = report["manifest_payload_sha256"]
    if not isinstance(manifest_digest, str) or SHA256_RE.fullmatch(manifest_digest) is None:
        raise ReprocessInputError("recovery report manifest_payload_sha256 is invalid")

    files_raw = report["files"]
    if not isinstance(files_raw, list):
        raise ReprocessInputError("recovery report files must be a list")
    file_count = _nonnegative_int(report["file_count"], "recovery report file_count")
    if file_count != len(files_raw):
        raise ReprocessInputError("recovery report file_count does not match files")

    files: list[_ReportFile] = []
    for index, raw_file in enumerate(files_raw):
        row = _exact_mapping(raw_file, FILE_KEYS, f"recovery file {index}")
        relative_path = _relative_path(row["relative_path"], index)
        digest = row["sha256"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ReprocessInputError(f"recovery file {index} has invalid sha256")
        size = _nonnegative_int(row["size"], f"recovery file {index} size")
        files.append(_ReportFile(relative_path, digest, size))
    return manifest_digest, files


def _expected_relative_path(filing: FilingSpec) -> str:
    return destination_for(Path(), filing).as_posix()


def _assert_exact_file_set(files: list[_ReportFile], filings: list[FilingSpec]) -> None:
    actual_paths = [record.relative_path for record in files]
    duplicates = sorted(
        path for path in set(actual_paths) if actual_paths.count(path) > 1
    )
    if duplicates:
        raise ReprocessInputError(
            f"recovery report contains duplicate relative paths: {duplicates}"
        )

    expected_paths = [_expected_relative_path(filing) for filing in filings]
    missing = sorted(set(expected_paths) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_paths))
    if missing or extra:
        raise ReprocessInputError(
            f"recovery report does not match manifest; missing={missing}, extra={extra}"
        )
    if actual_paths != expected_paths:
        raise ReprocessInputError(
            "recovery report files must follow deterministic manifest accession order"
        )


def _confined_regular_file(root: Path, relative_path: str) -> Path:
    current = root
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ReprocessInputError(f"cannot inspect recovered path {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ReprocessInputError(f"recovered path must not contain symlinks: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ReprocessInputError(f"recovered path parent is not a directory: {current}")
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise ReprocessInputError(f"recovered submission is not a regular file: {current}")

    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ReprocessInputError(f"recovered path escapes recovery root: {relative_path}")
    return resolved


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReprocessInputError(f"cannot open recovered submission {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ReprocessInputError(f"recovered submission is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def load_reprocessing_inputs(
    manifest_path: Path,
    recovery_report_path: Path,
    recovery_root: Path,
) -> ReprocessingInputs:
    """Return only submissions proven to match the signed manifest and report."""
    manifest = _regular_input_path(manifest_path, "manifest")
    report = _regular_input_path(recovery_report_path, "recovery report")
    root = _recovery_root(recovery_root)
    try:
        manifest_digest, filings = load_manifest(manifest)
    except RecoveryError as exc:
        raise ReprocessInputError(str(exc)) from exc
    report_digest, report_files = _load_report(report)
    if report_digest != manifest_digest:
        raise ReprocessInputError("recovery report manifest digest does not match manifest")
    _assert_exact_file_set(report_files, filings)

    submissions: list[VerifiedSubmission] = []
    for filing, record in zip(filings, report_files, strict=True):
        path = _confined_regular_file(root, record.relative_path)
        actual_digest, actual_size = _hash_regular_file(path)
        if actual_digest != record.sha256:
            raise ReprocessInputError(
                f"recovered submission sha256 mismatch: {record.relative_path}"
            )
        if actual_size != record.size:
            raise ReprocessInputError(
                f"recovered submission size mismatch: {record.relative_path}"
            )
        submissions.append(
            VerifiedSubmission(
                filing=filing,
                path=path,
                relative_path=record.relative_path,
                sha256=actual_digest,
                size=actual_size,
            )
        )

    return ReprocessingInputs(
        manifest_path=manifest,
        manifest_payload_sha256=manifest_digest,
        recovery_report_path=report,
        recovery_root=root,
        submissions=tuple(submissions),
    )


def validate_filing_target(submission: VerifiedSubmission, target: FilingTarget) -> None:
    """Require the target row to be the exact manifest filing, not a year/ticker guess."""
    filing = submission.filing
    if isinstance(target.filing_id, bool) or target.filing_id <= 0:
        raise ReprocessInputError(f"{filing.accession} target has an invalid filing id")
    if not target.sector or target.sector != target.sector.strip():
        raise ReprocessInputError(f"{filing.accession} target has an invalid company sector")
    expected = {
        "ticker": filing.ticker,
        "form": filing.form,
        "fiscal_year": filing.fiscal_year,
        "period": filing.period,
        "filed_date": filing.filed_date,
        "period_of_report": filing.period_of_report,
        "accession": filing.accession,
        "cik": filing.cik,
        "source_url": filing.source_url,
    }
    actual = {
        key: getattr(target, key)
        for key in expected
    }
    if actual != expected:
        mismatches = sorted(key for key in expected if actual[key] != expected[key])
        raise ReprocessInputError(
            f"{filing.accession} target metadata does not match manifest: {mismatches}"
        )


def _default_cleaner_factory(form: str) -> Any:
    from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner

    return HTMLCleaner(form)


def _default_chunker() -> Any:
    from corpcheck.ingestion.processors.chunker import Chunker

    return Chunker()


def _default_embedder() -> Any:
    from corpcheck.ingestion.processors.embedder import Embedder

    return Embedder()


def process_one_submission(
    submission: VerifiedSubmission,
    target: FilingTarget,
    *,
    cleaner_factory: Callable[[str], Any] = _default_cleaner_factory,
    chunker: Any = None,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """Convert one verified raw submission into complete validated DB chunk rows.

    This function performs no database writes and no discovery or download calls.
    Callers must atomically replace the target filing only after it returns.
    """
    validate_filing_target(submission, target)
    digest, size = _hash_regular_file(submission.path)
    if digest != submission.sha256 or size != submission.size:
        raise ReprocessInputError(
            f"{submission.filing.accession} raw submission changed after input validation"
        )

    cleaner = cleaner_factory(submission.filing.form)
    segments = cleaner.clean_segments(submission.path)
    if not segments:
        raise ReprocessInputError(
            f"{submission.filing.accession} cleaner produced no segments"
        )
    if any(
        not isinstance(segment.text, str) or not segment.text.strip()
        for segment in segments
    ):
        raise ReprocessInputError(
            f"{submission.filing.accession} cleaner produced an empty segment"
        )
    active_chunker = chunker if chunker is not None else _default_chunker()
    payloads = active_chunker.chunk_segments(segments)
    if not payloads:
        raise ReprocessInputError(
            f"{submission.filing.accession} chunker produced no chunks"
        )
    if any(not payload.text or not payload.text.strip() for payload in payloads):
        raise ReprocessInputError(
            f"{submission.filing.accession} chunker produced empty content"
        )

    active_embedder = embedder if embedder is not None else _default_embedder()
    embeddings = np.asarray(
        active_embedder.encode([payload.text for payload in payloads]),
        dtype=np.float32,
    )
    expected_shape = (len(payloads), EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape:
        raise ReprocessInputError(
            f"{submission.filing.accession} embeddings have shape {embeddings.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ReprocessInputError(
            f"{submission.filing.accession} embeddings contain non-finite values"
        )

    from corpcheck.ingestion.chunk_features import compute_chunk_features

    rows: list[dict[str, Any]] = []
    for index, (payload, embedding) in enumerate(zip(payloads, embeddings, strict=True)):
        token_count = payload.token_count
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ReprocessInputError(
                f"{submission.filing.accession} chunk {index} has invalid token_count"
            )
        features = compute_chunk_features(payload.section_name, payload.text)
        rows.append(
            {
                "filing_id": target.filing_id,
                "ticker": target.ticker,
                "sector": target.sector,
                "filing_type": target.form,
                "fiscal_year": target.fiscal_year,
                "period": target.period,
                "filed_date": target.filed_date,
                "section_name": payload.section_name,
                "chunk_index": index,
                "content": payload.text,
                "char_count": len(payload.text),
                "token_count": token_count,
                "numeric_token_count": features["numeric_token_count"],
                "number_density": features["number_density"],
                "data_signal_score": features["data_signal_score"],
                "is_quantitative": features["is_quantitative"],
                "content_kind": payload.content_kind,
                "chunk_strategy": payload.chunk_strategy,
                "display_title": payload.display_title,
                "chunk_group_key": payload.chunk_group_key,
                "structure_meta": payload.structure_meta,
                "embedding": embedding,
                "source_url": target.source_url,
            }
        )
    return rows


def processing_source_fingerprint() -> str:
    """Bind a checkpoint contract to the exact local processing implementation."""
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "evaluation/reprocess_sec_corpus.py",
        "src/corpcheck/ingestion/chunk_features.py",
        "src/corpcheck/ingestion/config.py",
        "src/corpcheck/ingestion/loaders/db_loader.py",
        "src/corpcheck/ingestion/processors/chunker.py",
        "src/corpcheck/ingestion/processors/embedder.py",
        "src/corpcheck/ingestion/processors/html_cleaner.py",
        "src/corpcheck/ingestion/processors/segment_types.py",
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ReprocessInputError(f"cannot fingerprint processing source {path}") from exc
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()
