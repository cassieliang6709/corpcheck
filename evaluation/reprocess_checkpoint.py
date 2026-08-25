"""Crash-safe, hash-chained filesystem checkpoints for corpus reprocessing.

中文：重处理过程使用 SHA-256 哈希链，以便崩溃后安全恢复，并检测损坏或未同步
重算摘要的修改。它不是带密钥的认证签名；格式不满足运行契约时必须停止恢复。
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 2
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATABASE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PROFILE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class CheckpointError(RuntimeError):
    """A checkpoint violates the reprocessing run contract."""


@dataclasses.dataclass(frozen=True)
class RunContract:
    """Immutable identity and configuration contract for one reprocessing run.

    中文：将源/目标数据库、冻结输入、表示配置和预期 accession 绑定在一起；任一字段
    漂移都会使检查点不可复用。
    """

    old_database_name: str
    new_database_name: str
    manifest_sha256: str
    recovery_report_sha256: str
    cleaner_source_sha256: str
    representation_profile: str
    profile_source_fingerprint: str
    embedding_model: str
    embedding_dimension: int
    expected_accessions: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_database_name(self.old_database_name, "old database name")
        _validate_database_name(self.new_database_name, "new database name")
        if self.old_database_name == self.new_database_name:
            raise CheckpointError("old and new database names must differ")
        _validate_sha256(self.manifest_sha256, "manifest digest")
        _validate_sha256(self.recovery_report_sha256, "recovery report digest")
        _validate_sha256(self.cleaner_source_sha256, "cleaner source fingerprint")
        if (
            not isinstance(self.representation_profile, str)
            or PROFILE_RE.fullmatch(self.representation_profile) is None
        ):
            raise CheckpointError(
                "representation profile must be a lowercase canonical identifier"
            )
        _validate_sha256(
            self.profile_source_fingerprint,
            "representation profile source fingerprint",
        )
        _validate_name(self.embedding_model, "embedding model")
        if (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension <= 0
        ):
            raise CheckpointError("embedding dimension must be a positive integer")

        try:
            accessions = tuple(self.expected_accessions)
        except TypeError as exc:
            raise CheckpointError("expected accessions must be a sequence") from exc
        if not accessions:
            raise CheckpointError("expected accession set must not be empty")
        if any(ACCESSION_RE.fullmatch(value) is None for value in accessions):
            raise CheckpointError("expected accession set contains an invalid accession")
        canonical = tuple(sorted(set(accessions)))
        if accessions != canonical:
            raise CheckpointError("expected accessions must be unique and sorted")
        object.__setattr__(self, "expected_accessions", accessions)


@dataclasses.dataclass(frozen=True)
class SuccessRecord:
    """Digest-backed evidence that one accession completed reprocessing.

    中文：一条成功记录同时绑定原始文件与生成 chunk 的摘要，防止恢复时只凭 accession
    名称跳过错误版本。
    """

    accession: str
    raw_sha256: str
    chunk_count: int
    chunk_sha256: str

    def __post_init__(self) -> None:
        if ACCESSION_RE.fullmatch(self.accession) is None:
            raise CheckpointError("success record has an invalid accession")
        _validate_sha256(self.raw_sha256, "raw filing digest")
        _validate_sha256(self.chunk_sha256, "chunk digest")
        if (
            isinstance(self.chunk_count, bool)
            or not isinstance(self.chunk_count, int)
            or self.chunk_count <= 0
        ):
            raise CheckpointError("chunk count must be a positive integer")


@dataclasses.dataclass(frozen=True)
class CheckpointState:
    """Validated checkpoint header, completed records, and hash-chain tail.

    中文：解析后的检查点状态；它只在每条哈希链记录及运行契约均通过时存在。
    """

    contract: RunContract
    records: tuple[SuccessRecord, ...]
    last_entry_sha256: str
    checkpoint_sha256: str

    @property
    def completed_accessions(self) -> frozenset[str]:
        return frozenset(record.accession for record in self.records)

    @property
    def complete(self) -> bool:
        return self.completed_accessions == frozenset(self.contract.expected_accessions)


def _validate_name(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise CheckpointError(f"{field} must be non-empty canonical text")


def _validate_database_name(value: str, field: str) -> None:
    if not isinstance(value, str) or DATABASE_NAME_RE.fullmatch(value) is None:
        raise CheckpointError(f"{field} must be a plain database name, not a URL")


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CheckpointError(f"{field} must be a lowercase SHA-256 digest")


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint contains a non-JSON value") from exc
    return rendered.encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signed_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "entry_sha256": _digest(_canonical_json(payload))}


def _entry_line(payload: dict[str, Any]) -> bytes:
    return _canonical_json(_signed_entry(payload)) + b"\n"


def _contract_payload(contract: RunContract) -> dict[str, Any]:
    payload = dataclasses.asdict(contract)
    payload["expected_accessions"] = list(contract.expected_accessions)
    return payload


def _header_payload(contract: RunContract) -> dict[str, Any]:
    return {
        "contract": _contract_payload(contract),
        "kind": "header",
        "schema_version": SCHEMA_VERSION,
    }


def _record_payload(record: SuccessRecord, previous_entry_sha256: str) -> dict[str, Any]:
    return {
        "accession": record.accession,
        "chunk_count": record.chunk_count,
        "chunk_sha256": record.chunk_sha256,
        "kind": "success",
        "previous_entry_sha256": previous_entry_sha256,
        "raw_sha256": record.raw_sha256,
        "schema_version": SCHEMA_VERSION,
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_line(raw_line: bytes, line_number: int) -> dict[str, Any]:
    if not raw_line.endswith(b"\n"):
        raise CheckpointError(f"checkpoint line {line_number} is truncated")
    try:
        value = json.loads(
            raw_line[:-1].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint line {line_number} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint line {line_number} must be a JSON object")
    if raw_line != _canonical_json(value) + b"\n":
        raise CheckpointError(f"checkpoint line {line_number} is not canonical JSON")

    supplied_digest = value.get("entry_sha256")
    _validate_sha256(supplied_digest, f"checkpoint line {line_number} digest")
    unsigned = {key: item for key, item in value.items() if key != "entry_sha256"}
    actual_digest = _digest(_canonical_json(unsigned))
    if not hmac.compare_digest(supplied_digest, actual_digest):
        raise CheckpointError(f"checkpoint line {line_number} digest mismatch")
    return value


def _contract_from_payload(value: Any) -> RunContract:
    if not isinstance(value, dict):
        raise CheckpointError("checkpoint header contract must be an object")
    expected_fields = {field.name for field in dataclasses.fields(RunContract)}
    if set(value) != expected_fields:
        raise CheckpointError("checkpoint header has invalid contract fields")
    accessions = value.get("expected_accessions")
    if not isinstance(accessions, list) or any(
        not isinstance(accession, str) for accession in accessions
    ):
        raise CheckpointError("checkpoint header has invalid expected accessions")
    return RunContract(
        old_database_name=value["old_database_name"],
        new_database_name=value["new_database_name"],
        manifest_sha256=value["manifest_sha256"],
        recovery_report_sha256=value["recovery_report_sha256"],
        cleaner_source_sha256=value["cleaner_source_sha256"],
        representation_profile=value["representation_profile"],
        profile_source_fingerprint=value["profile_source_fingerprint"],
        embedding_model=value["embedding_model"],
        embedding_dimension=value["embedding_dimension"],
        expected_accessions=tuple(accessions),
    )


def _parse_checkpoint(content: bytes) -> CheckpointState:
    if not content:
        raise CheckpointError("checkpoint is empty")
    raw_lines = content.splitlines(keepends=True)
    header = _parse_line(raw_lines[0], 1)
    if set(header) != {"contract", "entry_sha256", "kind", "schema_version"}:
        raise CheckpointError("checkpoint header has invalid fields")
    if (
        header["kind"] != "header"
        or type(header["schema_version"]) is not int
        or header["schema_version"] != SCHEMA_VERSION
    ):
        raise CheckpointError("checkpoint has an unsupported header")
    contract = _contract_from_payload(header["contract"])

    records: list[SuccessRecord] = []
    seen: set[str] = set()
    previous_digest = header["entry_sha256"]
    expected = set(contract.expected_accessions)
    record_fields = {
        "accession",
        "chunk_count",
        "chunk_sha256",
        "entry_sha256",
        "kind",
        "previous_entry_sha256",
        "raw_sha256",
        "schema_version",
    }
    for line_number, raw_line in enumerate(raw_lines[1:], start=2):
        entry = _parse_line(raw_line, line_number)
        if set(entry) != record_fields:
            raise CheckpointError(f"checkpoint line {line_number} has invalid fields")
        if (
            entry["kind"] != "success"
            or type(entry["schema_version"]) is not int
            or entry["schema_version"] != SCHEMA_VERSION
        ):
            raise CheckpointError(f"checkpoint line {line_number} is not a success record")
        _validate_sha256(
            entry["previous_entry_sha256"],
            f"checkpoint line {line_number} previous digest",
        )
        if not hmac.compare_digest(entry["previous_entry_sha256"], previous_digest):
            raise CheckpointError(f"checkpoint line {line_number} breaks the hash chain")
        record = SuccessRecord(
            accession=entry["accession"],
            raw_sha256=entry["raw_sha256"],
            chunk_count=entry["chunk_count"],
            chunk_sha256=entry["chunk_sha256"],
        )
        if record.accession not in expected:
            raise CheckpointError(f"unexpected accession in checkpoint: {record.accession}")
        if record.accession in seen:
            raise CheckpointError(f"duplicate accession in checkpoint: {record.accession}")
        seen.add(record.accession)
        records.append(record)
        previous_digest = entry["entry_sha256"]

    return CheckpointState(
        contract=contract,
        records=tuple(records),
        last_entry_sha256=previous_digest,
        checkpoint_sha256=_digest(content),
    )


def _read_locked(handle: Any) -> tuple[bytes, CheckpointState]:
    handle.seek(0)
    content = handle.read()
    return content, _parse_checkpoint(content)


def _assert_contract(actual: RunContract, expected: Optional[RunContract]) -> None:
    if expected is not None and actual != expected:
        raise CheckpointError("checkpoint belongs to a different run contract")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_checkpoint(path: Path, contract: RunContract) -> CheckpointState:
    """Create the canonical header, or validate an identical existing checkpoint."""
    line = _entry_line(_header_payload(contract))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except FileExistsError:
        return load_checkpoint(path, expected_contract=contract)
    return _parse_checkpoint(line)


def load_checkpoint(
    path: Path, *, expected_contract: Optional[RunContract] = None
) -> CheckpointState:
    """Load and fully validate a checkpoint without modifying it."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"unable to read checkpoint: {path}") from exc
    state = _parse_checkpoint(content)
    _assert_contract(state.contract, expected_contract)
    return state


def append_success(
    path: Path,
    record: SuccessRecord,
    *,
    expected_contract: Optional[RunContract] = None,
) -> CheckpointState:
    """Append and fsync exactly one new canonical success record."""
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_APPEND)
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            content, state = _read_locked(handle)
            _assert_contract(state.contract, expected_contract)
            if record.accession not in state.contract.expected_accessions:
                raise CheckpointError(f"unexpected accession: {record.accession}")
            if record.accession in state.completed_accessions:
                raise CheckpointError(f"duplicate accession: {record.accession}")

            line = _entry_line(_record_payload(record, state.last_entry_sha256))
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            return _parse_checkpoint(content + line)
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint does not exist: {path}") from exc


def _final_report_payload(state: CheckpointState) -> dict[str, Any]:
    return {
        "checkpoint_sha256": state.checkpoint_sha256,
        "contract": _contract_payload(state.contract),
        "kind": "reprocess_final_report",
        "records": [
            dataclasses.asdict(record)
            for record in sorted(state.records, key=lambda item: item.accession)
        ],
        "schema_version": SCHEMA_VERSION,
    }


def write_final_report(
    checkpoint_path: Path,
    output_path: Path,
    *,
    expected_contract: Optional[RunContract] = None,
) -> dict[str, Any]:
    """Write a final report only after every expected accession succeeded exactly once."""
    state = load_checkpoint(checkpoint_path, expected_contract=expected_contract)
    if not state.complete:
        missing = sorted(set(state.contract.expected_accessions) - state.completed_accessions)
        raise CheckpointError(
            f"checkpoint is incomplete; {len(missing)} expected accessions are missing"
        )

    report = _signed_entry(_final_report_payload(state))
    encoded = _canonical_json(report) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(output_path.parent)
    except FileExistsError as exc:
        if output_path.is_file() and output_path.read_bytes() == encoded:
            return report
        raise CheckpointError(
            f"refusing to overwrite different final report: {output_path}"
        ) from exc
    return report
