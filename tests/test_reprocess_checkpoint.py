from __future__ import annotations

import dataclasses
import json

import pytest
from evaluation import reprocess_checkpoint as checkpoint

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
ACCESSION_A = "0000320187-18-000142"
ACCESSION_B = "0001018724-20-000004"


def contract() -> checkpoint.RunContract:
    return checkpoint.RunContract(
        old_database_name="corpcheck_old",
        new_database_name="corpcheck_reprocessed",
        manifest_sha256=SHA_A,
        recovery_report_sha256=SHA_B,
        cleaner_source_sha256=SHA_C,
        representation_profile="candidate",
        profile_source_fingerprint=SHA_A,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        expected_accessions=(ACCESSION_A, ACCESSION_B),
    )


def success(accession: str = ACCESSION_A) -> checkpoint.SuccessRecord:
    return checkpoint.SuccessRecord(
        accession=accession,
        raw_sha256=SHA_A,
        chunk_count=12,
        chunk_sha256=SHA_B,
    )


def test_create_and_append_are_canonical_fsynced_and_round_trip(tmp_path, monkeypatch) -> None:
    fsync_calls = []
    open_calls = []
    real_open = checkpoint.os.open

    def tracking_open(path, flags, *args):
        open_calls.append((path, flags))
        return real_open(path, flags, *args)

    monkeypatch.setattr(checkpoint.os, "fsync", fsync_calls.append)
    monkeypatch.setattr(checkpoint.os, "open", tracking_open)
    path = tmp_path / "checkpoint.jsonl"

    initial = checkpoint.create_checkpoint(path, contract())
    state = checkpoint.append_success(path, success(), expected_contract=contract())

    assert not initial.complete
    assert state.records == (success(),)
    assert fsync_calls
    assert any(
        candidate == path and flags & checkpoint.os.O_APPEND for candidate, flags in open_calls
    )
    for raw_line in path.read_bytes().splitlines(keepends=True):
        parsed = json.loads(raw_line)
        assert raw_line == (
            json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    assert checkpoint.load_checkpoint(path, expected_contract=contract()) == state


def test_create_is_idempotent_only_for_the_same_contract(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.create_checkpoint(path, contract())

    different = dataclasses.replace(contract(), embedding_dimension=768)
    with pytest.raises(checkpoint.CheckpointError, match="different run contract"):
        checkpoint.create_checkpoint(path, different)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"old_database_name": "same", "new_database_name": "same"}, "must differ"),
        (
            {"old_database_name": "postgresql://user:secret@localhost/old"},
            "plain database name",
        ),
        ({"manifest_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"representation_profile": "Candidate"}, "canonical identifier"),
        ({"representation_profile": "../candidate"}, "canonical identifier"),
        ({"profile_source_fingerprint": "A" * 64}, "lowercase SHA-256"),
        ({"embedding_dimension": True}, "positive integer"),
        ({"expected_accessions": (ACCESSION_B, ACCESSION_A)}, "unique and sorted"),
        ({"expected_accessions": (ACCESSION_A, ACCESSION_A)}, "unique and sorted"),
    ],
)
def test_contract_rejects_ambiguous_identity(changes, message) -> None:
    with pytest.raises(checkpoint.CheckpointError, match=message):
        dataclasses.replace(contract(), **changes)


def test_append_rejects_duplicate_and_unexpected_accessions(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.append_success(path, success())

    with pytest.raises(checkpoint.CheckpointError, match="duplicate accession"):
        checkpoint.append_success(path, success())
    with pytest.raises(checkpoint.CheckpointError, match="unexpected accession"):
        checkpoint.append_success(path, success("0000320193-23-000106"))


def test_load_rejects_tampered_payload_even_when_json_is_valid(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.append_success(path, success())
    content = path.read_text()
    path.write_text(content.replace('"chunk_count":12', '"chunk_count":13'))

    with pytest.raises(checkpoint.CheckpointError, match="digest mismatch"):
        checkpoint.load_checkpoint(path)


def test_load_rejects_legacy_v1_checkpoint_without_profile_binding(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    legacy_contract = dataclasses.asdict(contract())
    legacy_contract.pop("representation_profile")
    legacy_contract.pop("profile_source_fingerprint")
    legacy_contract["expected_accessions"] = list(
        legacy_contract["expected_accessions"]
    )
    path.write_bytes(
        checkpoint._entry_line(
            {
                "contract": legacy_contract,
                "kind": "header",
                "schema_version": 1,
            }
        )
    )

    with pytest.raises(checkpoint.CheckpointError, match="unsupported header"):
        checkpoint.load_checkpoint(path)


def test_load_rejects_broken_chain_and_duplicate_records(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.append_success(path, success(ACCESSION_A))
    checkpoint.append_success(path, success(ACCESSION_B))
    lines = path.read_bytes().splitlines(keepends=True)

    path.write_bytes(lines[0] + lines[2] + lines[1])
    with pytest.raises(checkpoint.CheckpointError, match="hash chain"):
        checkpoint.load_checkpoint(path)

    first_record = json.loads(lines[1])
    duplicate_payload = checkpoint._record_payload(
        success(ACCESSION_A), first_record["entry_sha256"]
    )
    duplicate_line = checkpoint._entry_line(duplicate_payload)
    path.write_bytes(lines[0] + lines[1] + duplicate_line)
    with pytest.raises(checkpoint.CheckpointError, match="duplicate accession"):
        checkpoint.load_checkpoint(path)


@pytest.mark.parametrize("mutation", ["truncate", "noncanonical", "blank"])
def test_load_rejects_truncated_or_noncanonical_lines(tmp_path, mutation) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint.create_checkpoint(path, contract())
    original = path.read_bytes()
    if mutation == "truncate":
        path.write_bytes(original[:-1])
        message = "truncated"
    elif mutation == "noncanonical":
        path.write_bytes(original.replace(b"{", b"{ ", 1))
        message = "not canonical"
    else:
        path.write_bytes(original + b"\n")
        message = "invalid JSON"

    with pytest.raises(checkpoint.CheckpointError, match=message):
        checkpoint.load_checkpoint(path)


def test_whole_record_tail_loss_is_incomplete_not_success(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "final.json"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.append_success(path, success(ACCESSION_A))
    checkpoint.append_success(path, success(ACCESSION_B))
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))

    state = checkpoint.load_checkpoint(path)
    assert not state.complete
    with pytest.raises(checkpoint.CheckpointError, match="incomplete"):
        checkpoint.write_final_report(path, output)


def test_final_report_requires_exact_completion_and_is_write_once(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "final.json"
    checkpoint.create_checkpoint(path, contract())
    checkpoint.append_success(path, success(ACCESSION_A))
    with pytest.raises(checkpoint.CheckpointError, match="1 expected accessions"):
        checkpoint.write_final_report(path, output)

    checkpoint.append_success(path, success(ACCESSION_B))
    report = checkpoint.write_final_report(path, output, expected_contract=contract())
    assert checkpoint.write_final_report(path, output) == report
    assert [row["accession"] for row in report["records"]] == [
        ACCESSION_A,
        ACCESSION_B,
    ]
    assert report["checkpoint_sha256"] == checkpoint.load_checkpoint(path).checkpoint_sha256

    output.write_text("different\n")
    with pytest.raises(checkpoint.CheckpointError, match="refusing to overwrite"):
        checkpoint.write_final_report(path, output)
