from __future__ import annotations

import json

import pytest
from evaluation import materialize_financebench_holdout as materializer
from evaluation.financebench import parse_gold_doc


def row(identifier: str, company: str, doc_name: str) -> dict:
    return {
        "financebench_id": identifier,
        "company": company,
        "doc_name": doc_name,
        "question": f"Question {identifier}",
        "answer": "Answer",
        "evidence": [],
    }


def test_select_holdout_is_company_disjoint_and_enriches_provenance() -> None:
    development = [row("dev", "Amazon", "AMAZON_2019_10K")]
    source = [
        development[0],
        row("same-company", "Amazon", "AMAZON_2017_10K"),
        row("held", "3M", "3M_2022_10K"),
        row("unsupported-form", "American Express", "AMERICANEXPRESS_2022Q1_10Q"),
    ]

    selected = materializer.select_holdout(source, development)

    assert [item["financebench_id"] for item in selected] == ["held"]
    assert selected[0]["doc_type"] == "10k"
    assert selected[0]["doc_period"] == 2022
    gold_doc = parse_gold_doc(selected[0])
    assert gold_doc.is_resolvable
    assert gold_doc.ticker == "MMM"


def test_select_holdout_sorts_by_financebench_id() -> None:
    development = [row("dev", "Amazon", "AMAZON_2019_10K")]
    source = [
        development[0],
        row("financebench_id_00002", "Boeing", "BOEING_2022_10K"),
        row("financebench_id_00001", "Adobe", "ADOBE_2022_10K"),
    ]

    selected = materializer.select_holdout(source, development)

    assert [item["financebench_id"] for item in selected] == [
        "financebench_id_00001",
        "financebench_id_00002",
    ]


def test_select_holdout_rejects_duplicate_or_missing_development_ids() -> None:
    development = [row("dev", "Amazon", "AMAZON_2019_10K")]
    with pytest.raises(materializer.ContractError, match="duplicate"):
        materializer.select_holdout(
            [development[0], development[0]],
            development,
        )

    with pytest.raises(materializer.ContractError, match="absent from source"):
        materializer.select_holdout(
            [row("other", "3M", "3M_2022_10K")],
            development,
        )


def test_select_holdout_rejects_unresolved_company() -> None:
    development = [row("dev", "Amazon", "AMAZON_2019_10K")]
    source = [development[0], row("held", "Unknown Issuer", "UNKNOWN_2022_10K")]

    with pytest.raises(materializer.ContractError, match="unresolved provenance"):
        materializer.select_holdout(source, development)


def test_validate_frozen_counts_rejects_drift() -> None:
    with pytest.raises(materializer.ContractError, match="frozen contract"):
        materializer.validate_frozen_counts(
            [row("held", "3M", "3M_2022_10K")]
        )


def test_build_ingestion_manifest_deduplicates_documents() -> None:
    rows = [
        {
            **row("held-2", "American Express", "AMERICANEXPRESS_2022_10K"),
            "doc_type": "10k",
            "doc_period": 2022,
        },
        {
            **row("held-1", "American Express", "AMERICANEXPRESS_2022_10K"),
            "doc_type": "10k",
            "doc_period": 2022,
        },
    ]

    manifest = materializer.build_ingestion_manifest(rows)

    assert manifest == [
        {
            "doc_name": "AMERICANEXPRESS_2022_10K",
            "company": "American Express",
            "ticker": "AXP",
            "filing_type": "10-K",
            "fiscal_year": 2022,
        }
    ]


def test_write_once_is_idempotent_and_refuses_different_output(tmp_path) -> None:
    output = tmp_path / "heldout.json"
    content = json.dumps([{"financebench_id": "held"}]) + "\n"

    materializer.write_once(output, content)
    materializer.write_once(output, content)

    with pytest.raises(materializer.ContractError, match="refusing to overwrite"):
        materializer.write_once(output, "[]\n")


def test_validate_content_hashes_rejects_member_drift(monkeypatch) -> None:
    output = "locked output\n"
    manifest = "locked manifest\n"
    monkeypatch.setattr(
        materializer,
        "EXPECTED_OUTPUT_SHA256",
        materializer.sha256_text(output),
    )
    monkeypatch.setattr(
        materializer,
        "EXPECTED_MANIFEST_SHA256",
        materializer.sha256_text(manifest),
    )

    materializer.validate_content_hashes(output, manifest)
    with pytest.raises(materializer.ContractError, match="held-out content"):
        materializer.validate_content_hashes("drifted\n", manifest)
    with pytest.raises(materializer.ContractError, match="ingestion manifest"):
        materializer.validate_content_hashes(output, "drifted\n")


def test_main_checks_locked_hashes_and_all_outputs_before_writing(
    monkeypatch, tmp_path
) -> None:
    development = [row("dev", "Amazon", "AMAZON_2019_10K")]
    source = [development[0], row("held", "Boeing", "BOEING_2022_10K")]
    source_path = tmp_path / "source.jsonl"
    development_path = tmp_path / "development.json"
    output_path = tmp_path / "heldout.json"
    manifest_path = tmp_path / "manifest.json"
    source_path.write_text(
        "".join(json.dumps(item) + "\n" for item in source),
        encoding="utf-8",
    )
    development_path.write_text(json.dumps(development), encoding="utf-8")

    selected = materializer.select_holdout(source, development)
    manifest = materializer.build_ingestion_manifest(selected)
    output_content = materializer.render_json(selected)
    manifest_content = materializer.render_json(manifest)
    monkeypatch.setattr(materializer, "SOURCE_SHA256", materializer.sha256_file(source_path))
    monkeypatch.setattr(
        materializer,
        "DEVELOPMENT_SHA256",
        materializer.sha256_file(development_path),
    )
    monkeypatch.setattr(materializer, "EXPECTED_SOURCE_ROWS", 2)
    monkeypatch.setattr(materializer, "EXPECTED_HELDOUT_ROWS", 1)
    monkeypatch.setattr(materializer, "EXPECTED_HELDOUT_DOCUMENTS", 1)
    monkeypatch.setattr(materializer, "EXPECTED_HELDOUT_COMPANIES", 1)
    monkeypatch.setattr(
        materializer,
        "EXPECTED_OUTPUT_SHA256",
        materializer.sha256_text(output_content),
    )
    monkeypatch.setattr(
        materializer,
        "EXPECTED_MANIFEST_SHA256",
        materializer.sha256_text(manifest_content),
    )

    manifest_path.write_text("different\n", encoding="utf-8")
    with pytest.raises(materializer.ContractError, match="refusing to overwrite"):
        materializer.main(
            [
                "--source",
                str(source_path),
                "--development",
                str(development_path),
                "--output",
                str(output_path),
                "--manifest-output",
                str(manifest_path),
            ]
        )
    assert not output_path.exists()

    manifest_path.unlink()
    assert (
        materializer.main(
            [
                "--source",
                str(source_path),
                "--development",
                str(development_path),
                "--output",
                str(output_path),
                "--manifest-output",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert output_path.read_text(encoding="utf-8") == output_content
    assert manifest_path.read_text(encoding="utf-8") == manifest_content
