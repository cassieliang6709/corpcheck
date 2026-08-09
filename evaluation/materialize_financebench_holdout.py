#!/usr/bin/env python3
"""Materialize the frozen company-disjoint FinanceBench 10-K held-out set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from evaluation.financebench import parse_gold_doc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = REPO_ROOT / "evaluation/datasets/financebench_filtered.json"

SOURCE_SHA256 = "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877"
DEVELOPMENT_SHA256 = "1fdce65e8fe0ae03f5dcee732b8e70d9a22969d95981359c9505e105a9ca0f5f"
EXPECTED_SOURCE_ROWS = 150
EXPECTED_HELDOUT_ROWS = 80
EXPECTED_HELDOUT_DOCUMENTS = 45
EXPECTED_HELDOUT_COMPANIES = 21
EXPECTED_OUTPUT_SHA256 = "f2d8ba8b3f1717166c862cc320c8c7a7678d19a4f3dd9c9438f1a74519ad5eae"
EXPECTED_MANIFEST_SHA256 = "2b17d7354f4a49f78f249422d49d3df9269d2c0977da0403ed878671f7aa4b10"

_TEN_K_DOC_RE = re.compile(r"_(\d{4})_10K$", re.I)


class ContractError(ValueError):
    """The source, development set, or derived split violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ContractError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def load_json_list(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError(f"{path} must contain a JSON list of objects")
    return rows


def _unique_ids(rows: Sequence[dict[str, Any]], label: str) -> set[str]:
    ids = [str(row.get("financebench_id") or "") for row in rows]
    if any(not value for value in ids):
        raise ContractError(f"{label} contains a row without financebench_id")
    if len(ids) != len(set(ids)):
        raise ContractError(f"{label} contains duplicate financebench_id values")
    return set(ids)


def select_holdout(
    source_rows: Sequence[dict[str, Any]],
    development_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the frozen metadata-only selection and derive provenance fields."""
    source_ids = _unique_ids(source_rows, "source")
    development_ids = _unique_ids(development_rows, "development set")
    missing_development = development_ids - source_ids
    if missing_development:
        raise ContractError(
            f"development IDs are absent from source: {sorted(missing_development)}"
        )

    development_companies = {
        str(row.get("company") or "").strip() for row in development_rows
    }
    selected: list[dict[str, Any]] = []
    for row in source_rows:
        if str(row["financebench_id"]) in development_ids:
            continue
        company = str(row.get("company") or "").strip()
        if not company or company in development_companies:
            continue
        doc_name = str(row.get("doc_name") or "").strip()
        match = _TEN_K_DOC_RE.search(doc_name)
        if not match:
            continue

        enriched = dict(row)
        enriched["doc_type"] = "10k"
        enriched["doc_period"] = int(match.group(1))
        gold_doc = parse_gold_doc(enriched)
        if not gold_doc.is_resolvable:
            raise ContractError(
                f"held-out row {row['financebench_id']} has unresolved provenance: "
                f"company={company!r}, doc_name={doc_name!r}"
            )
        selected.append(enriched)

    return sorted(selected, key=lambda row: str(row["financebench_id"]))


def validate_frozen_counts(rows: Sequence[dict[str, Any]]) -> None:
    documents = {str(row["doc_name"]) for row in rows}
    companies = {str(row["company"]) for row in rows}
    actual = (len(rows), len(documents), len(companies))
    expected = (
        EXPECTED_HELDOUT_ROWS,
        EXPECTED_HELDOUT_DOCUMENTS,
        EXPECTED_HELDOUT_COMPANIES,
    )
    if actual != expected:
        raise ContractError(
            "held-out counts do not match frozen contract: "
            f"rows/documents/companies={actual}, expected={expected}"
        )


def build_ingestion_manifest(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one exact corpus requirement per held-out source document."""
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        doc_name = str(row["doc_name"])
        gold_doc = parse_gold_doc(row)
        if not gold_doc.is_resolvable:
            raise ContractError(f"cannot build manifest for unresolved document {doc_name}")
        requirement = {
            "doc_name": doc_name,
            "company": str(row["company"]),
            "ticker": gold_doc.ticker,
            "filing_type": gold_doc.filing_type,
            "fiscal_year": gold_doc.fiscal_year,
        }
        previous = documents.setdefault(doc_name, requirement)
        if previous != requirement:
            raise ContractError(f"conflicting metadata for held-out document {doc_name}")
    return sorted(documents.values(), key=lambda item: str(item["doc_name"]))


def render_json(rows: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n"


def validate_content_hashes(output_content: str, manifest_content: str) -> None:
    if sha256_text(output_content) != EXPECTED_OUTPUT_SHA256:
        raise ContractError("held-out content SHA-256 does not match frozen contract")
    if sha256_text(manifest_content) != EXPECTED_MANIFEST_SHA256:
        raise ContractError("ingestion manifest SHA-256 does not match frozen contract")


def assert_writeable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise ContractError(f"refusing to overwrite different existing output: {path}")


def write_once(path: Path, content: str) -> None:
    assert_writeable(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ContractError("official FinanceBench source SHA-256 does not match source lock")
    if sha256_file(args.development) != DEVELOPMENT_SHA256:
        raise ContractError("development dataset SHA-256 does not match source lock")

    source_rows = load_jsonl(args.source)
    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise ContractError(
            f"official source contains {len(source_rows)} rows, expected {EXPECTED_SOURCE_ROWS}"
        )
    rows = select_holdout(source_rows, load_json_list(args.development))
    validate_frozen_counts(rows)
    manifest = build_ingestion_manifest(rows)
    if len(manifest) != EXPECTED_HELDOUT_DOCUMENTS:
        raise ContractError(
            f"ingestion manifest contains {len(manifest)} documents, "
            f"expected {EXPECTED_HELDOUT_DOCUMENTS}"
        )

    output_content = render_json(rows)
    manifest_content = render_json(manifest)
    validate_content_hashes(output_content, manifest_content)

    assert_writeable(args.output, output_content)
    if args.manifest_output:
        assert_writeable(args.manifest_output, manifest_content)
    write_once(args.output, output_content)
    if args.manifest_output:
        write_once(args.manifest_output, manifest_content)
    print(
        f"Materialized {len(rows)} held-out questions across "
        f"{len(manifest)} documents and "
        f"{len({row['company'] for row in rows})} companies: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
