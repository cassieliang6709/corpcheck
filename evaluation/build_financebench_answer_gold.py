"""Build the manually curated FinanceBench final-answer evaluation gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from evaluation.financebench import parse_gold_doc

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_INPUT = DATASETS_DIR / "financebench_filtered.json"
DEFAULT_OUTPUT = DATASETS_DIR / "financebench_answer_eval.json"
DEFAULT_EXCLUSIONS_OUTPUT = DATASETS_DIR / "financebench_answer_eval_exclusions.json"

# Human-curated final-answer requirements. These are deliberately short values
# or phrases, never a copy of FinanceBench's long prose answer.
CURATIONS: dict[str, dict[str, Any]] = {
    "financebench_id_08286": {"expected_values": [11588, "net income"], "category": "extraction"},
    "financebench_id_00222": {"expected_values": [1.57, "quick ratio"], "category": "mixed"},
    "financebench_id_00995": {
        "expected_values": [
            "server microprocessors", "DPUs", "FPGAs", "semi-custom SoC",
            "development services", "embedded CPUs",
        ],
        "category": "qualitative",
    },
    "financebench_id_01198": {
        "expected_values": [
            "EPYC server processors", "semi-custom product sales",
            "Xilinx embedded product sales",
        ],
        "category": "qualitative",
    },
    "financebench_id_00917": {
        "expected_values": ["amortization of intangible assets", "Xilinx acquisition"],
        "category": "qualitative",
    },
    "financebench_id_01279": {
        "expected_values": ["operating activities"],
        "accepted_answers": ["Operations", "Operating activities"],
        "category": "calculation",
    },
    "financebench_id_00563": {"expected_values": ["Data Center"], "category": "calculation"},
    "financebench_id_00757": {
        "expected_values": ["one customer", 16, "consolidated net revenue"],
        "category": "extraction",
    },
    "financebench_id_04209": {
        "expected_values": [59268, "total assets"],
        "category": "extraction",
    },
    "financebench_id_05915": {
        "expected_values": [17.98, "fixed asset turnover"],
        "category": "calculation",
    },
    "financebench_id_00790": {
        "expected_values": [1.82, "ROA", 5.6, "goodwill"],
        "category": "mixed",
    },
    "financebench_id_01107": {
        "expected_values": [
            "usual and customary pricing", "PBM", "opioid", 4.3, 625,
        ],
        "category": "qualitative",
    },
    "financebench_id_01244": {
        "expected_values": ["paid", 0.55, "per share"],
        "category": "extraction",
    },
    "financebench_id_00956": {
        "expected_values": ["not a high growth company", 1.3],
        "category": "mixed",
    },
    "financebench_id_00669": {
        "expected_values": [
            "COVID-19 vaccine manufacturing exit",
            "currency impacts in the Pharmaceutical segment",
            "commodity inflation in the MedTech and Consumer Health segments",
            "supply chain benefits in the Consumer Health segment",
            "partially offset",
        ],
        "category": "qualitative",
    },
    "financebench_id_00711": {
        "expected_values": [2.7, "times"],
        "category": "calculation",
    },
    "financebench_id_00299": {
        "expected_values": ["Corporate", -473, "million"],
        "category": "extraction",
    },
    "financebench_id_02119": {
        "expected_values": [66.56, "per share"],
        "category": "calculation",
    },
    "financebench_id_00206": {
        "expected_values": ["financial institution", "not a relevant metric"],
        "category": "qualitative",
    },
    "financebench_id_00394": {
        "expected_values": ["Corporate & Investment Bank", 3725, "million"],
        "category": "extraction",
    },
    "financebench_id_02049": {
        "expected_values": ["decreased", 7, "million"],
        "accepted_answers": ["Yes. It decreased."],
        "category": "calculation",
    },
    "financebench_id_00552": {
        "expected_values": ["decreased", 2.5, "billion"],
        "category": "calculation",
    },
    "financebench_id_04302": {"expected_values": [55.1], "category": "calculation"},
    "financebench_id_03531": {
        "expected_values": [16525, "million"],
        "category": "extraction",
    },
    "financebench_id_04080": {"expected_values": [3.46], "category": "calculation"},
    "financebench_id_01163": {
        "expected_values": ["cash flow from operations"],
        "accepted_answers": ["Operating activities.", "Cash flow from operations."],
        "category": "calculation",
    },
    "financebench_id_00302": {
        "expected_values": [14882, 13745],
        "accepted_answers": ["Yes. PP&E grew.", "Yes, PP&E grew."],
        "category": "calculation",
    },
    "financebench_id_00702": {
        "expected_values": ["gain", "Consumer Healthcare JV transaction"],
        "category": "qualitative",
    },
    "financebench_id_02416": {
        "expected_values": ["Trillium", "Array", "Therachon"],
        "category": "extraction",
    },
    "financebench_id_00724": {
        "expected_values": ["Developed Rest of World"],
        "category": "calculation",
    },
    "financebench_id_02419": {"expected_values": ["Upjohn"], "category": "qualitative"},
    "financebench_id_06247": {"expected_values": [42.69], "category": "calculation"},
    "financebench_id_04784": {"expected_values": [0.2], "category": "calculation"},
    "financebench_id_06741": {"expected_values": [6.2], "category": "calculation"},
}

EXCLUSIONS: dict[str, dict[str, Any]] = {
    "financebench_id_00283": {
        "original_gold": 77.78,
        "evidence_derived_value": 70,
        "reason": (
            "FinanceBench's 77.78 assumes $700 million is 90% of total cost, but "
            "the evidence says $700 million is the total expected cost and 90% has "
            "already been incurred; the evidence-derived remaining amount is about "
            "$70 million."
        ),
    }
}


class BuildError(ValueError):
    """The source slice and curated mapping are inconsistent."""


def read_source(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise BuildError(f"{path}: expected a JSON list of objects")
    return rows


def build_records(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = [row.get("financebench_id") for row in rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in source_ids):
        raise BuildError("every source record must have a non-empty financebench_id")
    if len(source_ids) != len(set(source_ids)):
        raise BuildError("source financebench_id values must be unique")

    curated_ids = set(CURATIONS)
    excluded_ids = set(EXCLUSIONS)
    if curated_ids & excluded_ids:
        raise BuildError("curated and excluded ids must be disjoint")
    if set(source_ids) != curated_ids | excluded_ids:
        missing = sorted(set(source_ids) - curated_ids - excluded_ids)
        extra = sorted((curated_ids | excluded_ids) - set(source_ids))
        raise BuildError(f"curation partition mismatch: missing={missing}, extra={extra}")

    answer_gold: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        record_id = row["financebench_id"]
        if record_id in EXCLUSIONS:
            exclusions.append(
                {
                    "id": record_id,
                    "question": row["question"],
                    **EXCLUSIONS[record_id],
                    "source": "FinanceBench",
                }
            )
            continue

        doc = parse_gold_doc(row)
        if not doc.is_resolvable:
            raise BuildError(f"{record_id}: source document metadata is not resolvable")
        curation = CURATIONS[record_id]
        record = {
            "id": record_id,
            "question": row["question"],
            "should_abstain": False,
            "expected_values": curation["expected_values"],
            "company": doc.ticker,
            "filing_type": doc.filing_type,
            "year": doc.fiscal_year,
            "source": "FinanceBench",
            "category": curation["category"],
        }
        if curation.get("accepted_answers"):
            record["accepted_answers"] = curation["accepted_answers"]
        answer_gold.append(record)

    return answer_gold, exclusions


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclusions-output", type=Path, default=DEFAULT_EXCLUSIONS_OUTPUT
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    answer_gold, exclusions = build_records(read_source(args.input))
    write_json(args.output, answer_gold)
    write_json(args.exclusions_output, exclusions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
