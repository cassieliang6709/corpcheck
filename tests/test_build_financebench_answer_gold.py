"""Tests for the manually curated FinanceBench answer-gold builder."""

from __future__ import annotations

import json

from evaluation import build_financebench_answer_gold as builder


def test_builder_partitions_all_source_ids_into_34_plus_1():
    rows = builder.read_source(builder.DEFAULT_INPUT)
    gold, exclusions = builder.build_records(rows)

    source_ids = {row["financebench_id"] for row in rows}
    gold_ids = {row["id"] for row in gold}
    exclusion_ids = {row["id"] for row in exclusions}
    assert len(gold) == 34
    assert len(exclusions) == 1
    assert gold_ids.isdisjoint(exclusion_ids)
    assert gold_ids | exclusion_ids == source_ids


def test_every_gold_record_has_required_answer_eval_and_provenance_fields():
    gold, _ = builder.build_records(builder.read_source(builder.DEFAULT_INPUT))

    for record in gold:
        assert record["id"].startswith("financebench_id_")
        assert record["question"]
        assert record["should_abstain"] is False
        assert record["expected_values"]
        assert record["company"]
        assert record["filing_type"] in {"10-K", "10-Q"}
        assert isinstance(record["year"], int)
        assert record["source"] == "FinanceBench"
        assert record["category"] in {"extraction", "calculation", "qualitative", "mixed"}
        if "accepted_answers" in record:
            assert record["accepted_answers"]


def test_key_curated_values_and_conflicting_gold_exclusion_are_preserved():
    gold, exclusions = builder.build_records(builder.read_source(builder.DEFAULT_INPUT))
    by_id = {record["id"]: record for record in gold}

    assert by_id["financebench_id_04209"]["expected_values"] == [59268, "total assets"]
    assert by_id["financebench_id_00563"]["expected_values"] == ["Data Center"]
    assert by_id["financebench_id_01107"]["expected_values"] == [
        "usual and customary pricing", "PBM", "opioid", 4.3, 625,
    ]
    assert exclusions == [
        {
            "id": "financebench_id_00283",
            "question": (
                "How much does Pfizer expect to pay to spin off Upjohn in the future "
                "in USD million?"
            ),
            "original_gold": 77.78,
            "evidence_derived_value": 70,
            "reason": builder.EXCLUSIONS["financebench_id_00283"]["reason"],
            "source": "FinanceBench",
        }
    ]


def test_committed_outputs_match_the_deterministic_builder(tmp_path):
    output = tmp_path / "gold.json"
    exclusions_output = tmp_path / "exclusions.json"
    assert builder.main(
        [
            "--input", str(builder.DEFAULT_INPUT),
            "--output", str(output),
            "--exclusions-output", str(exclusions_output),
        ]
    ) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        builder.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert json.loads(exclusions_output.read_text(encoding="utf-8")) == json.loads(
        builder.DEFAULT_EXCLUSIONS_OUTPUT.read_text(encoding="utf-8")
    )
