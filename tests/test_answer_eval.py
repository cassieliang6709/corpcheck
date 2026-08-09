"""Tests for the deterministic end-to-end answer evaluator."""

from __future__ import annotations

import json

import pytest
from evaluation.answer_eval import RecordError, evaluate, extract_citation_indices, main


def prediction(**overrides):
    record = {
        "id": "q1",
        "answer": "Revenue was $100 million [1].",
        "abstained": False,
        "chunks": [{"chunk_id": "gold-chunk"}],
    }
    record.update(overrides)
    return record


def gold(**overrides):
    record = {
        "id": "q1",
        "should_abstain": False,
        "expected_values": [100, "million"],
        "supporting_chunk_ids": ["gold-chunk"],
    }
    record.update(overrides)
    return record


def test_extracts_single_and_grouped_citations():
    assert extract_citation_indices("First [1], then [2, 3][4].") == [1, 2, 3, 4]


def test_scores_correct_answer_citations_support_and_end_to_end_pass():
    result = evaluate([prediction()], [gold()])

    record = result["records"][0]
    assert record["answer_correct"] is True
    assert record["citation_present"] is True
    assert record["citation_indices_valid"] is True
    assert record["support_proxy"] is True
    assert record["support_precision"] == 1.0
    assert record["end_to_end_pass"] is True
    assert result["summary"]["end_to_end_pass_rate"] == 1.0


def test_accepted_answer_uses_normalized_exact_match():
    result = evaluate(
        [prediction(answer="Not material. [1]")],
        [gold(expected_values=[], accepted_answers=["NOT MATERIAL!"])],
    )
    # Citations are not part of the semantic answer and therefore should not
    # prevent an otherwise exact accepted-answer match.
    assert result["records"][0]["answer_correct"] is True


def test_invalid_citation_index_fails_validity_and_support_proxy():
    result = evaluate([prediction(answer="Revenue was $100 million [2].")], [gold()])
    record = result["records"][0]
    assert record["invalid_citation_indices"] == [2]
    assert record["citation_indices_valid"] is False
    assert record["support_proxy"] is False
    assert record["end_to_end_pass"] is False


def test_non_gold_citation_fails_conservative_support_proxy():
    pred = prediction(
        answer="Revenue was $100 million [1][2].",
        chunks=["gold-chunk", "unrelated-chunk"],
    )
    record = evaluate([pred], [gold()])["records"][0]
    assert record["support_precision"] == 0.5
    assert record["support_proxy"] is False


def test_correct_abstention_passes_without_answer_or_citation():
    pred = prediction(answer="", abstained=True, chunks=[])
    target = gold(should_abstain=True, expected_values=[], supporting_chunk_ids=[])
    record = evaluate([pred], [target])["records"][0]
    assert record["correct_abstention"] is True
    assert record["answer_correct"] is None
    assert record["end_to_end_pass"] is True


def test_wrong_abstention_counts_as_wrong_answer():
    record = evaluate(
        [prediction(answer="", abstained=True, chunks=[])],
        [gold(supporting_chunk_ids=[])],
    )["records"][0]
    assert record["correct_abstention"] is False
    assert record["answer_correct"] is False
    assert record["end_to_end_pass"] is False


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([prediction()], [gold(id="q2")], "missing prediction ids"),
        ([prediction(), prediction()], [gold()], "duplicate id"),
        ([prediction(abstained="no")], [gold()], "'abstained' must be bool"),
        ([prediction()], [gold(expected_values=[])], "answerable gold must supply"),
    ],
)
def test_malformed_or_unaligned_records_fail_clearly(predictions, targets, message):
    with pytest.raises(RecordError, match=message):
        evaluate(predictions, targets)


def test_cli_reads_prediction_jsonl_and_gold_json_and_writes_result(tmp_path, capsys):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction()) + "\n", encoding="utf-8")
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps([gold()]), encoding="utf-8")
    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "--predictions",
            str(predictions_path),
            "--gold",
            str(gold_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    stdout_result = json.loads(capsys.readouterr().out)
    file_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_result == file_result
    assert file_result["summary"]["records"] == 1


def test_cli_returns_two_and_names_bad_json_line(tmp_path, capsys):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text('{"id": "q1"}\nnot-json\n', encoding="utf-8")
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps(gold()) + "\n", encoding="utf-8")

    exit_code = main(["--predictions", str(predictions_path), "--gold", str(gold_path)])

    assert exit_code == 2
    assert f"{predictions_path}:2: invalid JSON" in capsys.readouterr().err
