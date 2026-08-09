from __future__ import annotations

import json

from evaluation.table_child_experiment import (
    children_from_parent_records,
    main,
    parse_table_children,
)


def test_multi_year_headers_are_inherited_by_each_numeric_data_row() -> None:
    text = """[TABLE] Consolidated Statements of Operations
[HEADER] Year Ended December 31
[ROW] 2022 | 2021 | 2020
[ROW] Revenue | $ 12,000 | $ 10,000 | $ 8,000
[ROW] Net income | 900 | 700 | 500
[/TABLE]"""

    children = parse_table_children(text, "parent-1")

    assert [child.parent_chunk_id for child in children] == ["parent-1", "parent-1"]
    assert [child.data_row for child in children] == [
        "[ROW] Revenue | $ 12,000 | $ 10,000 | $ 8,000",
        "[ROW] Net income | 900 | 700 | 500",
    ]
    assert children[0].header_rows == (
        "[HEADER] Year Ended December 31",
        "[ROW] 2022 | 2021 | 2020",
    )
    assert children[0].text.count("[ROW] Revenue") == 1
    assert "[ROW] Net income" not in children[0].text


def test_numeric_rows_cover_currency_percentages_and_parentheses() -> None:
    text = """[TABLE] Margin Summary
[ROW] Fiscal year | 2023 | 2022
[ROW] Gross margin | 42.5% | 40.1%
[ROW] Loss | ($1,250) | (900)
[/TABLE]"""

    children = parse_table_children(text, 42)

    assert [child.data_row for child in children] == [
        "[ROW] Gross margin | 42.5% | 40.1%",
        "[ROW] Loss | ($1,250) | (900)",
    ]
    assert children[0].header_rows == ("[ROW] Fiscal year | 2023 | 2022",)
    assert all(child.parent_chunk_id == "42" for child in children)


def test_continuation_chunk_keeps_repeated_context() -> None:
    text = """[TABLE] Debt Maturities (continued)
[HEADER] Payments due by period
[ROW] December 31,
[ROW] 2025 | 2026 | 2027
[ROW] Senior notes | 100 | 200 | 300
[/TABLE]"""

    children = parse_table_children(text, "continuation-2")

    assert len(children) == 1
    assert children[0].table_title == "Debt Maturities (continued)"
    assert children[0].header_rows == (
        "[HEADER] Payments due by period",
        "[ROW] December 31,",
        "[ROW] 2025 | 2026 | 2027",
    )


def test_duplicate_rows_and_duplicate_parent_records_are_deduplicated() -> None:
    text = """[TABLE] Revenue
[ROW] 2023 | 2022
[ROW] Product sales | 100 | 90
[ROW] Product sales | 100 | 90
[/TABLE]"""

    children = children_from_parent_records(
        [{"chunk_id": "p1", "text": text}, {"chunk_id": "p1", "text": text}]
    )

    assert len(children) == 1
    assert children[0].child_index == 0


def test_malformed_and_no_data_tables_are_skipped_deterministically() -> None:
    malformed = """[TABLE] Missing close
[ROW] 2023 | 2022
[ROW] Revenue | 100 | 90"""
    no_data = """[TABLE] Labels only
[HEADER] Description
[ROW] Current assets
[/TABLE]"""

    assert parse_table_children(malformed, "bad") == []
    assert parse_table_children(no_data, "empty") == []
    assert children_from_parent_records([{"chunk_id": "missing-text"}, {"text": no_data}]) == []


def test_cli_emits_jsonl_children_from_text_and_content_fields(tmp_path, capsys) -> None:
    parent_path = tmp_path / "parents.jsonl"
    output_path = tmp_path / "children.jsonl"
    table = "[TABLE] Assets\n[ROW] 2023 | 2022\n[ROW] Cash | 10 | 8\n[/TABLE]"
    parent_path.write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": "a", "text": table}),
                json.dumps({"chunk_id": "b", "content": table}),
            ]
        ),
        encoding="utf-8",
    )

    assert main([str(parent_path), "--output", str(output_path)]) == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["parent_chunk_id"] for record in records] == ["a", "b"]
    assert all(record["data_row"] == "[ROW] Cash | 10 | 8" for record in records)
    assert capsys.readouterr().out == ""
