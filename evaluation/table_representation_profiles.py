"""Locked baseline and candidate cleaners for table-representation evaluation.

中文：将基线与候选清洗器固定为可追溯的表示配置，以便归因评测差异；不要把这里
的评测配置当作生产默认行为。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner

REPRESENTATION_PROFILES = ("baseline", "candidate")
BASELINE_SOURCE_COMMIT = "3774e69ce74d7c010aecbb07b5ce5faebf2f1d95"


class _BaselineHTMLCleaner(HTMLCleaner):
    """The table-context behavior immediately before ``97c9fed``."""

    def _infer_table_title(self, table: Tag, table_index: int) -> str:
        """Best-effort table title extraction."""
        caption = table.find("caption")
        if caption:
            caption_text = self._normalize_cell_text(caption.get_text(" ", strip=True))
            if caption_text:
                return caption_text

        for attr_name in ("summary", "aria-label", "title"):
            attr_value = table.get(attr_name)
            if attr_value:
                cleaned = self._normalize_cell_text(str(attr_value))
                if cleaned:
                    return cleaned

        for sibling in table.previous_siblings:
            if not isinstance(sibling, Tag):
                continue
            if sibling.name not in {"p", "div", "strong", "b", "h1", "h2", "h3", "h4"}:
                continue
            sibling_text = self._normalize_cell_text(sibling.get_text(" ", strip=True))
            if sibling_text and len(sibling_text) <= 160:
                return sibling_text

        return f"Table {table_index}"

    def _serialize_table(self, table: Tag, table_index: int) -> str | None:
        """Convert an HTML table into structured plain text."""
        title = self._infer_table_title(table, table_index)
        row_texts: list[str] = []
        header_line: str | None = None

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            values = [self._normalize_cell_text(cell.get_text(" ", strip=True)) for cell in cells]
            values = [value for value in values if value]
            if not values:
                continue

            line = " | ".join(values)
            if header_line is None and row.find("th"):
                header_line = f"[HEADER] {line}"
            else:
                row_texts.append(f"[ROW] {line}")

        if not row_texts and header_line is None:
            return None

        lines = [f"[TABLE] {title}"]
        if header_line:
            lines.append(header_line)
        lines.extend(row_texts)
        lines.append("[/TABLE]")
        return "\n".join(lines)

    def _replace_tables_with_structured_text(self, soup: BeautifulSoup) -> None:
        """
        Replace HTML tables with a structured plain-text representation so
        row/column relationships survive downstream chunking and retrieval.
        """
        table_index = 0
        for table in soup.find_all("table"):
            if table.find_parent("table") is not None:
                continue
            table_index += 1
            serialized = self._serialize_table(table, table_index)
            if serialized:
                table.replace_with(NavigableString(f"\n{serialized}\n"))
            else:
                table.decompose()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract visible text from the cleaned soup.
        Paragraphs are separated by double newlines.
        """
        texts: list[str] = []
        for element in soup.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    texts.append(text)
            elif isinstance(element, Tag) and element.name in {
                "p",
                "div",
                "tr",
                "br",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "li",
            }:
                texts.append("\n")

        raw = " ".join(texts)
        # Collapse excessive whitespace while preserving paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def cleaner_for_profile(profile: str, filing_type: str) -> HTMLCleaner:
    """Create the cleaner locked to an evaluation representation profile."""
    if profile == "baseline":
        return _BaselineHTMLCleaner(filing_type=filing_type)
    if profile == "candidate":
        return HTMLCleaner(filing_type=filing_type)
    raise ValueError(
        f"unknown representation profile {profile!r}; expected one of {REPRESENTATION_PROFILES}"
    )


def profile_source_fingerprint(profile: str) -> str:
    """Fingerprint the executable cleaner source and the profile's provenance."""
    cleaner = cleaner_for_profile(profile, "10-K")
    current_source_path = Path(inspect.getfile(HTMLCleaner))
    payload = {
        "profile": profile,
        "current_html_cleaner_sha256": hashlib.sha256(current_source_path.read_bytes()).hexdigest(),
        "profile_class_source": inspect.getsource(type(cleaner)),
    }
    if profile == "baseline":
        payload["baseline_source_commit"] = BASELINE_SOURCE_COMMIT
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
