"""
SEC HTML/XBRL cleaner
========================
Parses SEC EDGAR filing documents (10-K, 10-Q, 8-K) and returns a list of
(section_name, clean_text) tuples.  The cleaner:

  1. Removes boilerplate: XBRL inline tags, style/script elements, cover
     pages, signature blocks, and exhibit indexes.
  2. Detects and labels sections by their Item numbers.
  3. Maps Item numbers to human-readable names.

中文：清洗器优先保留可以解释的业务正文和表格结构，去除封面、签名、展品等检索价值较低的内容；
它是启发式解析器，无法保证恢复原始 HTML 的视觉布局。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from corpcheck.ingestion.processors.segment_types import CleanerSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section name mappings
# ---------------------------------------------------------------------------

SECTION_MAP_10K: dict[str, str] = {
    "item 1": "Business Description",
    "item 1a": "Risk Factors",
    "item 1b": "Unresolved Staff Comments",
    "item 2": "Properties",
    "item 3": "Legal Proceedings",
    "item 4": "Mine Safety Disclosures",
    "item 5": "Market for Common Equity",
    "item 6": "Selected Financial Data",
    "item 7": "MD&A",
    "item 7a": "Quantitative and Qualitative Disclosures about Market Risk",
    "item 8": "Financial Statements",
    "item 9": "Changes in and Disagreements with Accountants",
    "item 9a": "Controls and Procedures",
    "item 9b": "Other Information",
    "item 10": "Directors and Corporate Governance",
    "item 11": "Executive Compensation",
    "item 12": "Security Ownership",
    "item 13": "Certain Relationships",
    "item 14": "Principal Accountant Fees",
    "item 15": "Exhibits",
}

# A 10-Q numbers its items from 1 in both Part I and Part II, so the item
# number alone is ambiguous.  Keep one table per Part and resolve using the
# ``PART <roman>`` heading in effect at the item's position in the document.
SECTION_MAP_10Q_PART_I: dict[str, str] = {
    "item 1": "Financial Statements",
    "item 2": "MD&A",
    "item 3": "Quantitative and Qualitative Disclosures about Market Risk",
    "item 4": "Controls and Procedures",
}

SECTION_MAP_10Q_PART_II: dict[str, str] = {
    "item 1": "Legal Proceedings",
    "item 1a": "Risk Factors",
    "item 2": "Unregistered Sales of Equity Securities",
    "item 3": "Defaults upon Senior Securities",
    "item 4": "Mine Safety Disclosures",
    "item 5": "Other Information",
    "item 6": "Exhibits",
}

# Used when no Part heading precedes the item: Part I wins for the overlapping
# numbers 1-4, and Part II supplies the items that only exist there (1a, 5, 6).
SECTION_MAP_10Q: dict[str, str] = {
    **SECTION_MAP_10Q_PART_II,
    **SECTION_MAP_10Q_PART_I,
}

SECTION_MAP_8K: dict[str, str] = {
    "item 1.01": "Entry into a Material Definitive Agreement",
    "item 1.02": "Termination of a Material Definitive Agreement",
    "item 2.01": "Completion of Acquisition or Disposition",
    "item 2.02": "Results of Operations and Financial Condition",
    "item 2.03": "Creation of a Direct Financial Obligation",
    "item 4.01": "Changes in Registrant's Certifying Accountant",
    "item 5.02": "Departure of Directors or Officers",
    "item 7.01": "Regulation FD Disclosure",
    "item 8.01": "Other Events",
    "item 9.01": "Financial Statements and Exhibits",
}

# Patterns matching SEC cover page boilerplate
COVER_PAGE_PATTERNS = [
    re.compile(r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION", re.I | re.S),
    re.compile(r"Washington,\s*D\.C\.\s*20549", re.I),
    re.compile(r"FORM\s+10-[KQ]\s*\n", re.I),
]

# Patterns for signature blocks
SIGNATURE_PATTERNS = [
    re.compile(r"SIGNATURES?\s*\n", re.I),
    re.compile(r"Pursuant to the requirements of.*Securities Exchange Act", re.I | re.S),
]

# Exhibit index patterns
EXHIBIT_PATTERNS = [
    re.compile(r"EXHIBIT\s+INDEX\s*\n", re.I),
    re.compile(r"List of Exhibits\s*\n", re.I),
]

# XBRL inline tags to unwrap (keep their text content)
XBRL_TAGS_UNWRAP = {
    "ix:nonnumeric",
    "ix:nonfraction",
    "ix:continuation",
    "xbrli:period",
    "xbrli:instant",
}

# XBRL tags to remove entirely (no useful text)
XBRL_TAGS_REMOVE = {
    "ix:header",
    "ix:hidden",
    "ix:resources",
    "xbrli:xbrl",
    "xbrl:context",
    "link:linkbase",
}

# General boilerplate phrases to strip
BOILERPLATE_PHRASES = [
    re.compile(r"Table of Contents", re.I),
    re.compile(r"^\s*Page\s*$", re.I | re.M),
    re.compile(r"^\s*F-\d+\s*$", re.M),  # Financial statement page numbers
]

_TABLE_BLOCK_RE = re.compile(r"\[TABLE\].*?\[/TABLE\]", re.S)
_FINANCIAL_TABLE_TITLE_RE = re.compile(
    r"\b(?:(?:unaudited|condensed|combined)\s+)*(?:consolidated\s+)?(?:"
    r"statements?\s+of\s+(?:operations|income(?:\s+and\s+comprehensive\s+income)?|"
    r"earnings|comprehensive\s+income|"
    r"cash\s+flows?|financial\s+position|changes\s+in\s+(?:shareholders['’]?|"
    r"stockholders['’]?)\s+equity|(?:shareholders['’]?|stockholders['’]?)\s+equity)|"
    r"balance\s+sheets?|selected\s+financial\s+data"
    r")\b",
    re.I,
)
_REPORTING_PERIOD_CELL_RE = re.compile(
    r"^(?:for\s+the\s+)?(?:"
    r"(?:(?:one|two|three|six|nine|twelve|\d+)\s+)?"
    r"(?:years?|months?|quarters?)\s+ended(?:\s+|,\s*)?"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)?\s*\d{0,2}|"
    r"(?:as\s+of\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}"
    r")[,\s]*$",
    re.I,
)
_YEAR_CELL_RE = re.compile(r"^(?:FY\s*)?(?:19|20)\d{2}$", re.I)
_TABLE_UNIT_CELL_RE = re.compile(
    r"^\(?\s*in\s+(?:thousands|millions|billions)(?:,?\s+except\b.*)?\)?$",
    re.I,
)
_ITEM_ROW_RE = re.compile(
    r"^\[ROW\]\s*(ITEM\s+\d+[A-Z]?(?:\.\d+)?)\.?\s*\|\s*([^\|\n]+?)\s*$",
    re.I | re.M,
)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Matches headings like "ITEM 1A.", "Item 7 —", "ITEM 1.", etc.
_ITEM_HEADER_RE = re.compile(
    r"^\s*(ITEM\s+(\d+[AB]?(?:\.\d+)?))[\s\.\-–—:]+(.{0,80})?$",
    re.I | re.M,
)

# Matches "PART II", "PART I — FINANCIAL INFORMATION", "Part I." etc.  The
# optional title must start with a capital so prose such as
# "part ii of this report" is not mistaken for a heading.
_PART_HEADER_RE = re.compile(
    r"^\s*PART\s+(IV|III|II|I)\b(?:[\s\.\-–—:]+([A-Z][^\n]{0,60}?))?\s*$",
    re.I | re.M,
)

_HEADING_ALIAS_MAP_10K: dict[str, str] = {
    "business": "Business Description",
    "management’s discussion and analysis of financial condition and results of operations": "MD&A",
    "management's discussion and analysis of financial condition and results of operations": "MD&A",
    "financial statements and supplementary data": "Financial Statements",
    "consolidated statements of operations": "Financial Statements",
    "consolidated balance sheets": "Financial Statements",
    "notes to consolidated financial statements": "Financial Statements",
    "quantitative and qualitative disclosures about market risk": (
        "Quantitative and Qualitative Disclosures about Market Risk"
    ),
    "changes in and disagreements with accountants on accounting and financial disclosure": (
        "Changes in and Disagreements with Accountants"
    ),
}

_HEADING_ALIAS_MAP_10Q: dict[str, str] = {
    "management’s discussion and analysis of financial condition and results of operations": "MD&A",
    "management's discussion and analysis of financial condition and results of operations": "MD&A",
    "financial statements": "Financial Statements",
    "quantitative and qualitative disclosures about market risk": (
        "Quantitative and Qualitative Disclosures about Market Risk"
    ),
}

_GENERIC_HEADING_BLACKLIST = {
    "form 10-k cross-reference index",
    "annual report on form 10-k",
    "citigroup’s 2018 annual report on form 10-k",
    "for the year ended december 31, 2018",
    "table continues on the next page, including footnotes.",
    "nm not meaningful",
    "n/a not applicable",
}


def _normalize_item(raw: str) -> str:
    """Normalize an item label to lowercase for map lookup, e.g. 'ITEM 1A' -> 'item 1a'."""
    raw = raw.replace("\xa0", " ")
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    return raw


def _find_part_markers(text: str) -> list[tuple[int, str]]:
    """Return ``(offset, part)`` for every ``PART <roman>`` heading in *text*."""
    return [(m.start(), m.group(1).upper()) for m in _PART_HEADER_RE.finditer(text)]


def _part_for_offset(part_markers: list[tuple[int, str]], offset: int) -> str | None:
    """Return the Part heading in effect at *offset*, or None if there is none."""
    part: str | None = None
    for start, label in part_markers:
        if start > offset:
            break
        part = label
    return part


def _map_section(item_label: str, filing_type: str, part: str | None = None) -> str:
    """
    Map an item label to a human-readable section name.

    *part* is the roman numeral of the ``PART`` heading the item appears under
    (10-Q only); it disambiguates the item numbers that Part I and Part II
    share.  Returns the raw item label if no mapping is found.

    中文：10-Q 的 Part I/II 复用 Item 编号，必须结合前面的 Part 标记才能正确命名。
    """
    key = _normalize_item(item_label)
    if filing_type == "10-K":
        return SECTION_MAP_10K.get(key, item_label.title())
    elif filing_type == "10-Q":
        if part == "I":
            part_map = SECTION_MAP_10Q_PART_I
        elif part == "II":
            part_map = SECTION_MAP_10Q_PART_II
        else:
            part_map = SECTION_MAP_10Q
        # An item missing from its own Part (e.g. Part I has no Item 5) still
        # falls back to the merged table rather than degrading to "Item 5".
        return part_map.get(key) or SECTION_MAP_10Q.get(key, item_label.title())
    elif filing_type == "8-K":
        return SECTION_MAP_8K.get(key, item_label.title())
    return item_label.title()


def _normalize_heading_text(text: str) -> str:
    """Normalize a candidate heading for alias lookup and comparisons."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

class HTMLCleaner:
    """
    Parse and clean an SEC EDGAR HTML/text filing.

    Parameters
    ----------
    filing_type:
        One of '10-K', '10-Q', or '8-K'.  Affects section name mapping.
    min_section_length:
        Minimum character count for a section to be included in output.

    中文：将 EDGAR 文档变为可检索段落与表格。规则优先避免错误拼接，而非追求完整的视觉还原。
    """

    def __init__(
        self,
        filing_type: str = "10-K",
        min_section_length: int = 200,
    ) -> None:
        self.filing_type = filing_type.upper()
        self.min_section_length = min_section_length

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _load_soup(self, content: str | bytes) -> BeautifulSoup:
        """Parse HTML using lxml, with BeautifulSoup as interface."""
        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")
        return soup

    def _extract_submission_documents(
        self,
        content: str | bytes,
    ) -> list[tuple[str | bytes, bool]]:
        """Return the primary filing and a qualifying incorporated annual report.

        The boolean marks an annual-report attachment. Attachments are considered
        only for a 10-K that explicitly incorporates an annual report by reference;
        unrelated exhibits remain excluded.

        中文：只有明确引用且包含财务报表信号的附件才并入，避免把无关 Exhibit 混进主体 filing。
        """
        text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
        if "<DOCUMENT>" not in text.upper():
            return [(content, False)]

        target_type = self.filing_type.upper()
        parsed: list[tuple[str, str]] = []
        for block in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, re.I | re.S):
            type_match = re.search(r"<TYPE>\s*([^\n\r<]+)", block, re.I)
            text_match = re.search(r"<TEXT>(.*)", block, re.I | re.S)
            if type_match and text_match:
                parsed.append(
                    (type_match.group(1).strip().upper(), text_match.group(1).strip())
                )

        primary = next((body for doc_type, body in parsed if doc_type == target_type), None)
        if primary is None:
            return [(content, False)]

        documents: list[tuple[str | bytes, bool]] = [(primary, False)]
        if target_type != "10-K":
            return documents

        primary_words = " ".join(self._load_soup(primary).stripped_strings).lower()
        incorporates_annual_report = (
            "annual report" in primary_words
            and re.search(r"incorporat(?:ed|ion).{0,120}by reference", primary_words)
            is not None
        )
        if not incorporates_annual_report:
            return documents

        candidates: list[str] = []
        for doc_type, body in parsed:
            if doc_type != "ARS" and not doc_type.startswith("EX-13"):
                continue
            normalized = " ".join(self._load_soup(body).stripped_strings).lower()
            financial_signals = sum(
                signal in normalized
                for signal in (
                    "consolidated statements of operations",
                    "consolidated balance sheets",
                    "financial statements",
                )
            )
            if financial_signals >= 2:
                candidates.append(body)

        if candidates:
            documents.append((max(candidates, key=len), True))
        return documents

    def _extract_primary_document(self, content: str | bytes) -> str | bytes:
        """
        When SEC filings are stored as ``full-submission.txt``, extract the main
        filing document (the 10-K / 10-Q body) instead of parsing the entire
        multi-document submission with exhibits attached.
        """
        return self._extract_submission_documents(content)[0][0]

    def _strip_xbrl(self, soup: BeautifulSoup) -> None:
        """
        Remove XBRL-specific tags.  Tags in XBRL_TAGS_REMOVE are deleted
        entirely; tags in XBRL_TAGS_UNWRAP have their tag removed but their
        text content preserved.
        """
        # Lower-case tag names for matching
        for tag in soup.find_all(True):
            name = tag.name.lower() if tag.name else ""
            if name in XBRL_TAGS_REMOVE:
                tag.decompose()
            elif name in XBRL_TAGS_UNWRAP:
                tag.unwrap()

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        """Remove script, style, and other non-content tags."""
        for tag in soup.find_all(
            ["script", "style", "meta", "link", "noscript", "iframe", "img"]
        ):
            tag.decompose()

    def _normalize_cell_text(self, text: str) -> str:
        """Collapse whitespace inside a table cell."""
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _semantic_table_title_text(self, text: str) -> str | None:
        """Return a title-shaped financial-statement label, if present."""
        combined = self._normalize_cell_text(text)
        if not combined or len(combined) > 160 or len(combined.split()) > 18:
            return None

        match = _FINANCIAL_TABLE_TITLE_RE.search(combined)
        if not match:
            return None

        prefix = combined[: match.start()].strip(" -–—:")
        suffix = combined[match.end() :].strip()
        if prefix and not (
            prefix.isupper()
            or re.search(
                r"\b(?:inc\.?|corp\.?|corporation|company|ltd\.?|plc|llc)\s*$",
                prefix,
                re.I,
            )
        ):
            return None
        if suffix and not re.fullmatch(r"(?:\([^)]{1,60}\))?[\s\-–—:]*", suffix):
            return None
        return combined

    def _semantic_table_title(self, tag: Tag) -> str | None:
        return self._semantic_table_title_text(tag.get_text(" ", strip=True))

    def _table_is_data_like(self, table: Tag) -> bool:
        rows = table.find_all("tr")
        if len(rows) < 2:
            return False
        return any(
            re.search(r"\d", cell.get_text(" ", strip=True))
            for row in rows
            for cell in row.find_all(["th", "td"])
        )

    def _nearby_semantic_table_title(self, table: Tag) -> str | None:
        """Inspect only the nearest non-empty preceding block at each DOM level."""
        if not self._table_is_data_like(table):
            return None

        node = table
        for _depth in range(3):
            sibling = node.previous_sibling
            while sibling is not None:
                if isinstance(sibling, NavigableString):
                    sibling_text = self._normalize_cell_text(str(sibling))
                    if sibling_text:
                        return self._semantic_table_title_text(sibling_text)
                if isinstance(sibling, Tag):
                    sibling_text = self._normalize_cell_text(
                        sibling.get_text(" ", strip=True)
                    )
                    if not sibling_text:
                        sibling = sibling.previous_sibling
                        continue
                    if sibling.name == "table":
                        nonempty_cells = [
                            cell
                            for cell in sibling.find_all(["th", "td"])
                            if self._normalize_cell_text(
                                cell.get_text(" ", strip=True)
                            )
                        ]
                        if self._table_is_data_like(sibling) or len(nonempty_cells) != 1:
                            return None
                    return self._semantic_table_title_text(sibling_text)
                sibling = sibling.previous_sibling

            parent = node.parent
            if not isinstance(parent, Tag) or parent.name in {"body", "html"}:
                break
            node = parent
        return None

    def _looks_like_td_header(self, values: list[str]) -> bool:
        """Recognise reporting-period rows in SEC tables that omit ``<th>``."""
        first_cell = self._normalize_cell_text(values[0])
        if len(values) == 1 and _TABLE_UNIT_CELL_RE.fullmatch(first_cell):
            return True
        if _REPORTING_PERIOD_CELL_RE.fullmatch(first_cell):
            return len(values) == 1 or all(
                _YEAR_CELL_RE.fullmatch(self._normalize_cell_text(value))
                for value in values[1:]
            )

        year_cells = 0
        for value in values:
            cleaned = self._normalize_cell_text(value)
            if _YEAR_CELL_RE.fullmatch(cleaned):
                year_cells += 1
            elif not _TABLE_UNIT_CELL_RE.fullmatch(cleaned):
                return False
        return year_cells >= 2

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

        for row in table.find_all("tr", limit=2):
            for cell in row.find_all(["th", "td"]):
                inline_title = self._semantic_table_title(cell)
                if inline_title:
                    return inline_title

        nearby_title = self._nearby_semantic_table_title(table)
        if nearby_title:
            return nearby_title

        for sibling in table.previous_siblings:
            if not isinstance(sibling, Tag):
                continue
            if sibling.name == "table":
                break
            if sibling.name not in {"p", "div", "strong", "b", "h1", "h2", "h3", "h4"}:
                continue
            sibling_text = self._normalize_cell_text(sibling.get_text(" ", strip=True))
            if _FINANCIAL_TABLE_TITLE_RE.search(sibling_text) or _ITEM_HEADER_RE.match(
                sibling_text
            ):
                break
            if sibling_text and len(sibling_text) <= 160:
                return sibling_text

        return f"Table {table_index}"

    def _serialize_table(self, table: Tag, table_index: int) -> str | None:
        """Convert an HTML table into structured plain text."""
        title = self._infer_table_title(table, table_index)
        rows: list[tuple[list[str], bool]] = []
        header_line: str | None = None

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            values = [
                self._normalize_cell_text(cell.get_text(" ", strip=True))
                for cell in cells
            ]
            values = [value for value in values if value]
            if not values:
                continue

            rows.append((values, row.find("th") is not None))

        if not rows:
            return None

        if title != f"Table {table_index}":
            for index, (values, _has_th) in enumerate(rows[:2]):
                for value_index, value in enumerate(values):
                    if self._normalize_cell_text(value) != title:
                        continue
                    remaining_values = values[:value_index] + values[value_index + 1 :]
                    if remaining_values:
                        rows[index] = (remaining_values, _has_th)
                    else:
                        rows.pop(index)
                    break
                else:
                    continue
                break

        if not rows:
            return None

        explicit_header_index = next(
            (index for index, (_values, has_th) in enumerate(rows) if has_th),
            None,
        )
        if explicit_header_index is not None:
            header_values, _ = rows.pop(explicit_header_index)
            header_line = f"[HEADER] {' | '.join(header_values)}"
        else:
            inferred_header_rows: list[list[str]] = []
            while rows:
                values, _has_th = rows[0]
                if not self._looks_like_td_header(values):
                    break
                inferred_header_rows.append(rows.pop(0)[0])
            if inferred_header_rows:
                header_line = "[HEADER] " + " | ".join(
                    value for values in inferred_header_rows for value in values
                )

        row_texts = [f"[ROW] {' | '.join(values)}" for values, _has_th in rows]

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

        中文：序列化为 ``[TABLE]``、``[HEADER]`` 和 ``[ROW]`` 标记，供后续按行分块。
        """
        tables = [
            table
            for table in soup.find_all("table")
            if table.find_parent("table") is None
        ]
        serialized_tables = [
            (table, self._serialize_table(table, table_index))
            for table_index, table in enumerate(tables, start=1)
        ]
        for table, serialized in serialized_tables:
            if serialized:
                table.replace_with(NavigableString(f"\n{serialized}\n"))
            else:
                table.decompose()

    def _table_segment_from_block(
        self,
        section_name: str,
        block: str,
        table_index: int,
    ) -> CleanerSegment | None:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return None

        title = None
        header_text = None
        row_count = 0
        for line in lines:
            if line.startswith("[TABLE]"):
                title = line.replace("[TABLE]", "", 1).strip() or None
            elif line.startswith("[HEADER]"):
                header_text = line.replace("[HEADER]", "", 1).strip() or None
            elif line.startswith("[ROW]"):
                row_count += 1

        if row_count == 0 and header_text is None:
            return None

        return CleanerSegment(
            source_type="sec",
            section_name=section_name,
            content_kind="table",
            display_title=title,
            text=block.strip(),
            meta={
                "table_index": table_index,
                "header_text": header_text,
                "row_count": row_count,
            },
        )

    def _section_to_segments(
        self,
        section_name: str,
        section_text: str,
    ) -> list[CleanerSegment]:
        segments: list[CleanerSegment] = []
        cursor = 0
        table_index = 0

        for match in _TABLE_BLOCK_RE.finditer(section_text):
            narrative_text = section_text[cursor:match.start()].strip()
            if len(narrative_text) >= self.min_section_length:
                segments.append(
                    CleanerSegment(
                        source_type="sec",
                        section_name=section_name,
                        content_kind="narrative",
                        display_title=section_name,
                        text=narrative_text,
                        meta={},
                    )
                )

            table_index += 1
            table_segment = self._table_segment_from_block(
                section_name,
                match.group(0),
                table_index,
            )
            if table_segment is not None:
                segments.append(table_segment)
            cursor = match.end()

        trailing_text = section_text[cursor:].strip()
        if len(trailing_text) >= self.min_section_length:
            segments.append(
                CleanerSegment(
                    source_type="sec",
                    section_name=section_name,
                    content_kind="narrative",
                    display_title=section_name,
                    text=trailing_text,
                    meta={},
                )
            )

        if not segments and len(section_text.strip()) >= self.min_section_length:
            segments.append(
                CleanerSegment(
                    source_type="sec",
                    section_name=section_name,
                    content_kind="narrative",
                    display_title=section_name,
                    text=section_text.strip(),
                    meta={},
                )
            )

        return segments

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
                    if text.startswith("[TABLE]"):
                        texts.append(f"\n{text}\n")
                    else:
                        texts.append(text)
            elif isinstance(element, Tag) and element.name in {
                "p", "div", "tr", "br", "h1", "h2", "h3", "h4", "h5", "li"
            }:
                texts.append("\n")

        raw = " ".join(texts)
        # Collapse excessive whitespace while preserving paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()

    # ------------------------------------------------------------------
    # Boilerplate removal
    # ------------------------------------------------------------------

    def _remove_cover_page(self, text: str) -> str:
        """
        Remove the SEC cover page section (everything before the first
        actual business content heading).
        """
        for pattern in COVER_PAGE_PATTERNS:
            m = pattern.search(text)
            if m:
                # Find the next Item heading after the cover page
                item_m = _ITEM_HEADER_RE.search(text, m.end())
                if item_m and item_m.start() <= min(50_000, max(8_000, int(len(text) * 0.2))):
                    return text[item_m.start():]
        return text

    def _remove_signature_block(self, text: str) -> str:
        """Truncate text at the first signature block."""
        for pattern in SIGNATURE_PATTERNS:
            m = pattern.search(text)
            if m and m.start() > max(5_000, int(len(text) * 0.5)):
                return text[: m.start()].rstrip()
        return text

    def _remove_exhibit_index(self, text: str) -> str:
        """Truncate text at the exhibit index."""
        for pattern in EXHIBIT_PATTERNS:
            m = pattern.search(text)
            if m and m.start() > max(5_000, int(len(text) * 0.5)):
                return text[: m.start()].rstrip()
        return text

    def _strip_boilerplate_phrases(self, text: str) -> str:
        """Remove common boilerplate lines."""
        for pattern in BOILERPLATE_PHRASES:
            text = pattern.sub("", text)
        return text

    def _promote_item_heading_rows(self, text: str) -> str:
        """
        Convert single-row structured table headings like
        ``[ROW] Item 7. | Management's Discussion`` into plain heading lines
        so section splitting works for filings whose item headers are rendered
        as one-row tables instead of normal text headings.
        """
        def replace_table_block(match: re.Match[str]) -> str:
            block = match.group(0)
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            row_lines = [line for line in lines if line.startswith("[ROW]")]
            has_header = any(line.startswith("[HEADER]") for line in lines)
            if has_header or len(row_lines) != 1:
                return block

            row_match = _ITEM_ROW_RE.match(row_lines[0])
            if row_match is None:
                return block

            item_label = row_match.group(1).strip()
            heading = row_match.group(2).strip()
            return f"\n{item_label}. {heading}\n"

        def replace_heading(match: re.Match[str]) -> str:
            item_label = match.group(1).strip()
            heading = match.group(2).strip()
            return f"\n{item_label}. {heading}\n"

        text = _TABLE_BLOCK_RE.sub(replace_table_block, text)
        return _ITEM_ROW_RE.sub(replace_heading, text)

    def _map_heading_fallback(self, heading: str) -> str:
        """Map non-item fallback headings to canonical section names when possible."""
        normalized = _normalize_heading_text(heading)
        alias_map: dict[str, str]
        if self.filing_type == "10-K":
            alias_map = _HEADING_ALIAS_MAP_10K
        elif self.filing_type == "10-Q":
            alias_map = _HEADING_ALIAS_MAP_10Q
        else:
            alias_map = {}
        return alias_map.get(normalized, heading.strip())

    def _is_generic_heading_candidate(
        self,
        lines: list[str],
        line_index: int,
        char_offset: int,
        text_length: int,
    ) -> bool:
        """Heuristic filter for title-like fallback headings in older filings."""
        raw_line = lines[line_index]
        heading = " ".join(raw_line.split())
        if not heading:
            return False
        if text_length > 20_000 and char_offset < max(5_000, text_length // 100):
            return False
        if heading.startswith("["):
            return False
        normalized = _normalize_heading_text(heading)
        if normalized in _GENERIC_HEADING_BLACKLIST:
            return False
        if normalized.startswith("part "):
            return False
        if len(heading) < 3 or len(heading) > 100:
            return False
        if heading.endswith("."):
            return False
        if sum(ch.isalpha() for ch in heading) < 3:
            return False
        if len(heading.split()) > 14:
            return False
        if re.match(r"^\d+$", heading):
            return False
        if self._map_heading_fallback(heading) != heading.strip():
            return True

        previous_nonempty = ""
        for idx in range(line_index - 1, -1, -1):
            candidate = lines[idx].strip()
            if candidate:
                previous_nonempty = candidate
                break

        next_nonempty = ""
        for idx in range(line_index + 1, len(lines)):
            candidate = lines[idx].strip()
            if candidate:
                next_nonempty = candidate
                break

        has_blank_before = any(not lines[idx].strip() for idx in range(max(0, line_index - 4), line_index))
        has_blank_after = any(
            not lines[idx].strip()
            for idx in range(line_index + 1, min(len(lines), line_index + 5))
        )

        if not (
            line_index == 0
            or has_blank_before
            or previous_nonempty.startswith("[/TABLE]")
        ):
            return False
        if not (has_blank_after or next_nonempty.startswith("[TABLE]")):
            return False

        if any(ch.islower() for ch in heading):
            alpha_tokens = [token for token in re.split(r"\s+", heading) if token]
            capitalized_tokens = 0
            for token in alpha_tokens:
                token_alpha = re.sub(r"[^A-Za-z]", "", token)
                if not token_alpha:
                    continue
                if token_alpha.isupper() or token_alpha[0].isupper():
                    capitalized_tokens += 1
            if alpha_tokens and capitalized_tokens < max(1, int(len(alpha_tokens) * 0.6)):
                return False

        return True

    def _split_generic_headings(self, text: str) -> list[tuple[str, str]]:
        """
        Fallback section splitting for filings that do not contain standard
        ``ITEM`` headers but do have isolated title lines.

        中文：标准 Item 标题不可靠时才启用，宁愿退回完整文档也不把普通句子误认作标题。
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            return [("Full Document", text.strip())]

        heading_matches: list[tuple[int, str]] = []
        offset = 0
        last_normalized = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if self._is_generic_heading_candidate(lines, index, offset, len(text)):
                normalized = _normalize_heading_text(stripped)
                if normalized != last_normalized:
                    heading_matches.append((offset + line.find(stripped), stripped))
                    last_normalized = normalized
            elif stripped:
                last_normalized = None
            offset += len(line)

        if not heading_matches:
            return [("Full Document", text.strip())]

        sections: list[tuple[str, str]] = []
        for idx, (start, heading) in enumerate(heading_matches):
            end = heading_matches[idx + 1][0] if idx + 1 < len(heading_matches) else len(text)
            heading_end = start + len(heading)
            section_text = text[heading_end:end].strip()
            if len(section_text) < self.min_section_length:
                continue
            sections.append((self._map_heading_fallback(heading), section_text))

        return sections if sections else [("Full Document", text.strip())]

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split *text* on Item headings and return a list of
        (section_name, section_text) tuples.

        Consecutive matches to the same section label are merged.

        中文：优先按 SEC Item 切分；只有标题缺失或明显太晚出现时才切换到保守的通用标题规则。
        """
        matches = list(_ITEM_HEADER_RE.finditer(text))
        if matches:
            first_match = matches[0]
            late_match_cutoff = min(50_000, max(12_000, int(len(text) * 0.2)))
            if first_match.start() > late_match_cutoff:
                return self._split_generic_headings(text)
            if len(matches) == 1 and first_match.start() > max(8_000, int(len(text) * 0.05)):
                return self._split_generic_headings(text)
        if not matches:
            return self._split_generic_headings(text)

        part_markers = _find_part_markers(text) if self.filing_type == "10-Q" else []

        sections: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            item_label = m.group(1)
            section_name = _map_section(
                item_label,
                self.filing_type,
                _part_for_offset(part_markers, m.start()),
            )
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if len(section_text) >= self.min_section_length:
                sections.append((section_name, section_text))

        return sections if sections else [("Full Document", text.strip())]

    def _filter_sections(
        self,
        sections: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Drop low-value appendix sections that are usually harmful for retrieval.
        """
        filtered: list[tuple[str, str]] = []
        for section_name, section_text in sections:
            normalized = _normalize_item(section_name)
            if section_name == "Exhibits":
                continue
            if normalized in {"item 16", "item 6"} and len(section_text) > 100_000:
                continue
            filtered.append((section_name, section_text))
        return filtered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _clean_content_segments(self, content: str | bytes) -> list[CleanerSegment]:
        """Clean one already-selected filing document into retrieval segments.

        中文：阶段顺序固定为解析、去噪、保留表格、提取文本、移除样板、分 section，再生成段。
        """
        soup = self._load_soup(content)
        self._strip_xbrl(soup)
        self._strip_noise(soup)
        self._replace_tables_with_structured_text(soup)

        raw_text = self._extract_text(soup)
        text = self._remove_cover_page(raw_text)
        text = self._remove_signature_block(text)
        text = self._remove_exhibit_index(text)
        text = self._strip_boilerplate_phrases(text)
        text = self._promote_item_heading_rows(text)

        sections = self._filter_sections(self._split_sections(text))
        if not sections and len(text.strip()) >= self.min_section_length:
            sections = [("Full Document", text.strip())]

        segments: list[CleanerSegment] = []
        for section_name, section_text in sections:
            segments.extend(self._section_to_segments(section_name, section_text))
        return segments

    def clean_segments(self, file_path: str | Path) -> list[CleanerSegment]:
        """
        Load, clean, and section-split an SEC filing document.

        Parameters
        ----------
        file_path:
            Path to the downloaded HTML or text filing document.

        Returns
        -------
        list[CleanerSegment]
            Content-aware segments for downstream chunking.

        中文：文件不存在或读取失败时记录错误并返回空列表，允许批量处理继续。
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Filing not found: %s", path)
            return []

        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return []

        segments: list[CleanerSegment] = []
        documents = self._extract_submission_documents(content)
        for document, is_annual_report in documents:
            document_segments = self._clean_content_segments(document)
            if is_annual_report:
                document_segments = [
                    segment
                    for segment in document_segments
                    if segment.section_name == "Financial Statements"
                ]
                if not document_segments:
                    logger.warning(
                        "Ignored incorporated annual report without recognised "
                        "Financial Statements sections in %s",
                        path,
                    )
            segments.extend(document_segments)

        logger.debug(
            "Cleaned %s: %d documents, %d segments, %d total chars",
            path.name,
            len(documents),
            len(segments),
            sum(len(segment.text) for segment in segments),
        )
        return segments

    def clean(self, file_path: str | Path) -> list[tuple[str, str]]:
        """
        Backward-compatible wrapper returning section/text tuples.

        中文：旧调用方只需要文本时使用；新代码应优先使用 ``clean_segments`` 保留结构信息。
        """
        return [
            (segment.section_name, segment.text)
            for segment in self.clean_segments(file_path)
        ]

    def clean_text_segments(
        self,
        raw_html: str | bytes,
        filing_type: str | None = None,
    ) -> list[CleanerSegment]:
        """
        Clean HTML provided as a string/bytes rather than a file path.
        Useful for in-memory processing.

        中文：传入 ``filing_type`` 会更新当前清洗器实例的类型，复用实例时调用方需注意该状态。
        """
        if filing_type:
            self.filing_type = filing_type.upper()

        segments: list[CleanerSegment] = []
        for document, is_annual_report in self._extract_submission_documents(raw_html):
            document_segments = self._clean_content_segments(document)
            if is_annual_report:
                document_segments = [
                    segment
                    for segment in document_segments
                    if segment.section_name == "Financial Statements"
                ]
            segments.extend(document_segments)
        return segments

    def clean_text(self, raw_html: str | bytes, filing_type: str | None = None) -> list[tuple[str, str]]:
        """Return legacy section/text tuples for HTML already held in memory.

        中文：这是 ``clean_text_segments`` 的兼容包装，会丢弃结构化元数据。
        """
        return [
            (segment.section_name, segment.text)
            for segment in self.clean_text_segments(raw_html, filing_type=filing_type)
        ]


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def clean_filing(
    file_path: str | Path,
    filing_type: str = "10-K",
) -> list[tuple[str, str]]:
    """Clean a single filing and return (section_name, text) tuples.

    中文：模块级便捷入口；需要调整最小 section 长度或读取结构信息时请使用 ``HTMLCleaner``。
    """
    cleaner = HTMLCleaner(filing_type=filing_type)
    return cleaner.clean(file_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: html_cleaner.py <path_to_filing> [10-K|10-Q|8-K]")
        sys.exit(1)

    ftype = sys.argv[2] if len(sys.argv) > 2 else "10-K"
    sections = clean_filing(sys.argv[1], ftype)
    for name, text in sections:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print(text[:500], "...")
