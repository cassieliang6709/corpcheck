"""
SEC HTML/XBRL cleaner
========================
Parses SEC EDGAR filing documents (10-K, 10-Q, 8-K) and returns a list of
(section_name, clean_text) tuples.  The cleaner:

  1. Removes boilerplate: XBRL inline tags, style/script elements, cover
     pages, signature blocks, and exhibit indexes.
  2. Detects and labels sections by their Item numbers.
  3. Maps Item numbers to human-readable names.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

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

    def _extract_primary_document(self, content: str | bytes) -> str | bytes:
        """
        When SEC filings are stored as ``full-submission.txt``, extract the main
        filing document (the 10-K / 10-Q body) instead of parsing the entire
        multi-document submission with exhibits attached.
        """
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = content

        if "<DOCUMENT>" not in text.upper():
            return content

        target_type = self.filing_type.upper()
        document_blocks = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, re.I | re.S)
        for block in document_blocks:
            type_match = re.search(r"<TYPE>\s*([^\n\r<]+)", block, re.I)
            if not type_match:
                continue
            doc_type = type_match.group(1).strip().upper()
            if doc_type != target_type:
                continue

            text_match = re.search(r"<TEXT>(.*)", block, re.I | re.S)
            if text_match:
                return text_match.group(1).strip()

        return content

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
            values = [
                self._normalize_cell_text(cell.get_text(" ", strip=True))
                for cell in cells
            ]
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

        if not (has_blank_before or previous_nonempty.startswith("[/TABLE]")):
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

        content = self._extract_primary_document(content)

        # Parse
        soup = self._load_soup(content)

        # Clean HTML
        self._strip_xbrl(soup)
        self._strip_noise(soup)
        self._replace_tables_with_structured_text(soup)

        # Extract text
        raw_text = self._extract_text(soup)

        # Remove boilerplate
        text = self._remove_cover_page(raw_text)
        text = self._remove_signature_block(text)
        text = self._remove_exhibit_index(text)
        text = self._strip_boilerplate_phrases(text)
        text = self._promote_item_heading_rows(text)

        # Split into sections
        sections = self._split_sections(text)
        sections = self._filter_sections(sections)
        if not sections and len(text.strip()) >= self.min_section_length:
            sections = [("Full Document", text.strip())]

        segments: list[CleanerSegment] = []
        for section_name, section_text in sections:
            segments.extend(self._section_to_segments(section_name, section_text))

        logger.debug(
            "Cleaned %s: %d sections, %d segments, %d total chars",
            path.name,
            len(sections),
            len(segments),
            sum(len(s[1]) for s in sections),
        )
        return segments

    def clean(self, file_path: str | Path) -> list[tuple[str, str]]:
        """
        Backward-compatible wrapper returning section/text tuples.
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
        """
        if filing_type:
            self.filing_type = filing_type.upper()

        raw_html = self._extract_primary_document(raw_html)
        soup = self._load_soup(raw_html)
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

    def clean_text(self, raw_html: str | bytes, filing_type: str | None = None) -> list[tuple[str, str]]:
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
    """Clean a single filing and return (section_name, text) tuples."""
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
