"""
Text chunker
=============
Splits section text into overlapping token-window chunks while trying not
to break sentences mid-way.  Uses the HuggingFace AutoTokenizer for the
all-MiniLM-L6-v2 model so token counts are consistent with the embedder.

Chunk strategy
--------------
1. Tokenise the full section text.
2. Walk through tokens in steps of (CHUNK_SIZE - OVERLAP), collecting
   CHUNK_SIZE-token windows.
3. For each window, decode to text and then *trim to the nearest sentence
   boundary* so chunks don't end mid-sentence.
4. Return (section_name, chunk_text, token_count) tuples.

中文：分块以 embedding 模型的 token 为准。叙述按句子打包，表格按行打包，二者的重叠规则不同，
以尽量保留可读性和表格关系。
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from corpcheck.ingestion.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_MODEL,
)
from corpcheck.ingestion.processors.segment_types import ChunkPayload, CleanerSegment

logger = logging.getLogger(__name__)

# Sentence-boundary regex: split on . ! ? followed by whitespace + capital
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")
_TABLE_BLOCK_RE = re.compile(r"\[TABLE\].*?\[/TABLE\]", re.DOTALL)
# Minimum chunk length (characters) to bother embedding
MIN_CHUNK_CHARS = 50
TABLE_ROW_OVERLAP = 3

# Type alias
ChunkTuple = tuple[str, str, int]  # (section_name, text, token_count)

_CHUNK_CONFIGS = {
    ("sec", "narrative"): {"chunk_size": 420, "overlap": 60, "fallback_chunk_size": 480, "fallback_overlap": 60},
    ("sec", "table"): {"chunk_size": 320, "overlap": 60, "row_overlap": 3},
    ("news", "narrative"): {"chunk_size": 380, "overlap": 40, "fallback_chunk_size": 420, "fallback_overlap": 40},
    ("news", "table"): {"chunk_size": 320, "overlap": 40, "row_overlap": 3},
    ("transcript", "narrative"): {"chunk_size": 420, "overlap": 60, "fallback_chunk_size": 480, "fallback_overlap": 60},
    ("transcript", "qa"): {"chunk_size": 520, "overlap": 80, "fallback_chunk_size": 560, "fallback_overlap": 80},
}


# ---------------------------------------------------------------------------
# Tokeniser singleton (load once per process)
# ---------------------------------------------------------------------------

_tokenizer: PreTrainedTokenizerBase | None = None


def get_tokenizer(model_name: str = EMBEDDING_MODEL) -> PreTrainedTokenizerBase:
    """Return a cached tokenizer for *model_name*.

    中文：进程内只加载一次，以保证 token 计数与 embedding 模型一致并避免重复启动开销。
    """
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_to_sentence(text: str) -> str:
    """
    Trim *text* so it ends at a sentence boundary (. ! ?).
    If no sentence boundary is found, return *text* unchanged.
    """
    # Find the last sentence-ending punctuation
    for m in reversed(list(re.finditer(r"[.!?]['\"]?", text))):
        candidate = text[: m.end()].strip()
        if len(candidate) >= MIN_CHUNK_CHARS:
            return candidate
    return text.strip()


def _start_at_sentence(text: str) -> str:
    """
    Drop leading text that is a fragment (doesn't start with a capital
    letter or quotation mark).  Best-effort only.
    """
    text = text.lstrip()
    if not text:
        return text
    # If it already starts with a capital, return as-is
    if text[0].isupper() or text[0] in ('"', "'", "("):
        return text
    # Otherwise, find the first sentence start
    m = _SENTENCE_END_RE.search(text)
    if m:
        return text[m.end():].lstrip()
    return text


def _split_into_sentence_units(text: str) -> list[str]:
    """
    Best-effort sentence splitting that preserves sentence-ending punctuation.
    """
    text = text.strip()
    if not text:
        return []
    units = [part.strip() for part in _SENTENCE_END_RE.split(text) if part.strip()]
    return units or [text]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """
    Split text into overlapping token windows, trimming at sentence boundaries.

    Parameters
    ----------
    chunk_size:
        Target number of tokens per chunk.
    overlap:
        Number of tokens of overlap between consecutive chunks.
    model_name:
        HuggingFace model name used to load the tokenizer.

    中文：根据来源和内容类型选择分块预算。它不丢失结构元数据，返回的 payload 可直接交给
    向量化和数据库加载阶段。
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = get_tokenizer(model_name)

    def _resolve_config(
        self,
        source_type: str,
        content_kind: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, int]:
        """Resolve source-specific sizes while allowing legacy caller overrides.

        中文：未配置的来源继续使用实例默认值，保持旧的 ``chunk_section`` 调用语义。
        """
        defaults = _CHUNK_CONFIGS.get(
            (source_type, content_kind),
            {
                "chunk_size": chunk_size or self.chunk_size,
                "overlap": overlap or self.overlap,
                "fallback_chunk_size": chunk_size or self.chunk_size,
                "fallback_overlap": overlap or self.overlap,
                "row_overlap": TABLE_ROW_OVERLAP,
            },
        )
        return {
            "chunk_size": chunk_size or defaults["chunk_size"],
            "overlap": overlap or defaults["overlap"],
            "fallback_chunk_size": defaults.get("fallback_chunk_size", chunk_size or defaults["chunk_size"]),
            "fallback_overlap": defaults.get("fallback_overlap", overlap or defaults["overlap"]),
            "row_overlap": defaults.get("row_overlap", TABLE_ROW_OVERLAP),
        }

    # ------------------------------------------------------------------
    # Single section
    # ------------------------------------------------------------------

    def _chunk_narrative_text(
        self,
        section_name: str,
        text: str,
        *,
        chunk_size: int,
        overlap: int,
        fallback_chunk_size: int,
        fallback_overlap: int,
    ) -> list[ChunkTuple]:
        """Pack complete sentences into overlapping chunks, with token windows as fallback.

        中文：超长单句或无法可靠切句时退回 token 窗口，保证任何可用文本仍可进入检索流程。
        """
        text = text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            return []

        token_ids: list[int] = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        total_tokens = len(token_ids)
        if total_tokens == 0:
            return []

        if total_tokens <= chunk_size:
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded = decoded.strip()
            if len(decoded) >= MIN_CHUNK_CHARS:
                return [(section_name, decoded, total_tokens)]
            return []

        sentence_units = _split_into_sentence_units(text)
        if len(sentence_units) <= 1:
            return self._chunk_windowed_text(
                section_name,
                text,
                chunk_size=fallback_chunk_size,
                overlap=fallback_overlap,
            )

        def encode_len(unit: str) -> int:
            return len(
                self.tokenizer.encode(
                    unit,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

        sentence_tokens = [encode_len(unit) for unit in sentence_units]
        if any(token_len > chunk_size for token_len in sentence_tokens):
            return self._chunk_windowed_text(
                section_name,
                text,
                chunk_size=fallback_chunk_size,
                overlap=fallback_overlap,
            )

        chunks: list[ChunkTuple] = []
        start_idx = 0

        while start_idx < len(sentence_units):
            end_idx = start_idx
            current_tokens = 0
            current_units: list[str] = []

            while end_idx < len(sentence_units):
                next_tokens = sentence_tokens[end_idx]
                if current_units and current_tokens + next_tokens > chunk_size:
                    break
                current_units.append(sentence_units[end_idx])
                current_tokens += next_tokens
                end_idx += 1

            if not current_units:
                break

            chunk_text = " ".join(current_units).strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append((section_name, chunk_text, current_tokens))

            if end_idx >= len(sentence_units):
                break

            overlap_tokens = 0
            overlap_start = end_idx
            while overlap_start > start_idx and overlap_tokens < overlap:
                overlap_start -= 1
                overlap_tokens += sentence_tokens[overlap_start]

            next_start = overlap_start if overlap_start > start_idx else end_idx
            if next_start <= start_idx:
                next_start = start_idx + 1
            start_idx = next_start

        return chunks

    def _chunk_windowed_text(
        self,
        section_name: str,
        text: str,
        *,
        chunk_size: int,
        overlap: int,
    ) -> list[ChunkTuple]:
        """Split text by token windows and trim only boundary fragments when possible.

        中文：这是叙述分块的兜底路径；重叠保护跨窗口上下文，最小长度过滤避免噪声向量。
        """
        token_ids: list[int] = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        total_tokens = len(token_ids)
        if total_tokens == 0:
            return []

        step = max(1, chunk_size - overlap)
        chunks: list[ChunkTuple] = []
        start = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            window_ids = token_ids[start:end]

            raw_text = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            raw_text = raw_text.strip()

            if end < total_tokens:
                trimmed = _trim_to_sentence(raw_text)
            else:
                trimmed = raw_text

            if start > 0:
                trimmed = _start_at_sentence(trimmed)

            if len(trimmed) >= MIN_CHUNK_CHARS:
                actual_ids = self.tokenizer.encode(
                    trimmed,
                    add_special_tokens=False,
                    truncation=False,
                )
                chunks.append((section_name, trimmed, len(actual_ids)))

            start += step
            if end == total_tokens:
                break

        return chunks

    def _chunk_table_block(
        self,
        section_name: str,
        text: str,
        *,
        chunk_size: int,
        row_overlap: int,
    ) -> list[ChunkTuple]:
        """
        Chunk a structured table block by rows rather than sentence boundaries.

        中文：表格的单元格语义依赖行和表头，因此只在行边界切分，并复制有限的相邻行作为上下文。
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        prefix_lines: list[str] = []
        suffix_lines: list[str] = []
        row_lines: list[str] = []
        for line in lines:
            if line.startswith("[ROW]"):
                row_lines.append(line)
            elif line.startswith("[/TABLE]"):
                suffix_lines.append(line)
            else:
                prefix_lines.append(line)

        if not row_lines:
            return self._chunk_narrative_text(
                section_name,
                text,
                chunk_size=chunk_size,
                overlap=self.overlap,
                fallback_chunk_size=chunk_size,
                fallback_overlap=self.overlap,
            )

        def token_count_for(candidate_lines: list[str]) -> int:
            return len(
                self.tokenizer.encode(
                    "\n".join(candidate_lines),
                    add_special_tokens=False,
                    truncation=False,
                )
            )

        chunks: list[ChunkTuple] = []
        start_idx = 0
        suffix_lines = suffix_lines or ["[/TABLE]"]

        while start_idx < len(row_lines):
            candidate_rows: list[str] = []
            end_idx = start_idx

            while end_idx < len(row_lines):
                next_rows = candidate_rows + [row_lines[end_idx]]
                next_lines = prefix_lines + next_rows + suffix_lines
                if end_idx > start_idx and token_count_for(next_lines) > chunk_size:
                    break
                candidate_rows = next_rows
                end_idx += 1
                if token_count_for(next_lines) > chunk_size:
                    break

            candidate_lines = prefix_lines + candidate_rows + suffix_lines
            chunk_text = "\n".join(candidate_lines).strip()
            chunk_tokens = token_count_for(candidate_lines)
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append((section_name, chunk_text, chunk_tokens))

            if end_idx >= len(row_lines):
                break

            next_start = max(start_idx + 1, end_idx - row_overlap)
            if next_start <= start_idx:
                next_start = end_idx
            start_idx = next_start

        return chunks

    def chunk_section(
        self,
        section_name: str,
        text: str,
    ) -> list[ChunkTuple]:
        """
        Chunk a single section of text.

        Parameters
        ----------
        section_name:
            Human-readable section label (e.g. 'MD&A').
        text:
            The section's plain text.

        Returns
        -------
        list[ChunkTuple]
            Each element is (section_name, chunk_text, token_count).

        中文：兼容旧接口。含 ``[TABLE]`` 标记的文本会拆为叙述与表格两种策略。
        """
        text = text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            return []

        if "[TABLE]" not in text:
            return self._chunk_narrative_text(
                section_name,
                text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                fallback_chunk_size=self.chunk_size,
                fallback_overlap=self.overlap,
            )

        chunks: list[ChunkTuple] = []
        cursor = 0
        for match in _TABLE_BLOCK_RE.finditer(text):
            narrative_prefix = text[cursor:match.start()].strip()
            if narrative_prefix:
                chunks.extend(
                    self._chunk_narrative_text(
                        section_name,
                        narrative_prefix,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        fallback_chunk_size=self.chunk_size,
                        fallback_overlap=self.overlap,
                    )
                )

            table_block = match.group(0).strip()
            if table_block:
                chunks.extend(
                    self._chunk_table_block(
                        section_name,
                        table_block,
                        chunk_size=self.chunk_size,
                        row_overlap=TABLE_ROW_OVERLAP,
                    )
                )

            cursor = match.end()

        narrative_suffix = text[cursor:].strip()
        if narrative_suffix:
            chunks.extend(
                self._chunk_narrative_text(
                    section_name,
                    narrative_suffix,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    fallback_chunk_size=self.chunk_size,
                    fallback_overlap=self.overlap,
                )
            )

        return chunks

    def chunk_segment(self, segment: CleanerSegment) -> list[ChunkPayload]:
        """Chunk one typed segment and preserve source-specific retrieval metadata.

        中文：新闻标题会前置到每个 chunk；表格行跨度和电话会发言人会随 payload 一起保留。
        """
        config = self._resolve_config(segment.source_type, segment.content_kind)
        chunk_rows: list[ChunkTuple]

        if segment.content_kind == "table":
            chunk_rows = self._chunk_table_block(
                segment.section_name,
                segment.text,
                chunk_size=config["chunk_size"],
                row_overlap=config["row_overlap"],
            )
        else:
            chunk_rows = self._chunk_narrative_text(
                segment.section_name,
                segment.text,
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                fallback_chunk_size=config["fallback_chunk_size"],
                fallback_overlap=config["fallback_overlap"],
            )

        payloads: list[ChunkPayload] = []
        title_prefix = None
        if segment.source_type == "news" and segment.display_title:
            title_prefix = segment.display_title.strip()
        table_row_cursor = 0

        for _, chunk_text, token_count in chunk_rows:
            final_text = chunk_text.strip()
            if title_prefix and not final_text.startswith(title_prefix):
                final_text = f"{title_prefix}\n\n{final_text}"
                token_count = len(
                    self.tokenizer.encode(
                        final_text,
                        add_special_tokens=False,
                        truncation=False,
                    )
                )

            structure_meta = dict(segment.meta)
            if segment.content_kind == "table":
                row_lines = [line for line in final_text.splitlines() if line.startswith("[ROW]")]
                if row_lines:
                    start_row = table_row_cursor
                    end_row = start_row + len(row_lines) - 1
                    structure_meta["row_count"] = len(row_lines)
                    structure_meta["row_start"] = start_row
                    structure_meta["row_end"] = end_row
                    structure_meta["chunk_row_span"] = [structure_meta["row_start"], structure_meta["row_end"]]
                    table_row_cursor = max(start_row + 1, end_row - config["row_overlap"] + 1)

            payloads.append(
                ChunkPayload(
                    source_type=segment.source_type,
                    section_name=segment.section_name,
                    content_kind=segment.content_kind,
                    chunk_strategy=(
                        "table_rows"
                        if segment.content_kind == "table"
                        else (
                            "qa_pair"
                            if segment.content_kind == "qa"
                            else (
                                "article_sentence_pack"
                                if segment.source_type == "news"
                                else (
                                    "speaker_turn"
                                    if segment.source_type == "transcript" and segment.meta.get("speaker")
                                    else "sentence_pack"
                                )
                            )
                        )
                    ),
                    display_title=segment.display_title,
                    text=final_text,
                    token_count=token_count,
                    chunk_group_key=segment.meta.get("chunk_group_key"),
                    structure_meta=structure_meta,
                )
            )

        return payloads

    def chunk_segments(self, segments: Sequence[CleanerSegment]) -> list[ChunkPayload]:
        """Chunk an ordered segment sequence into a flat payload list.

        中文：保持输入顺序；空段只贡献空列表，不会中断同一文档的其他段。
        """
        all_chunks: list[ChunkPayload] = []
        for segment in segments:
            segment_chunks = self.chunk_segment(segment)
            all_chunks.extend(segment_chunks)
            logger.debug(
                "Segment '%s' (%s/%s): %d chunks",
                segment.section_name,
                segment.source_type,
                segment.content_kind,
                len(segment_chunks),
            )
        return all_chunks

    # ------------------------------------------------------------------
    # Multiple sections
    # ------------------------------------------------------------------

    def chunk_sections(
        self,
        sections: Sequence[tuple[str, str]],
    ) -> list[ChunkTuple]:
        """
        Chunk a list of (section_name, text) pairs.

        Parameters
        ----------
        sections:
            Typically the output of ``HTMLCleaner.clean()``.

        Returns
        -------
        list[ChunkTuple]
            Flat list of (section_name, chunk_text, token_count).

        中文：批量包装器没有额外的跨 section 合并，避免不同 SEC Item 的上下文混淆。
        """
        all_chunks: list[ChunkTuple] = []
        for section_name, text in sections:
            section_chunks = self.chunk_section(section_name, text)
            all_chunks.extend(section_chunks)
            logger.debug(
                "Section '%s': %d chunks from %d chars",
                section_name,
                len(section_chunks),
                len(text),
            )
        return all_chunks

    # ------------------------------------------------------------------
    # Convenience for transcript text
    # ------------------------------------------------------------------

    def chunk_transcript_section(
        self,
        section: str,
        text: str,
    ) -> list[ChunkTuple]:
        """
        Convenience wrapper for earnings call transcript sections.
        Identical to ``chunk_section`` but exposed with an explicit name.

        中文：保留显式方法名是为了让调用点表达来源语义；实际规则与普通文本分块相同。
        """
        return self.chunk_section(section, text)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def chunk_text(
    sections: Sequence[tuple[str, str]],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[ChunkTuple]:
    """Chunk a list of (section_name, text) pairs with default settings.

    中文：无状态便捷入口；需要来源感知分块时请传入 ``CleanerSegment`` 并调用类接口。
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_sections(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_text = """
    Item 7. Management's Discussion and Analysis of Financial Condition
    and Results of Operations.

    Overview
    --------
    Our revenue for fiscal year 2023 increased 12% year-over-year, driven
    primarily by strong growth in our cloud services segment.  Operating
    income expanded by 200 basis points as we achieved operating leverage.
    We generated $4.2 billion in free cash flow during the year.

    Segment Results
    ---------------
    Cloud Services revenue grew 34% to $18.5 billion, fueled by enterprise
    adoption of our platform.  Margins in this segment improved to 38%
    from 33% in the prior year.
    """
    chunker = Chunker()
    chunks = chunker.chunk_section("MD&A", sample_text)
    for i, (sec, text, tok) in enumerate(chunks):
        print(f"Chunk {i+1} [{sec}] ({tok} tokens): {text[:120]}...")
