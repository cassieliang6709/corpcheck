from __future__ import annotations

import re
from typing import Any

from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner
from corpcheck.ingestion.processors.segment_types import CleanerSegment

_SPEAKER_RE = re.compile(r"^(?P<speaker>[A-Z][A-Za-z0-9 .,&'/-]{1,80}):\s*(?P<body>.+)$")


def build_news_segments(article: dict[str, Any]) -> list[CleanerSegment]:
    title = (article.get("title") or "").strip()
    summary = (article.get("summary") or "").strip()
    content = (article.get("content") or "").strip()
    body = "\n\n".join(part for part in (summary, content) if part)
    if not body:
        return []

    if "<table" in body.lower():
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        raw_segments = cleaner.clean_text_segments(body)
        segments: list[CleanerSegment] = []
        for idx, segment in enumerate(raw_segments):
            segments.append(
                CleanerSegment(
                    source_type="news",
                    section_name=segment.section_name,
                    content_kind=segment.content_kind,
                    display_title=title or segment.display_title,
                    text=segment.text,
                    meta={
                        **segment.meta,
                        "chunk_group_key": f"news:{article['id']}:{segment.content_kind}:{idx}",
                        "article_id": article["id"],
                    },
                )
            )
        return segments

    return [
        CleanerSegment(
            source_type="news",
            section_name="news_article",
            content_kind="narrative",
            display_title=title or "News Article",
            text=body,
            meta={
                "chunk_group_key": f"news:{article['id']}:narrative:0",
                "article_id": article["id"],
            },
        )
    ]


def _speaker_turns(text: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
        match = _SPEAKER_RE.match(paragraph)
        if match:
            if current_parts:
                turns.append(
                    {
                        "speaker": current_speaker or "Unknown",
                        "text": "\n\n".join(current_parts).strip(),
                    }
                )
            current_speaker = match.group("speaker").strip()
            current_parts = [match.group("body").strip()]
        else:
            current_parts.append(paragraph)

    if current_parts:
        turns.append(
            {
                "speaker": current_speaker or "Unknown",
                "text": "\n\n".join(current_parts).strip(),
            }
        )

    return turns


def build_transcript_segments(
    ticker: str,
    fiscal_year: int,
    quarter: int,
    sections: dict[str, str],
) -> list[CleanerSegment]:
    segments: list[CleanerSegment] = []
    qa_pair_index = 0

    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue
        turns = _speaker_turns(section_text)
        if not turns:
            continue

        if section_name == "qa":
            turn_idx = 0
            while turn_idx < len(turns):
                question = turns[turn_idx]
                answer = turns[turn_idx + 1] if turn_idx + 1 < len(turns) else None
                pair_text_parts = [f"{question['speaker']}: {question['text']}"]
                speakers = [question["speaker"]]
                if answer is not None:
                    pair_text_parts.append(f"{answer['speaker']}: {answer['text']}")
                    speakers.append(answer["speaker"])
                segments.append(
                    CleanerSegment(
                        source_type="transcript",
                        section_name=section_name,
                        content_kind="qa",
                        display_title=f"{ticker} Q{quarter} {fiscal_year} Q&A",
                        text="\n\n".join(pair_text_parts),
                        meta={
                            "qa_pair_id": qa_pair_index,
                            "speakers": speakers,
                            "chunk_group_key": f"transcript:{ticker}:{fiscal_year}:Q{quarter}:qa:{qa_pair_index}",
                        },
                    )
                )
                qa_pair_index += 1
                turn_idx += 2
            continue

        for turn_idx, turn in enumerate(turns):
            segments.append(
                CleanerSegment(
                    source_type="transcript",
                    section_name=section_name,
                    content_kind="narrative",
                    display_title=f"{ticker} Q{quarter} {fiscal_year} {section_name.replace('_', ' ').title()}",
                    text=f"{turn['speaker']}: {turn['text']}",
                    meta={
                        "speaker": turn["speaker"],
                        "turn_index": turn_idx,
                        "chunk_group_key": (
                            f"transcript:{ticker}:{fiscal_year}:Q{quarter}:{section_name}:{turn_idx}"
                        ),
                    },
                )
            )

    return segments
