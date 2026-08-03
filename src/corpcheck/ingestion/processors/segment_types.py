from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanerSegment:
    source_type: str
    section_name: str
    content_kind: str
    display_title: str | None
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkPayload:
    source_type: str
    section_name: str
    content_kind: str
    chunk_strategy: str
    display_title: str | None
    text: str
    token_count: int
    chunk_group_key: str | None = None
    structure_meta: dict[str, Any] = field(default_factory=dict)
