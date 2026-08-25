"""Typed hand-off objects between cleaning, segmentation, and chunking stages.

中文：这些 dataclass 只表达处理阶段之间的数据契约，避免各模块依赖无约束的字典字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanerSegment:
    """One semantically coherent source fragment before token chunking.

    中文：``meta`` 保存来源特有的结构信息，如表格索引或发言人；正文仍由 ``text`` 承载。
    """
    source_type: str
    section_name: str
    content_kind: str
    display_title: str | None
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkPayload:
    """One chunk plus retrieval metadata ready for embedding and loading.

    中文：保留分块策略和结构范围，方便检索结果向用户解释 chunk 来自表格、问答还是叙述文本。
    """
    source_type: str
    section_name: str
    content_kind: str
    chunk_strategy: str
    display_title: str | None
    text: str
    token_count: int
    chunk_group_key: str | None = None
    structure_meta: dict[str, Any] = field(default_factory=dict)
