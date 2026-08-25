"""Post-fusion adjustments based on what kind of evidence the question needs.

These are additive offsets applied on top of the fused [0, 1] score, so their
magnitude is directly comparable to the fusion output.

中文：重排只添加小的、可解释的证据形态偏好；它不会替代混合检索或改变用户设置的过滤范围。
"""

from __future__ import annotations

from typing import Optional


def compute_rerank_bonus(
    *,
    needs_quant: bool,
    needs_explanatory: bool,
    content_kind: Optional[str],
    source_type: str,
) -> float:
    """Nudge chunks whose form matches the question's evidence requirements.

    A "what was revenue in FY2019" question wants a table row; a "why did margins
    compress" question wants MD&A prose or an earnings-call answer.

    中文：数值问题偏向表格，解释性问题偏向叙述或电话会；所有偏移均在融合分数之后叠加。
    """
    bonus = 0.0
    if needs_quant and content_kind == "table":
        bonus += 0.20
    if needs_explanatory and content_kind == "narrative":
        bonus += 0.12
    if needs_explanatory and source_type == "transcript" and content_kind in {
        "qa",
        "narrative",
    }:
        bonus += 0.08
    if needs_quant and source_type == "news":
        bonus -= 0.03
    return bonus


def compose_article_title(row: dict) -> str:
    """Build a human-readable source label for citation display.

    中文：优先使用导入时提供的显示标题，缺失时根据来源类型和文件元数据生成稳定回退标题。
    """
    display_title = (row.get("display_title") or "").strip()
    if display_title:
        return display_title

    company_name = (row.get("company_name") or row.get("company") or "").strip()
    source_type = row.get("source_type") or "sec"
    filing_type = (row.get("filing_type") or "").strip()
    fiscal_year = row.get("fiscal_year")
    period_label = (row.get("period_label") or "").strip()

    if source_type == "news":
        return f"{company_name} News".strip() or "News"
    if source_type == "transcript":
        parts = [company_name, "Earnings Call"]
        if fiscal_year:
            parts.append(str(fiscal_year))
        if period_label:
            parts.append(period_label)
        return " ".join(part for part in parts if part).strip()

    if fiscal_year:
        return f"{company_name} {filing_type} {fiscal_year}".strip()
    return f"{company_name} {filing_type}".strip()
