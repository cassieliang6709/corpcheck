from __future__ import annotations

import re

_NUMBER_RE = re.compile(
    r"""
    (?:
        \$\s*\d[\d,]*(?:\.\d+)? |
        \d[\d,]*(?:\.\d+)?\s*% |
        \d[\d,]*(?:\.\d+)?
    )
    """,
    re.VERBOSE,
)

_DATA_KEYWORD_RE = re.compile(
    r"\b("
    r"revenue|net income|operating income|gross margin|earnings per share|eps|"
    r"cash flow|free cash flow|operating cash flow|total assets|total liabilities|"
    r"shareholders'? equity|segment|inventory|guidance|backlog|debt|capex|"
    r"diluted|basic|million|billion"
    r")\b",
    re.IGNORECASE,
)

_QUANT_SECTIONS = {
    "financial statements",
    "selected financial data",
    "mda",
    "md&a",
    "management's discussion and analysis",
}


def compute_chunk_features(section_name: str | None, content: str) -> dict[str, float | int | bool]:
    text = content or ""
    section = (section_name or "").strip().lower()
    numeric_matches = _NUMBER_RE.findall(text)
    numeric_token_count = len(numeric_matches)
    char_count = max(len(text), 1)
    number_density = numeric_token_count / char_count
    keyword_hits = len(_DATA_KEYWORD_RE.findall(text))
    section_bonus = 2 if section in _QUANT_SECTIONS else 0
    table_bonus = 8 if "[TABLE]" in text or "[ROW]" in text or "[HEADER]" in text else 0
    data_signal_score = numeric_token_count + keyword_hits + section_bonus + table_bonus
    is_quantitative = data_signal_score >= 6

    return {
        "numeric_token_count": numeric_token_count,
        "number_density": round(number_density, 6),
        "data_signal_score": round(float(data_signal_score), 6),
        "is_quantitative": is_quantitative,
    }
