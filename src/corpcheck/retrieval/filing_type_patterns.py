import re


FILING_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "10-K": re.compile(
        r"\b10-k\b"                # "10-K" or "10-k" (with hyphen)
        r"|\b10k\b"                # "10K" or "10k" (no hyphen)
        r"|\bannual report\b"
        r"|\bannual filing\b"
        r"|\bform 10-k\b"          # "Form 10-K" (as seen on SEC filings)
        r"|\bform 10k\b"
        r"|\byear.?end report\b"   # "year-end report" or "year end report"
        r"|\bfull.?year report\b"  # "full-year report" or "full year report"
        r"|\byearly report\b",
        re.IGNORECASE
    ),
    "10-Q": re.compile(
        r"\b10-q\b"
        r"|\b10q\b"
        r"|\bquarterly report\b"
        r"|\bquarterly filing\b"
        r"|\bform 10-q\b"
        r"|\bform 10q\b"
        r"|\bq[1-4] report\b"      # "Q1 report", "Q2 report", etc.
        r"|\bquarter.?end report\b",
        re.IGNORECASE
    ),
}
