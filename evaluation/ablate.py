#!/usr/bin/env python3
"""Run :mod:`evaluation.ir_eval` with a named query-understanding feature turned off.

中文：在同一份实时语料上逐项关闭查询理解特性，避免把代码差异与语料变化混在一起；
该实验入口只服务于可复现的归因比较。

A "before" number is only trustworthy if it can be reproduced against the same
corpus as the "after" number. Checking out an old commit does not give that:
the corpus grows, so the two runs would differ by both code *and* data. This
module instead disables one feature at a time in the live tree, which keeps the
database, the chunking, and everything downstream identical.

Ablations
---------

``year-notation``
    Restores the pre-fix year regexes: ``\\b(20\\d{2})\\b`` for years, no
    two-digit ``FY22`` form, no ``Q2 2023`` form, and an annual hint that only
    recognises four-digit fiscal years. This is the state of
    ``query_parse.py`` before commit 5422531.

``company-shorthand``
    Disables camel-cased ticker shorthand (``JnJ`` -> ``JNJ``) and the
    single-head-word company alias (``Costco`` -> ``COST``).

``company-scope``
    Restores the pre-``COMPANY_SCOPE_ENABLED`` behaviour: a company detected in
    the query text only *boosts* its issuer's chunks instead of restricting the
    candidate pool to them.

Usage::

    python -m evaluation.ablate --off year-notation company-shorthand \\
        --label 2026-08-06_baseline
    python -m evaluation.ablate --off company-shorthand --label 2026-08-06_yearfix

Any argument not consumed here is forwarded verbatim to ``ir_eval``.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

from corpcheck.retrieval import pipeline as retrieval_pipeline
from corpcheck.retrieval import query_parse as qp
from evaluation import ir_eval

# A pattern that can never match, used to switch a detection path off without
# changing the surrounding control flow.
_NEVER = re.compile(r"(?!x)x")


def _disable_year_notation() -> None:
    """Revert year detection to the pre-5422531 regexes."""
    qp._YEAR_RE = re.compile(r"\b(20\d{2})\b")
    qp._YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*(?:-|–|to)\s*(20\d{2})\b", re.IGNORECASE)
    qp._SHORT_FISCAL_YEAR_RE = _NEVER
    qp._SHORT_YEAR_RANGE_RE = _NEVER
    qp._QUARTER_YEAR_RE = _NEVER
    qp._FISCAL_YEAR_RE = re.compile(
        r"\b(?:FY\s*20\d{2}|fiscal\s+year\s*20\d{2})\b", re.IGNORECASE
    )


def _disable_company_shorthand() -> None:
    """Drop camel-cased ticker shorthand and single-head-word company aliases.

    ``_MIN_HEAD_ALIAS_LEN`` is read inside ``_generate_company_aliases`` at call
    time, so raising it above any real head word suppresses the alias before
    ``load_known_tickers`` builds the cache.
    """
    qp._MIXED_CASE_TICKER_RE = _NEVER
    qp._MIN_HEAD_ALIAS_LEN = 10_000


def _disable_company_scope() -> None:
    """Turn the detected-company hard filter back into a soft boost.

    ``pipeline`` binds the flag by value at import time, so the module attribute
    -- not ``settings`` -- is what the retrieval path actually reads.
    """
    retrieval_pipeline.COMPANY_SCOPE_ENABLED = False


ABLATIONS = {
    "year-notation": _disable_year_notation,
    "company-shorthand": _disable_company_shorthand,
    "company-scope": _disable_company_scope,
}


def main(argv: Optional[list[str]] = None) -> int:
    """Disable selected query features, then delegate to the unchanged IR CLI.

    中文：入口只在当前进程中施加明确的消融开关；其余参数仍由原始 IR 评测器验证。
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--off",
        nargs="+",
        choices=sorted(ABLATIONS),
        required=True,
        help="Features to disable before running the evaluation",
    )
    args, passthrough = parser.parse_known_args(argv)

    for name in args.off:
        ABLATIONS[name]()
    print(f"ablate: disabled {', '.join(args.off)}", file=sys.stderr)

    return ir_eval.main(passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
