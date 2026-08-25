"""Deterministic IR metrics with token-overlap weak supervision.

中文：这里的函数是纯确定性的“量尺”，用于比较检索配置而非声称事实正确性；
统一的弱监督偏差可使 A/B 差异仍然具有解释性。

Everything here is pure: no database, no network, no model. That is deliberate —
these functions are the measuring instrument, so they must be unit-testable and
reproducible on their own.

The weak-supervision premise: FinanceBench gives a gold evidence *span* (a page
or table lifted out of the filing), not a chunk id. Our corpus is chunked on a
different boundary, so exact matching is impossible. Instead a retrieved chunk
counts as covering a gold span when it contains enough of that span's
content-bearing tokens.

This is a proxy, and it is biased in a knowable direction: it can credit a chunk
that shares vocabulary without stating the fact, and it can miss a chunk that
paraphrases. Both biases are held constant across configurations, which is what
makes it useful for A/B comparison even though the absolute number is soft.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Optional

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Multi-word filing boilerplate. These phrases appear on nearly every page of a
# 10-K, so leaving them in lets an arbitrary chunk share tokens with an arbitrary
# gold span and inflates overlap across the board. Stripped from gold and
# candidate alike so the comparison stays symmetric.
BOILERPLATE_PHRASES: tuple[str, ...] = (
    "see accompanying notes to consolidated financial statements",
    "the accompanying notes are an integral part of these consolidated financial statements",
    "the accompanying notes are an integral part of these financial statements",
    "see accompanying notes to the consolidated financial statements",
    "notes to consolidated financial statements",
    "see accompanying notes",
    "table of contents",
    "united states securities and exchange commission",
    "securities and exchange commission",
    "washington d c 20549",
    "annual report pursuant to section",
    "quarterly report pursuant to section",
    "of the securities exchange act of",
    "form 10 k",
    "form 10 q",
    "in millions except per share data",
    "in millions except per share amounts",
    "in thousands except per share data",
    "dollars in millions",
    "amounts in millions",
)

# Standard English function words. Gold spans here run ~150 tokens, so without
# this the overlap score is dominated by "the", "of", "and".
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no nor
    not of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why with would you your yours yourself yourselves will may
    shall must been also per such upon within
    """.split()
)

# Shared by ir_eval and oracle so the sweep and its ceiling are always
# reported over identical thresholds.
THRESHOLD_SWEEP_DEFAULT: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

_THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace.

    Thousands separators are removed *before* punctuation stripping so that
    ``11,588`` normalises to the single token ``11588`` rather than splitting
    into ``11`` and ``588``. In filing tables the exact figure is the most
    discriminative token available, so keeping it intact matters.
    """
    if not text:
        return ""
    lowered = text.lower()
    lowered = _THOUSANDS_SEP_RE.sub("", lowered)
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def strip_boilerplate(normalized: str) -> str:
    """Remove known filing boilerplate phrases from already-normalised text."""
    result = normalized
    for phrase in BOILERPLATE_PHRASES:
        result = result.replace(phrase, " ")
    return _WHITESPACE_RE.sub(" ", result).strip()


def content_tokens(
    text: str,
    *,
    remove_boilerplate: bool = True,
    remove_stopwords: bool = True,
) -> set[str]:
    """Normalise ``text`` and return its set of content-bearing tokens."""
    normalized = normalize_text(text)
    if remove_boilerplate:
        normalized = strip_boilerplate(normalized)
    tokens = normalized.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


def token_overlap(
    candidate: str,
    gold: str,
    *,
    remove_boilerplate: bool = True,
    remove_stopwords: bool = True,
) -> float:
    """Fraction of the *gold* span's content tokens that appear in ``candidate``.

    Recall-oriented on purpose: the denominator is the gold span, so a long chunk
    is not penalised for containing material beyond the evidence, but a chunk
    that only clips the edge of the evidence scores low. Returns 0.0 when the
    gold span has no content tokens left after cleaning.
    """
    gold_tokens = content_tokens(
        gold, remove_boilerplate=remove_boilerplate, remove_stopwords=remove_stopwords
    )
    if not gold_tokens:
        return 0.0
    cand_tokens = content_tokens(
        candidate, remove_boilerplate=remove_boilerplate, remove_stopwords=remove_stopwords
    )
    return len(cand_tokens & gold_tokens) / len(gold_tokens)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def first_hit_rank(
    overlaps: Sequence[float],
    threshold: float,
) -> Optional[int]:
    """Return the 1-indexed rank of the first overlap at or above ``threshold``.

    ``overlaps`` must be ordered by descending retrieval score.
    """
    for rank, value in enumerate(overlaps, start=1):
        if value >= threshold:
            return rank
    return None


def recall_at_k(hit_ranks: Iterable[Optional[int]], k: int) -> float:
    """Fraction of gold spans covered by some chunk within the top ``k``.

    Each gold evidence span is one relevant item, so a question with three gold
    spans of which two are found scores 2/3 — not 1.0. This keeps multi-span
    questions from looking easier than they are.
    """
    ranks = list(hit_ranks)
    if not ranks:
        return 0.0
    found = sum(1 for r in ranks if r is not None and r <= k)
    return found / len(ranks)


def hit_at_k(hit_ranks: Iterable[Optional[int]], k: int) -> float:
    """1.0 if *any* gold span is covered within the top ``k``, else 0.0.

    Reported alongside recall because it answers a different question: recall
    asks how completely the evidence was assembled, hit asks whether the query
    landed at all.
    """
    return 1.0 if any(r is not None and r <= k for r in hit_ranks) else 0.0


def reciprocal_rank(hit_ranks: Iterable[Optional[int]], k: Optional[int] = None) -> float:
    """``1 / rank`` of the earliest chunk covering any gold span; 0.0 if none.

    Optionally truncated at ``k`` so MRR@k does not credit hits below the cutoff.
    """
    ranks = [r for r in hit_ranks if r is not None]
    if k is not None:
        ranks = [r for r in ranks if r <= k]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0
