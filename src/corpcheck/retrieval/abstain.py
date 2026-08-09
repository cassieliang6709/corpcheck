"""Retrieval-confidence gating: refuse before the LLM is ever consulted.

In audit-grade financial QA, declining to answer is strictly preferable to
serving a confident fabrication built on irrelevant context. This module decides
whether a candidate set is good enough to answer from.

Why this gates on ``cos_sim`` and not on ``ChunkResult.score``
--------------------------------------------------------------
``score`` is the fused ranking score, and it carries no absolute meaning:

* Reciprocal Rank Fusion scores a document by its *rank* in each arm, not by how
  similar it actually is. A pool of completely irrelevant chunks still has a
  rank 1.
* :func:`~corpcheck.retrieval.fusion.compute_rrf_scores` then min-max normalises,
  which by construction maps the best candidate to exactly ``1.0``.

So any threshold on ``score`` is unfalsifiable — measured against a pool of pure
noise, top-1 comes back as ``1.0`` and the mean of the top 3 as ``~0.98``. The
first version of this module gated on ``score`` and therefore never fired.

``cos_sim`` is the raw, unboosted cosine similarity between the query embedding
and the chunk embedding. It is comparable across queries, which is exactly what
a fixed threshold needs. Metadata boosts are excluded deliberately: boosting can
multiply a score by up to ~2.5x, which would let a match on ticker alone carry
an otherwise irrelevant chunk past the gate.

Threshold calibration
---------------------
Measured against the live corpus (419,830 chunks, all-MiniLM-L6-v2) over 15
in-domain financial questions and 10 out-of-domain questions::

    metric         in-domain               out-of-domain
    top-1 cos_sim  min 0.566, p50 0.704    max 0.342, p50 0.275
    mean top-3     min 0.554, p50 0.697    max 0.335, p50 0.258

The two populations are separated by a ~0.22-wide empty band, so the exact
placement inside it barely matters. The defaults sit nearer the out-of-domain
edge, trading a slightly higher chance of answering a marginal question for a
lower chance of refusing a good one.

Both thresholds are env-tunable, and the calibration script lives at
``evaluation/calibrate_abstain.py`` — rerun it whenever the embedding model or
the corpus changes, because these numbers are properties of that pairing rather
than universal constants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

from corpcheck.retrieval.query_parse import (
    detect_company_in_query,
    detect_unresolved_company_in_query,
    detect_years_in_query,
    resolve_company_filter,
)
from corpcheck.settings import ABSTAIN_MEAN_TOP3_MIN, ABSTAIN_TOP1_MIN

if TYPE_CHECKING:
    from corpcheck.models import ChunkResult

logger = logging.getLogger(__name__)


class AbstainDecision:
    """Outcome of the confidence gate.

    Truthy when the system should refuse. ``reason`` is safe to show a user;
    ``detail`` carries the measured values for logs and evaluation.
    """

    __slots__ = ("abstain", "reason", "top1", "mean_top3", "status")

    def __init__(
        self,
        abstain: bool,
        reason: str = "",
        top1: Optional[float] = None,
        mean_top3: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        self.abstain = abstain
        self.reason = reason
        self.top1 = top1
        self.mean_top3 = mean_top3
        self.status = status or ("abstain" if abstain else "pass")

    def __bool__(self) -> bool:
        return self.abstain

    @property
    def detail(self) -> str:
        def fmt(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        return f"top1={fmt(self.top1)} mean_top3={fmt(self.mean_top3)}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AbstainDecision abstain={self.abstain} {self.detail}>"


def _dense_similarities(results: Sequence[ChunkResult]) -> list[float]:
    """Unboosted cosine similarities present on the candidate set, best first.

    Chunks retrieved only by the sparse arm have no ``cos_sim`` and are skipped
    rather than treated as 0.0 — a missing measurement is not evidence of a bad
    one, and counting it as zero would drag the mean down and cause spurious
    refusals on keyword-shaped queries.

    Sorted independently of the fused ranking: metadata boosts can reorder the
    final list, so the top-ranked row is not necessarily the most semantically
    similar one. The question this gate asks is "is anything relevant here?",
    which is about the pool rather than the ordering.
    """
    sims = [r.cos_sim for r in results if r.cos_sim is not None]
    return sorted(sims, reverse=True)


def evaluate_confidence(
    results: Sequence[ChunkResult],
    top1_min: float = ABSTAIN_TOP1_MIN,
    mean_top3_min: float = ABSTAIN_MEAN_TOP3_MIN,
) -> AbstainDecision:
    """Decide whether *results* are strong enough to answer from.

    Two thresholds, because they catch different failures: ``top1_min`` rejects
    a pool with no strong anchor at all, while ``mean_top3_min`` rejects a pool
    where one lucky hit is surrounded by noise — enough to look confident, not
    enough to support an answer.
    """
    if not results:
        return AbstainDecision(
            True, "No matching passages were retrieved.", status="no_results"
        )

    sims = _dense_similarities(results)
    if not sims:
        # Sparse-only candidate set: nothing to measure on an absolute scale.
        # Allow it through rather than refuse on missing data, but say so.
        logger.warning(
            "Abstain gate skipped: no dense scores on %d candidates", len(results)
        )
        return AbstainDecision(False, status="sparse_only_unmeasurable")

    top1 = sims[0]
    mean_top3 = sum(sims[:3]) / len(sims[:3])

    if top1 < top1_min:
        return AbstainDecision(
            True,
            "The filings searched do not contain passages relevant enough to "
            "answer this question.",
            top1,
            mean_top3,
            "below_top1_floor",
        )

    if mean_top3 < mean_top3_min:
        return AbstainDecision(
            True,
            "Only one passage came close to matching this question, which is "
            "not enough supporting evidence to answer from.",
            top1,
            mean_top3,
            "below_mean_top3_floor",
        )

    return AbstainDecision(False, "", top1, mean_top3, "pass")


def evaluate_answerability(
    query: str,
    results: Sequence[ChunkResult],
    expected_company: Optional[str] = None,
) -> AbstainDecision:
    """Apply explicit metadata coverage checks, then the cosine confidence gate."""
    if not results:
        return evaluate_confidence(results)

    # An explicit API/MCP/evaluation filter is authoritative context. In that
    # mode, possessives elsewhere in the question ("CEO's compensation") are
    # not attempts to name an issuer.
    unresolved_company = (
        None
        if expected_company is not None
        else detect_unresolved_company_in_query(query)
    )
    if unresolved_company is not None:
        return AbstainDecision(
            True,
            f"The named issuer ({unresolved_company}) is not represented in the "
            "indexed filings.",
            status="unknown_company",
        )

    requested_company = (
        resolve_company_filter(expected_company)
        if expected_company is not None
        else detect_company_in_query(query)
    )
    if requested_company:
        requested_company = requested_company.upper()
    companies = {result.company.upper() for result in results if result.company}
    if requested_company is not None and requested_company not in companies:
        return AbstainDecision(
            True,
            "No retrieved filing passage matches the requested company.",
            status="company_mismatch",
        )

    requested_years = set(detect_years_in_query(query))
    company_results = (
        [
            result
            for result in results
            if result.company and result.company.upper() == requested_company
        ]
        if requested_company is not None
        else results
    )
    evidence_years: set[str] = set()
    for result in company_results:
        if result.fiscal_year is not None:
            evidence_years.add(str(result.fiscal_year))
        # A later filing commonly contains comparative columns for an earlier
        # year. Explicit years in the evidence text count as period coverage;
        # detect_years_in_query already avoids accession-number substrings.
        evidence_years.update(detect_years_in_query(result.text))
    if requested_years and requested_years.isdisjoint(evidence_years):
        return AbstainDecision(
            True,
            "No retrieved filing passage matches the requested fiscal year.",
            status="year_mismatch",
        )

    return evaluate_confidence(results)


def should_abstain(
    results: Sequence[ChunkResult],
    top1_min: float = ABSTAIN_TOP1_MIN,
    mean_top3_min: float = ABSTAIN_MEAN_TOP3_MIN,
) -> tuple[bool, str]:
    """Tuple-returning wrapper around :func:`evaluate_confidence`."""
    decision = evaluate_confidence(results, top1_min, mean_top3_min)
    return decision.abstain, decision.reason
