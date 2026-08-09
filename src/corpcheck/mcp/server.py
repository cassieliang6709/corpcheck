"""stdio MCP server over the CorpCheck retrieval stack.

Three tools, in the order an honest agent should use them:

``check_answerable``
    Ask whether the corpus can support an answer *before* spending a generation.
    Runs retrieval, applies the same double-threshold abstain gate the ``/chat``
    endpoint uses, and returns the measured similarities. No LLM is contacted.
``search_filings``
    Retrieve evidence blocks with full provenance (company, filing type, fiscal
    year, period, accession number).
``get_filing_context``
    Widen a single hit into its surrounding passage, or open a filing by
    accession number.

All three call :func:`corpcheck.retrieval.pipeline.retrieve`. None of them
reimplements search, ranking, or the revision filter — the whole point of the
single-entry-point rule is that the offline IR numbers describe this path too.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from corpcheck import settings as config
from corpcheck.db import close_pool, get_pool
from corpcheck.mcp.provenance import (
    evidence_block,
    load_filing_provenance,
    superseded_check,
)
from corpcheck.models import ChunkResult
from corpcheck.retrieval import load_known_tickers, retrieve
from corpcheck.retrieval.abstain import _dense_similarities, evaluate_confidence
from corpcheck.retrieval.search import get_model

logger = logging.getLogger(__name__)

# stdio transport: stdout is the JSON-RPC channel and anything else written there
# corrupts the stream. Logs go to stderr, always.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

SERVER_INSTRUCTIONS = """\
CorpCheck exposes SEC filing evidence with audit-grade provenance.

Call check_answerable first when the answer matters. It reports, without invoking
any language model, whether the corpus actually contains evidence for a question.
If it says answerable=false, say so — do not answer from prior knowledge and
present it as if it came from the filings.

Every evidence block carries the company, filing type, fiscal year, period, and
SEC accession number. Cite those. Original chunks from sections replaced by a
later amendment (10-K/A, 10-Q/A) are excluded from results and withheld from
direct lookup; unchanged original sections remain available.
"""

# Truncation guard for search results: an agent's context is finite, and 10 full
# filing chunks can run to tens of thousands of characters. get_filing_context
# returns untruncated text when the agent decides it needs the whole passage.
_SEARCH_SNIPPET_CHARS = 2000


def _corpus_governance() -> dict[str, Any]:
    """State the governance actually in force, so a client never has to assume it."""
    return {
        "revision_filter_enabled": config.REVISION_FILTER_ENABLED,
        "note": (
            "Original chunks from amended sections are dropped from the candidate "
            "pool before ranking; unchanged sections remain available."
            if config.REVISION_FILTER_ENABLED
            else "REVISION_FILTER_ENABLED is off: results MAY include superseded "
            "filings. Do not rely on these results for authoritative figures."
        ),
    }


def _coverage(chunks: list[ChunkResult]) -> dict[str, Any]:
    """Describe the shape of the evidence pool, not just its score.

    A high top-1 similarity drawn entirely from one filing is a different
    situation from the same similarity spread over three companies and two fiscal
    years, and the caller is better placed than this server to decide which it
    needed.
    """
    return {
        "retrieved": len(chunks),
        "with_dense_score": sum(1 for c in chunks if c.cos_sim is not None),
        "sparse_only": sum(1 for c in chunks if c.cos_sim is None),
        "companies": sorted({c.company for c in chunks if c.company}),
        "filing_types": sorted({c.filing_type for c in chunks if c.filing_type}),
        "fiscal_years": sorted({c.fiscal_year for c in chunks if c.fiscal_year is not None}),
        "source_types": sorted({c.source_type for c in chunks if c.source_type}),
    }


def _gate_status(chunks: list[ChunkResult]) -> str:
    """Name *which* condition decided the gate, for logs and for the caller.

    ``evaluate_confidence`` returns a user-facing sentence; this classifies the
    same decision into a stable machine-readable code without duplicating the
    thresholds — the codes are derived from the identical similarity list the gate
    reads.
    """
    if not chunks:
        return "no_results"
    sims = _dense_similarities(chunks)
    if not sims:
        return "sparse_only_unmeasurable"
    if sims[0] < config.ABSTAIN_TOP1_MIN:
        return "below_top1_floor"
    if sum(sims[:3]) / len(sims[:3]) < config.ABSTAIN_MEAN_TOP3_MIN:
        return "below_mean_top3_floor"
    return "pass"


@asynccontextmanager
async def _lifespan(_server: MCPServer) -> AsyncIterator[None]:
    pool = await get_pool()
    # Company-name → ticker caches back the in-query company detection that
    # retrieve() relies on. Skipping this would silently degrade ranking on the
    # MCP path relative to the HTTP path.
    await load_known_tickers(pool)
    # Warm the embedding model here rather than on the first tool call. Loading it
    # lazily makes the first query take tens of seconds and, worse, turns a model
    # or hub failure into a mid-conversation tool error instead of a startup
    # failure the client can surface plainly.
    get_model()
    logger.info("CorpCheck MCP server ready (db=%s:%s)", config.DB_HOST, config.DB_PORT)
    try:
        yield
    finally:
        await close_pool()


def build_server() -> MCPServer:
    server = MCPServer(
        name="corpcheck",
        title="CorpCheck — SEC filing evidence",
        version="0.1.0",
        instructions=SERVER_INSTRUCTIONS,
        lifespan=_lifespan,
    )

    @server.tool(
        name="check_answerable",
        title="Check whether the filings can answer this",
        description=(
            "Report whether the indexed SEC filings contain evidence strong enough "
            "to answer a question, WITHOUT calling any language model. Runs the same "
            "retrieval and the same double-threshold confidence gate that guards "
            "answer generation, and returns the measured cosine similarities, the "
            "thresholds they were compared against, and the coverage of the evidence "
            "pool. Use this before answering anything where being wrong is worse "
            "than being silent."
        ),
    )
    async def check_answerable(
        query: Annotated[str, Field(description="The question, in natural language.")],
        k: Annotated[int, Field(ge=1, le=50, description="Candidates to assess.")] = 5,
        company: Annotated[
            Optional[str], Field(description="Restrict to a ticker or company name.")
        ] = None,
        filing_type: Annotated[
            Optional[str], Field(description="Restrict to a filing type, e.g. '10-K'.")
        ] = None,
        fiscal_year: Annotated[
            Optional[int], Field(description="Restrict to a fiscal year.")
        ] = None,
    ) -> dict[str, Any]:
        pool = await get_pool()
        chunks = await retrieve(
            pool=pool,
            query=query,
            k=k,
            alpha=config.DEFAULT_ALPHA,
            sector=None,
            company=company,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
        )
        decision = evaluate_confidence(chunks)
        status = _gate_status(chunks)
        logger.info("check_answerable %r -> %s (%s)", query[:120], status, decision.detail)

        return {
            "query": query,
            "answerable": not decision.abstain,
            "gate_status": status,
            "reason": decision.reason or "Evidence passed both confidence floors.",
            "llm_consulted": False,
            "similarity": {
                # Raw, unboosted dense cosine. The fused `score` on each evidence
                # block is min-max normalised and is therefore ~1.0 at rank 1 even
                # for a pool of noise; only cos_sim is comparable across queries.
                "top1_cos_sim": decision.top1,
                "mean_top3_cos_sim": decision.mean_top3,
                "top1_min": config.ABSTAIN_TOP1_MIN,
                "mean_top3_min": config.ABSTAIN_MEAN_TOP3_MIN,
            },
            "coverage": _coverage(chunks),
            "governance": _corpus_governance(),
        }

    @server.tool(
        name="search_filings",
        title="Search SEC filings",
        description=(
            "Hybrid dense + sparse retrieval over the SEC filing corpus. Returns "
            "evidence blocks with full provenance: company, filing type, fiscal "
            "year, period, and SEC accession number. Original chunks from sections "
            "replaced by a later amendment are excluded. The response also carries the "
            "abstain-gate verdict, so weak evidence is visibly labelled as weak "
            "rather than silently returned as if it were solid."
        ),
    )
    async def search_filings(
        query: Annotated[str, Field(description="The search query, in natural language.")],
        k: Annotated[int, Field(ge=1, le=25, description="Number of evidence blocks.")] = 5,
        company: Annotated[
            Optional[str], Field(description="Restrict to a ticker or company name.")
        ] = None,
        filing_type: Annotated[
            Optional[str], Field(description="Restrict to a filing type, e.g. '10-K'.")
        ] = None,
        fiscal_year: Annotated[
            Optional[int], Field(description="Restrict to a fiscal year.")
        ] = None,
        sector: Annotated[Optional[str], Field(description="Restrict to a sector.")] = None,
        alpha: Annotated[
            float,
            Field(ge=0.0, le=1.0, description="Dense/sparse blend weight."),
        ] = config.DEFAULT_ALPHA,
    ) -> dict[str, Any]:
        pool = await get_pool()
        chunks = await retrieve(
            pool=pool,
            query=query,
            k=k,
            alpha=alpha,
            sector=sector,
            company=company,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
        )
        prov = await load_filing_provenance(pool, chunks)
        decision = evaluate_confidence(chunks)

        return {
            "query": query,
            "results": [
                evidence_block(c, prov.get(c.chunk_id), max_chars=_SEARCH_SNIPPET_CHARS)
                for c in chunks
            ],
            "abstain": {
                "would_abstain": decision.abstain,
                "reason": decision.reason or None,
                "top1_cos_sim": decision.top1,
                "mean_top3_cos_sim": decision.mean_top3,
            },
            "coverage": _coverage(chunks),
            "governance": _corpus_governance(),
        }

    @server.tool(
        name="get_filing_context",
        title="Open the surrounding filing text",
        description=(
            "Fetch untruncated source text. Given a chunk_id, returns that chunk "
            "plus its neighbours in document order so a quotation can be read in "
            "context. Given an accession_number, opens that filing from the "
            "beginning. Text from an original section replaced by a later amendment "
            "is withheld; unchanged original sections remain available."
        ),
    )
    async def get_filing_context(
        chunk_id: Annotated[
            Optional[str],
            Field(description="A chunk_id from search_filings results."),
        ] = None,
        accession_number: Annotated[
            Optional[str],
            Field(description="An SEC accession number, e.g. '0000320193-23-000106'."),
        ] = None,
        window: Annotated[
            int,
            Field(ge=0, le=10, description="Neighbouring chunks to include on each side."),
        ] = 2,
        max_chunks: Annotated[
            int,
            Field(ge=1, le=40, description="Cap on returned chunks (accession mode)."),
        ] = 10,
    ) -> dict[str, Any]:
        if not chunk_id and not accession_number:
            return {"error": "Provide either chunk_id or accession_number."}
        if chunk_id and accession_number:
            return {"error": "Provide exactly one of chunk_id or accession_number."}

        pool = await get_pool()

        if chunk_id:
            try:
                numeric_id = int(chunk_id)
            except (TypeError, ValueError):
                return {
                    "error": (
                        f"chunk_id {chunk_id!r} is not an SEC filing chunk id. "
                        "Filing context is only available for source_type='sec'."
                    )
                }
            anchor = await pool.fetchrow(
                """
                SELECT c.filing_id, c.chunk_index, c.section_name, c.ticker,
                       c.filing_type, c.fiscal_year, c.period
                FROM chunks c
                WHERE c.id = $1
                """,
                numeric_id,
            )
            if anchor is None:
                return {"error": f"No SEC chunk found with chunk_id {chunk_id!r}."}
            filing_id = anchor["filing_id"]
            low = max((anchor["chunk_index"] or 0) - window, 0)
            high = (anchor["chunk_index"] or 0) + window
            limit = max_chunks
        else:
            anchor = await pool.fetchrow(
                """
                SELECT f.id AS filing_id, f.ticker, f.filing_type, f.fiscal_year,
                       f.period, first_chunk.section_name
                FROM filings AS f
                LEFT JOIN LATERAL (
                    SELECT c.section_name
                    FROM chunks AS c
                    WHERE c.filing_id = f.id
                    ORDER BY c.chunk_index NULLS LAST, c.id
                    LIMIT 1
                ) AS first_chunk ON TRUE
                WHERE f.accession_number = $1
                """,
                accession_number,
            )
            if anchor is None:
                return {
                    "error": f"No filing found with accession_number {accession_number!r}."
                }
            filing_id = anchor["filing_id"]
            low, high = 0, 10**9
            limit = max_chunks

        # Version governance applies to direct lookup too, not only to search.
        blocked, message = await superseded_check(
            pool,
            {
                "source_type": "sec",
                "company": anchor["ticker"],
                "filing_type": anchor["filing_type"],
                "fiscal_year": anchor["fiscal_year"],
                "period_label": anchor["period"],
                "section_name": anchor["section_name"],
            },
        )
        if blocked:
            return {
                "superseded": True,
                "text_withheld": True,
                "reason": message,
                "governance": _corpus_governance(),
            }

        filing = await pool.fetchrow(
            """
            SELECT f.ticker, co.name AS company_name, f.filing_type, f.fiscal_year,
                   f.period, f.filed_date, f.period_of_report, f.accession_number,
                   f.cik, f.source_url
            FROM filings f
            LEFT JOIN companies co ON co.ticker = f.ticker
            WHERE f.id = $1
            """,
            filing_id,
        )
        rows = await pool.fetch(
            """
            SELECT id::text AS chunk_id, chunk_index, section_name, content
            FROM chunks
            WHERE filing_id = $1 AND chunk_index BETWEEN $2 AND $3
            ORDER BY chunk_index
            LIMIT $4
            """,
            filing_id,
            low,
            high,
            limit,
        )

        return {
            "filing": {
                "company": filing["ticker"],
                "company_name": filing["company_name"],
                "filing_type": filing["filing_type"],
                "fiscal_year": filing["fiscal_year"],
                "period": filing["period"],
                "filed_date": filing["filed_date"].isoformat()
                if filing["filed_date"]
                else None,
                "period_of_report": filing["period_of_report"].isoformat()
                if filing["period_of_report"]
                else None,
                "accession_number": filing["accession_number"],
                "cik": filing["cik"],
                "source_url": filing["source_url"],
            },
            "superseded": False,
            "anchor_chunk_id": chunk_id,
            "chunks": [
                {
                    "chunk_id": r["chunk_id"],
                    "chunk_index": r["chunk_index"],
                    "section": r["section_name"],
                    "is_anchor": r["chunk_id"] == chunk_id,
                    "text": r["content"],
                }
                for r in rows
            ],
            "governance": _corpus_governance(),
        }

    return server


def main() -> None:
    """Console-script entry point. stdio transport only."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
