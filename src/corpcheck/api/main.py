import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from corpcheck import settings as config
from corpcheck.db import close_pool, get_pool
from corpcheck.llm.chat import chat_once, stream_chat
from corpcheck.models import (
    AnswerabilityCoverage,
    AnswerabilityRequest,
    AnswerabilityResponse,
    AnswerabilitySimilarity,
    ChatRequest,
    ChatResponse,
    FilterOptionsResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from corpcheck.retrieval import load_known_tickers, retrieve
from corpcheck.retrieval.abstain import evaluate_answerability
from corpcheck.retrieval.search import get_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    await load_known_tickers(pool)
    # Fail during startup, rather than on the first user's request, if the
    # configured embedding model cannot be loaded. get_model() is process-cached,
    # so retrieval reuses this same instance.
    get_model()
    yield
    await close_pool()


app = FastAPI(title="RAG Retrieve Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    # `*` origins + credentials is rejected by browsers; keep credentials off
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """No-op when API_KEY env var is empty; otherwise require a matching header."""
    if not config.API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/filters", response_model=FilterOptionsResponse)
async def filters_endpoint():
    pool = await get_pool()
    async with pool.acquire() as conn:
        companies = await conn.fetch(
            """
            SELECT DISTINCT ticker
            FROM filings
            WHERE ticker IS NOT NULL AND ticker <> ''
            ORDER BY ticker
            """
        )
        filing_types = await conn.fetch(
            """
            SELECT DISTINCT filing_type
            FROM filings
            WHERE filing_type IS NOT NULL AND filing_type <> ''
            ORDER BY filing_type
            """
        )
        fiscal_years = await conn.fetch(
            """
            SELECT DISTINCT fiscal_year
            FROM filings
            WHERE fiscal_year IS NOT NULL
            ORDER BY fiscal_year DESC
            """
        )

    return FilterOptionsResponse(
        companies=[row["ticker"] for row in companies],
        filing_types=[row["filing_type"] for row in filing_types],
        fiscal_years=[row["fiscal_year"] for row in fiscal_years],
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(req: RetrieveRequest):
    pool = await get_pool()
    chunks = await retrieve(
        pool=pool,
        query=req.query,
        k=req.k,
        alpha=req.alpha,
        sector=req.sector,
        company=req.company,
        filing_type=req.filing_type,
        fiscal_year=req.year,
    )
    return RetrieveResponse(chunks=chunks)


@app.post("/answerability", response_model=AnswerabilityResponse)
async def answerability_endpoint(req: AnswerabilityRequest):
    """Retrieve evidence and apply the same pre-LLM gate used by chat and MCP."""
    pool = await get_pool()
    chunks = await retrieve(
        pool=pool,
        query=req.query,
        k=req.k,
        alpha=config.DEFAULT_ALPHA,
        sector=None,
        company=req.company,
        filing_type=req.filing_type,
        fiscal_year=req.year,
    )
    decision = evaluate_answerability(req.query, chunks, expected_company=req.company)

    return AnswerabilityResponse(
        query=req.query,
        answerable=not decision.abstain,
        gate_status=decision.status,
        reason=decision.reason or "Evidence passed both confidence floors.",
        llm_consulted=False,
        similarity=AnswerabilitySimilarity(
            top1_cos_sim=decision.top1,
            mean_top3_cos_sim=decision.mean_top3,
            top1_min=config.ABSTAIN_TOP1_MIN,
            mean_top3_min=config.ABSTAIN_MEAN_TOP3_MIN,
        ),
        coverage=AnswerabilityCoverage(
            retrieved=len(chunks),
            with_dense_score=sum(chunk.cos_sim is not None for chunk in chunks),
            sparse_only=sum(chunk.cos_sim is None for chunk in chunks),
            companies=sorted({chunk.company for chunk in chunks if chunk.company}),
            filing_types=sorted(
                {chunk.filing_type for chunk in chunks if chunk.filing_type}
            ),
            fiscal_years=sorted(
                {chunk.fiscal_year for chunk in chunks if chunk.fiscal_year is not None}
            ),
            source_types=sorted(
                {chunk.source_type for chunk in chunks if chunk.source_type}
            ),
        ),
        chunks=chunks,
    )


@app.post("/chat", dependencies=[Depends(require_api_key)])
async def chat_endpoint(req: ChatRequest, request: Request):
    if not config.SGLANG_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured (SGLANG_BASE_URL is empty)",
        )

    pool = await get_pool()
    chunks = await retrieve(
        pool=pool,
        query=req.query,
        k=req.k,
        alpha=req.alpha,
        sector=req.sector,
        company=req.company,
        filing_type=req.filing_type,
        fiscal_year=req.year,
    )

    # Confidence gate. This runs before the LLM is contacted at all: if the
    # retrieved evidence is too weak, the correct output is a refusal, and
    # sending the context anyway would only invite a fluent guess. Sources are
    # still returned so the user can see what *was* found and judge for
    # themselves.
    decision = evaluate_answerability(req.query, chunks, expected_company=req.company)
    if decision:
        logger.info(
            "Abstained on query %r (%s)", req.query[:120], decision.detail
        )
        if req.stream:

            async def abstain_gen():
                yield {
                    "event": "chunks",
                    "data": json.dumps([c.model_dump(mode="json") for c in chunks]),
                }
                yield {"event": "abstain", "data": json.dumps({"reason": decision.reason})}
                yield {"event": "done", "data": ""}

            return EventSourceResponse(abstain_gen(), ping=15)

        return ChatResponse(
            answer=decision.reason,
            thinking="",
            chunks=chunks,
            abstained=True,
            abstain_reason=decision.reason,
        )

    if req.stream:

        async def event_gen():
            # First event: the retrieved sources so the frontend can render them
            # immediately while the LLM is still generating. model_dump(mode="json")
            # ensures date fields become ISO strings for safe json.dumps.
            yield {
                "event": "chunks",
                "data": json.dumps(
                    [c.model_dump(mode="json") for c in chunks]
                ),
            }
            async for ev in stream_chat(req.query, chunks, req.system_prompt):
                if await request.is_disconnected():
                    break
                yield ev

        # ping keeps the Cloudflare tunnel warm (idle connections get killed ~100s)
        return EventSourceResponse(event_gen(), ping=15)

    result = await chat_once(req.query, chunks, req.system_prompt)
    return ChatResponse(
        answer=result["answer"],
        thinking=result["thinking"],
        chunks=chunks,
    )
