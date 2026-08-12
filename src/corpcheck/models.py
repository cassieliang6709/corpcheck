from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    sector: Optional[str] = None
    company: Optional[str] = None
    filing_type: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    # Unboosted dense cosine similarity, carried through fusion untouched.
    # ``score`` above is a fused, min-max-normalised, rerank-adjusted number and
    # is only meaningful *relative* to the other rows in the same response;
    # ``cos_sim`` is the one field on the row that survives on an absolute
    # scale, which is what makes confidence gating possible. None when the chunk
    # was found only by the sparse arm.
    cos_sim: Optional[float] = None
    company: str
    sector: Optional[str] = None
    filing_type: Optional[str] = None
    # Provenance: which filing period this chunk came from. Needed to verify that
    # evidence is attributable to the document the question is actually about.
    fiscal_year: Optional[int] = None
    period_label: Optional[str] = None
    filed_date: Optional[date] = None
    source_url: Optional[str] = None
    article_title: Optional[str] = None
    page_num: Optional[int] = None
    source_type: str = "sec"
    content_kind: Optional[str] = None
    chunk_strategy: Optional[str] = None
    display_title: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]


class AnswerabilityRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    company: Optional[str] = None
    filing_type: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)


class AnswerabilitySimilarity(BaseModel):
    top1_cos_sim: Optional[float]
    mean_top3_cos_sim: Optional[float]
    top1_min: float
    mean_top3_min: float


class AnswerabilityCoverage(BaseModel):
    retrieved: int
    with_dense_score: int
    sparse_only: int
    companies: list[str]
    filing_types: list[str]
    fiscal_years: list[int]
    source_types: list[str]


class AnswerabilityResponse(BaseModel):
    query: str
    answerable: bool
    gate_status: str
    reason: str
    llm_consulted: bool
    similarity: AnswerabilitySimilarity
    coverage: AnswerabilityCoverage
    chunks: list[ChunkResult]


class FilterOptionsResponse(BaseModel):
    companies: list[str]
    filing_types: list[str]
    fiscal_years: list[int]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    sector: Optional[str] = None
    company: Optional[str] = None
    filing_type: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    stream: bool = True
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None
    chunks: list[ChunkResult]
    # True when the retrieval-confidence gate declined before the LLM was
    # consulted. ``answer`` then holds the refusal text and ``chunks`` still
    # carries what was retrieved, so a caller can show the user the evidence
    # that was judged insufficient. Defaults keep existing clients working.
    abstained: bool = False
    abstain_reason: Optional[str] = None
