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
    company: str
    sector: Optional[str] = None
    filing_type: Optional[str] = None
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
