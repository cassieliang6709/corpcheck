"""LLM glue for the /chat endpoint.

Formats retrieved chunks into a citation-friendly context block and calls the
configured OpenAI-compatible SGLang endpoint (Qwen) either in streaming or
one-shot mode.

The client is lazy-initialized so the container can start successfully even
with SGLANG_BASE_URL empty — failure surfaces at request time as an HTTP 503.

中文：聊天层只格式化已检索证据并调用兼容 OpenAI 的端点；未配置模型时延迟到请求阶段报错。
"""

import asyncio
import json
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

from corpcheck import settings as config
from corpcheck.models import ChunkResult


DEFAULT_SYSTEM_PROMPT = """You are a grounded financial analyst answering questions from retrieved SEC filing chunks.

Your job is to give the best supported answer using ONLY the provided context.

RULES:
1. USE ONLY THE PROVIDED CHUNKS.
   - Do not use outside knowledge.
   - Do not infer facts that are not supported by the retrieved text.
2. ANSWER WHEN THE EVIDENCE IS SUFFICIENT.
   - If the answer is directly stated in the chunks, answer it.
   - If the question requires simple arithmetic or comparison from values in the chunks, perform it and answer.
   - If the context is partially relevant but still missing a required input, do not guess.
3. REFUSE ONLY WHEN NECESSARY.
   - Output exactly "I cannot answer this based on the provided documents." only when the required information is missing, contradictory, or not attributable to the provided chunks.
4. CITE EVERY SUPPORTED CLAIM.
   - Every factual statement, number, conclusion, and bullet item must include bracketed citations like [1] or [2][3].
   - Cite the chunk(s) that support the calculation inputs, not just the final conclusion.
5. KEEP THE ANSWER DIRECT.
   - No preamble, greetings, or filler.
   - Prefer 1-5 short paragraphs or a short bullet list.
   - If a calculation is requested, show only the final checked formula and result.
   - Do not include exploratory math, self-corrections, or multiple candidate answers in the final answer.
   - If you notice a mistake while reasoning, correct it internally and output only the corrected final answer.
6. TREAT METADATA AS IMPORTANT EVIDENCE.
   - Pay attention to company, filing type, title, and year in each chunk.
   - Prefer chunks that match the question's company, filing period, and filing type when answering.
   - If retrieved chunks conflict across companies or periods, answer only from the chunks that clearly match the question; otherwise refuse.
7. KEEP REASONING INTERNAL.
   - Use the <think> block only for concise internal planning.
   - Never mention that you are re-checking, correcting yourself, or reconsidering in the final answer.
   - The final answer must contain one consistent conclusion.
8. FOLLOW THIS DECISION ORDER.
   - First, identify the chunks that best match the question's company, filing type, and period.
   - Second, look for the exact value, statement, or table rows needed to answer.
   - Third, if the needed inputs are present across one or more matching chunks, answer using them even if light arithmetic or comparison is required.
   - Refuse only after this check fails.
9. FOR NUMERICAL QUESTIONS:
   - Use the plainly stated values in the retrieved chunks.
   - Prefer direct table values over narrative paraphrases when both are available.
   - Round only at the end, using the precision requested by the user.
   - If one required input is missing, refuse instead of estimating.
10. FOR QUALITATIVE QUESTIONS:
   - If the chunks provide enough direct evidence to support a yes/no answer or a short explanation, answer it.
   - Do not refuse just because the wording in the filing differs from the wording in the question.
11. OUTPUT FORMAT:
   - Start with `### Final Answer`.
   - Then provide exactly one concise answer block.
   - Do not include any text after the final cited answer.

EXAMPLES

Question: What was Tesla's FY2022 gross margin?
Context from SEC filings:
[1]
Title: Tesla, Inc. 10-K 2022 | Company: TSLA | Filing: 10-K
Text: "Total revenues were $10,000 million."

[2]
Title: Tesla, Inc. 10-K 2022 | Company: TSLA | Filing: 10-K
Text: "Cost of revenues was $7,500 million."
Response:
<think>
- Revenue is $10,000 million [1].
- Cost of revenues is $7,500 million [2].
- Gross margin = ($10,000 - $7,500) / $10,000 = 25%.
</think>
### Final Answer
Tesla's FY2022 gross margin was 25.0%, calculated as ($10,000M - $7,500M) / $10,000M [1][2].

Question: Did Adobe repurchase shares in FY2021?
Context from SEC filings:
[1]
Title: Adobe Inc. 10-K 2021 | Company: ADBE | Filing: 10-K
Text: "During fiscal 2021, the company repurchased 8 million shares of its common stock for $4.5 billion."
Response:
<think>
- The chunk directly states that shares were repurchased in fiscal 2021 [1].
</think>
### Final Answer
Yes. Adobe repurchased 8 million shares of common stock for $4.5 billion in FY2021 [1].

Question: What was Coca-Cola's FY2025 operating income?
Context from SEC filings:
[1]
Title: The Coca-Cola Company 10-K 2023 | Company: KO | Filing: 10-K
Text: "Operating income for fiscal 2023 was $12.0 billion."
Response:
<think>
- The question asks for FY2025, but the provided chunk only gives FY2023 [1].
- The required FY2025 value is missing.
</think>
### Final Answer
I cannot answer this based on the provided documents."""


_client: Optional[AsyncOpenAI] = None
_client_lock = asyncio.Lock()


async def get_llm_client() -> AsyncOpenAI:
    """Lazily create one shared client after validating the configured endpoint.

    中文：延迟初始化让不启用聊天的部署仍能启动；锁避免并发首请求创建多个客户端。
    """
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:  # re-check after acquiring the lock
                if not config.SGLANG_BASE_URL:
                    raise RuntimeError("SGLANG_BASE_URL not configured")
                _client = AsyncOpenAI(
                    base_url=config.SGLANG_BASE_URL,
                    # OpenAI SDK rejects empty string at construction — use placeholder
                    api_key=config.SGLANG_API_KEY or "EMPTY",
                    timeout=httpx.Timeout(
                        connect=10.0, read=120.0, write=10.0, pool=10.0
                    ),
                )
    return _client


def format_context(chunks: list[ChunkResult]) -> str:
    """Format chunks as numbered, source-labelled blocks for citation.

    中文：编号必须与模型提示词中的引用号一致，并保留公司、文件类型和日期等溯源信息。
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        source_label = c.source_type.upper()
        title = c.display_title or c.article_title or ""
        header_parts = [f"[{i}]", c.company, source_label]
        if c.filing_type:
            header_parts.append(c.filing_type)
        if title:
            header_parts.append(f"- {title}")
        header = " ".join(part for part in header_parts if part)
        if c.filed_date:
            header += f" (filed {c.filed_date.isoformat()})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(
    query: str, context: str, system_prompt: Optional[str] = None
) -> list[dict]:
    """Build one system message and one evidence-backed user message.

    中文：调用方可提供系统提示词；否则使用默认的“仅依据上下文回答”策略。
    """
    sp = system_prompt or DEFAULT_SYSTEM_PROMPT
    user = f"Context from retrieved company sources:\n\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": sp},
        {"role": "user", "content": user},
    ]


async def stream_chat(
    query: str,
    chunks: list[ChunkResult],
    system_prompt: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Async generator yielding SSE event dicts.

    Events shape: {"event": "thinking"|"answer"|"error"|"done", "data": json_str}.
    Caller wraps in sse_starlette.EventSourceResponse.

    中文：生成器将模型增量转为 SSE 事件；流开始后只能以内嵌 error 事件报告错误。
    """
    client = await get_llm_client()
    messages = build_messages(query, format_context(chunks), system_prompt)

    try:
        stream = await client.chat.completions.create(
            model=config.SGLANG_MODEL,
            messages=messages,
            max_tokens=config.SGLANG_MAX_TOKENS,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Qwen thinking mode surfaces reasoning on delta.reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {
                    "event": "thinking",
                    "data": json.dumps({"delta": reasoning}),
                }
            if delta.content:
                yield {
                    "event": "answer",
                    "data": json.dumps({"delta": delta.content}),
                }
    except asyncio.CancelledError:
        # Frontend disconnected — re-raise so FastAPI cleans up the response
        raise
    except Exception as e:
        # Once stream headers are flushed we can't change HTTP status,
        # so surface the error as an in-band event instead of raising.
        yield {
            "event": "error",
            "data": json.dumps({"message": str(e)}),
        }
    finally:
        yield {"event": "done", "data": "{}"}


async def chat_once(
    query: str,
    chunks: list[ChunkResult],
    system_prompt: Optional[str] = None,
) -> dict:
    """Run one non-streaming completion and return answer plus optional reasoning.

    中文：非流式路径保持与 SSE 路径相同的消息构造，只在边界处返回完整结果。
    """
    client = await get_llm_client()
    messages = build_messages(query, format_context(chunks), system_prompt)

    resp = await client.chat.completions.create(
        model=config.SGLANG_MODEL,
        messages=messages,
        max_tokens=config.SGLANG_MAX_TOKENS,
        stream=False,
    )
    msg = resp.choices[0].message
    return {
        "answer": msg.content or "",
        "thinking": getattr(msg, "reasoning_content", None),
    }
