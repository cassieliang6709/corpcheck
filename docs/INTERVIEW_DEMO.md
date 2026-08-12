# CorpCheck interview demo

This is a retrieval-only demo. Its purpose is to show whether CorpCheck finds
strong, traceable SEC evidence before any language model is allowed to answer.
Do not present it as a production answer-generation system yet.

## Five-minute preflight

Run these commands from the repository root. They do not print `.env` or any
credentials.

1. Start the already-populated local database if it is not running. If you use
   the repository's Docker setup:

   ```bash
   docker compose -f docker/docker-compose.yml up -d db
   docker compose -f docker/docker-compose.yml exec db \
     pg_isready -U postgres -d financial_rag
   ```

2. Start the API. Startup waits for the embedding model to load, so a successful
   health response also means the slow first-load work is finished.

   ```bash
   .venv/bin/uvicorn corpcheck.api.main:app --host 127.0.0.1 --port 8000
   ```

3. In a second terminal, verify the API:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   ```

   Expected response:

   ```json
   {"status":"ok"}
   ```

4. Serve the local landing page and open the Demo Console:

   ```bash
   python3 -m http.server 4173 --directory landing
   ```

   Open <http://127.0.0.1:4173/#demo-console>. The console should call
   `http://127.0.0.1:8000/answerability`; it never needs an API key.

5. Keep these direct API checks ready in a terminal:

   ```bash
   curl -fsS http://127.0.0.1:8000/answerability \
     -H 'Content-Type: application/json' \
     -d '{"query":"What was Apple'"'"'s total net sales in fiscal 2022?","k":5}' \
     | python3 -m json.tool

   curl -fsS http://127.0.0.1:8000/answerability \
     -H 'Content-Type: application/json' \
     -d '{"query":"What is the best recipe for sourdough bread?","k":5}' \
     | python3 -m json.tool
   ```

The response includes `answerable`, `gate_status`, the measured similarity and
its thresholds, evidence coverage, and retrieved chunks with filing provenance.
It always reports `llm_consulted: false`.

## 60–90 second script — English

**0:00–0:12 — Frame the problem**

> CorpCheck is not another financial chatbot. I built it around a narrower
> question: before generating an answer, can the system prove that the SEC
> filings contain strong enough evidence?

**0:12–0:38 — Show a supported question**

Select the Apple FY2022 preset and click **Check evidence**.

> This request runs retrieval and a deterministic two-threshold confidence
> gate. It passes. I can inspect the raw similarity, the threshold it cleared,
> and the filing metadata—not just a polished answer. The evidence card links
> back to the SEC source and identifies the company, filing type, fiscal year,
> and filing date.

**0:38–0:58 — Show refusal**

Select the sourdough preset and click **Check evidence**.

> Retrieval still returns candidates, but their similarity is below the floor,
> so CorpCheck refuses before contacting an LLM. That distinction matters: an
> empty result is not the same as evidence that is too weak to trust.

**0:58–1:18 — Be honest about the current boundary**

> The current local snapshot contains 1,662 filings and 469,874 chunks. On the
> 34-question answer benchmark, answerability was correct on 33, but final-answer
> correctness was only 1 out of 34. That exposed retrieval as the bottleneck, so
> generation remains disabled in this demo while I rebuild and gate the corpus.

**Optional close**

> The product decision is simple: when the evidence is weak, silence is a
> feature—not a failure state to hide.

## 60–90 秒脚本 — 中文

**0:00–0:12 — 先讲问题**

> CorpCheck 不是另一个财报聊天机器人。我更想先回答一个窄一点、但更重要的
> 问题：在生成答案之前，系统能不能证明 SEC 文件里真的有足够可靠的证据？

**0:12–0:38 — 展示可以回答的问题**

选择 Apple FY2022 预设问题，点击 **Check evidence**。

> 这一步只做检索和一个确定性的双阈值判断。现在它通过了。我能直接看到原始
> 相似度、它跨过的阈值，以及文件出处，而不是只看到一段很流畅的答案。证据卡
> 会标出公司、10-K、财年、提交日期，并链接回 SEC 原文。

**0:38–0:58 — 展示拒答**

选择 sourdough 预设问题，点击 **Check evidence**。

> 检索还是会返回候选内容，但它们的相似度低于门槛，所以 CorpCheck 会在调用
> LLM 之前拒绝。这里我想区分两件事：不是“什么也没搜到”，而是“搜到的东西
> 不够可靠，不能拿来回答”。

**0:58–1:18 — 诚实交代边界**

> 当前本地快照有 1,662 份 filing 和 469,874 个 chunk。34 道题的基线里，
> answerability 判断对了 33 道，但最终答案只对了 1 道。这让我确认瓶颈主要在
> retrieval，所以这个 Demo 暂时不开生成答案；我会先完成同源语料重建并通过
> retrieval gate。

**可选收尾**

> 对这个产品来说，证据不足时保持沉默，不是一个要藏起来的失败状态，而是功能
> 本身。

## Offline fallback checklist

Prepare these before the interview; none should contain `.env`, database URLs,
API keys, or contact headers.

- [ ] Record a 60–90 second local video at 1080p: landing page → Apple pass →
  sourdough refusal → provenance link. Keep the browser address bar visible.
- [ ] Save one screenshot of each result state and one screenshot of the current
  benchmark section. Label the benchmark as a baseline, not a shipped accuracy
  claim.
- [ ] Save the two formatted `/answerability` JSON responses locally so the
  measured gate and provenance can still be discussed if the UI fails.
- [ ] Download and warm the embedding model once before interview day. Only
  after that succeeds, rehearse an offline start with `HF_HUB_OFFLINE=1`.
- [ ] Verify `/health`, both presets, and the SEC links on the same machine and
  network you will use for the interview.
- [ ] Put the video, screenshots, and this runbook in one locally available
  folder; do not depend on cloud playback.
- [ ] Keep the direct `curl` commands above in a terminal as the first fallback.
- [ ] If the live result differs from the recorded number, narrate the value on
  screen. Do not silently reuse an older metric.

## Claims boundary

Safe to demonstrate now:

- Retrieval-only answerability checking through `POST /answerability`.
- A deterministic pre-LLM refusal and `llm_consulted: false`.
- Evidence text plus company, filing type, fiscal year, filing date, and source
  URL when those fields are present on the retrieved chunk.
- The validated corpus snapshot: 1,662 filings and 469,874 chunks.
- The recorded baseline: 33/34 answerability decisions and 1/34 final answers.

Do not claim yet:

- That Qwen or SGLang powers the live demo; the generation backend is optional
  and currently not configured for this demo.
- That final answers are reliable or that citation presence proves citation
  support.
- That the new representation experiment passed its retrieval gates.
- That the public website is a production application.
