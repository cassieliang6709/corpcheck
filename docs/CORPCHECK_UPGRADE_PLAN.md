# CorpCheck portfolio upgrade plan

Status: active on `agent/corpcheck-evaluation-hardening`
Last reviewed: 2026-08-17

## Product thesis

CorpCheck should be presented as evidence infrastructure for SEC research, not
as a generic financial chatbot. Its strongest proof is the system around the
model: source-preserving ingestion, filing-aware hybrid retrieval, amendment
governance, deterministic refusal, and evaluation that exercises the same
retrieval function used in serving.

The portfolio surface must let an interviewer move through three depths without
changing the story:

1. **30 seconds:** understand the problem and run one supported and one refused
   question.
2. **5 minutes:** follow the end-to-end technical route and inspect the gate,
   provenance, and measured limitations.
3. **30 minutes:** open the pinned source snapshot, read the implementation line
   by line, and reproduce the relevant tests or evaluation run.

## What the hardening branch already accomplished

Relative to `main`, the branch is not a speculative rewrite. It adds the
evidence needed to decide what is safe to show:

- A retrieval-only `/answerability` product surface and bilingual interview
  demo.
- A frozen FinanceBench slice, deterministic answer scoring, and a recorded
  34-question baseline.
- Paired retrieval acceptance gates and fail-closed corpus reprocessing.
- Table-representation experiments that are rejected when they miss their
  gates rather than being marketed as improvements.
- Section-aware amendment governance, including isolated validation against a
  real 10-K / 10-K/A pair.
- CI and reproducible local Postgres setup.

This work establishes the current boundary: retrieval and evidence inspection
are demonstrable; generated financial answers are not yet reliable enough to
ship as the headline product.

## Upgrade sequence

### Phase 1 — Public technical surface

Goal: make the landing page useful to both a recruiter and an engineer.

- Keep the evidence console above the fold.
- Show the real route: EDGAR → representation → embedding/indexing → query
  understanding → hybrid retrieval → revision filtering → abstain gate →
  HTTP/MCP/evaluation.
- Pin code links to an immutable commit so claims can be checked line by line.
- Separate public base models, CorpCheck-owned engineering, optional generation,
  and future trained artifacts.
- Preserve complete Chinese `/` and English `/en/` routes.

Acceptance checks:

- Both routes expose the same six-stage architecture and code-reading path.
- Desktop and mobile layouts have no horizontal overflow.
- Recorded-mode presets still render pass and refusal states.
- Every number shown on the page is traceable to `evaluation/RESULTS.md` or the
  recorded `/answerability` response.

### Phase 2 — Hugging Face public release

Goal: give interviewers a stable public artifact they can inspect without a
local database.

Publish these as separate assets:

1. **Space — `cassieliang6709/corpcheck`**
   - Public retrieval-only experience.
   - Recorded, date-stamped evidence states by default.
   - Optional live endpoint only after rate limits, CORS, health checks, and
     cost controls exist.
   - Links to the exact GitHub commit and evaluation protocol.
2. **Dataset — `cassieliang6709/corpcheck-eval`**
   - Project-authored split manifest, exclusions, evaluation schemas, and run
     summaries that are legally safe to redistribute.
   - Do not republish SEC or FinanceBench content unless its license and source
     terms permit redistribution; store identifiers and reproducible builders
     when in doubt.
3. **Model repo — only after CorpCheck trains weights**
   - A reranker or embedding adapter with training config, base-model license,
     held-out metrics, limitations, and reproducible inference code.
   - Do not create a model repo that merely re-uploads
     `sentence-transformers/all-MiniLM-L6-v2` or labels Qwen as CorpCheck-owned.

The local machine is not authenticated with Hugging Face as of 2026-08-17, so
the external publish step remains pending.

### Phase 3 — Train a defensible retrieval artifact

Goal: create weights that genuinely belong in a CorpCheck model repository.

The smallest defensible candidate is a filing-aware reranker, not a new large
language model.

- Train on query–evidence pairs with hard negatives from the wrong fiscal year,
  wrong issuer, wrong filing type, and superseded sections.
- Keep companies or filing periods disjoint between training and held-out
  evaluation.
- Compare against the current retrieval stack on the same frozen corpus.
- Require improvements in retrieval recall and provenance failures without
  degrading refusal behavior.
- Recalibrate the abstain thresholds after any embedding change.

Do not promote the model if the paired gate fails. A rejected experiment is an
engineering result, not a launch artifact.

### Code health gates before model training

The repository review found four issues that should be removed before changing
the embedding stack:

- `EMBEDDING_MODEL` is environment-driven in `src/corpcheck/settings.py` but
  hard-coded in `src/corpcheck/ingestion/config.py`. A model change can therefore
  make serving queries incompatible with stored corpus vectors. Consolidate the
  model id, dimension, and corpus manifest behind one validated configuration.
- Landing-page snapshot counts and recorded evidence are currently duplicated
  by hand in HTML/JavaScript. Generate a public, credential-free artifact from a
  versioned evaluation run so the site cannot drift from `RESULTS.md`.
- The official CI lint scope passes, but a repository-wide `ruff check` reports
  85 pre-existing issues in legacy ingestion and LLM files. Pay this down by
  owned subsystem rather than mixing it into retrieval experiments.
- The local `pytest` entry point needs `PYTHONPATH=.` to import `evaluation`, as
  CI already specifies. Encode that path in project test configuration so the
  documented local command and CI command behave the same way.

These are reproducibility risks, not visual polish. They should be resolved
before publishing CorpCheck-trained weights.

### Phase 4 — Interview package

Goal: demonstrate the product even when network or model services fail.

- A 60–90 second live path: supported question → evidence metrics → SEC source
  → out-of-domain refusal.
- A five-minute architecture path using the landing-page route and pinned code
  map.
- A local fallback bundle: video, screenshots, formatted API responses, and
  direct `curl` commands.
- A deeper discussion path: one failed retrieval experiment, why the gate
  rejected it, and what changed next.

The existing runbook in `docs/INTERVIEW_DEMO.md` remains the operational source
of truth.

## Final release gate

The upgraded portfolio story is ready when all of the following are true:

- Landing page is deployed and verified in both languages.
- Pinned code links resolve publicly.
- Hugging Face Space and evaluation artifact are public.
- Any model repository contains actual CorpCheck-trained weights and a complete
  model card.
- The interview demo can run in live and offline fallback modes.
- README, landing page, Hugging Face cards, and résumé use the same qualified
  metrics and product boundary.
