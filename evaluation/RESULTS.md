# CorpCheck retrieval evaluation — results

Status: **complete**. Every number below is traceable to a run directory under
`evaluation/runs/`; each run directory carries the exact command in `cmd.txt`.

Report date: 2026-08-06. All numbers below come from runs in `evaluation/runs/`
executed on this machine against the live corpus; none are estimated, projected,
or carried over from an earlier corpus.

---

## 1. What is being measured

`evaluation/ir_eval.py` calls `corpcheck.retrieval.retrieve()` **in process** —
no HTTP server, no LLM. Only retrieval quality is measured, so a delta in the
numbers is attributable to retrieval and nothing else.

**Corpus (verified against `nlp-project-db-1` / `financial_rag` on 2026-08-06):**

| | |
| --- | --- |
| Filings | 1,662 |
| Chunks | 469,874 |
| Issuers | 50 tickers |
| Filing types | 10-K: 429, 10-Q: 1,233 |
| Amendments (`10-K/A`, `10-Q/A`) | **0** |
| Fiscal years | 2017–2026 |

Verification query:

```bash
docker exec nlp-project-db-1 psql -U postgres -d financial_rag -t -c "
SELECT 'filings', count(*) FROM filings
UNION ALL SELECT 'chunks', count(*) FROM chunks
UNION ALL SELECT 'amendments', count(*) FROM filings WHERE filing_type LIKE '%/A%';"
```

**Benchmark:** `evaluation/datasets/financebench_filtered.json` — 35 FinanceBench
questions, 44 gold evidence spans, over the 10 issuers present in the corpus
(AMD, Amazon, CVS Health, Costco, JPMorgan, Johnson & Johnson, Microsoft, Nike,
Pfizer, Walmart). 28 questions target a 10-K, 7 a 10-Q. Doc periods 2018–2023.

**How a "hit" is defined (weak supervision).** FinanceBench supplies a gold
evidence *span* (a page or a table), not a chunk id. Chunk boundaries differ, so
a retrieved chunk counts as covering a gold span when it contains at least
`threshold` of that span's content-bearing tokens after normalisation,
boilerplate stripping and stopword removal. This is a proxy. It is biased in a
knowable direction (it can credit vocabulary overlap without the fact, and miss
a paraphrase), but the bias is held constant across configurations, which is
what makes A/B comparison meaningful even though the absolute level is soft.

**Provenance gate (`doc_gate`).** A chunk's overlap counts only if the chunk
came from the filing the question is about (issuer + filing type + fiscal year,
+ quarter for 10-Q). The right sentence from the wrong fiscal year is exactly
the failure mode this project exists to prevent, so it is scored as a miss.
`--no-doc-gate` reports the same runs without it.

**No gold metadata reaches the retriever.** Only the question text is passed —
never the benchmark's `company` / `doc_period` / `doc_type`. Passing those as
filters would benchmark a system that does not exist at serving time.

### Why two thresholds are reported

`0.5` is the project's primary protocol and the stricter reading. `0.2` is the
looser reading, and is reported because that is where several public RAG
retrieval numbers effectively sit. **Both are reported for every configuration**
so no number can be quoted without its protocol.

---

## 2. Metric ceiling (oracle retrieval)

Computed by `evaluation/oracle.py`: for each gold span, the best token overlap
achievable by *any* chunk of the correct filing. This is the upper bound
`ir_eval` can report; anything above it is a scoring bug.

Run: `evaluation/runs/R4_current/ceiling.json` (2026-08-06, recomputed after the
final ablation set; byte-identical to the earlier
`2026-08-06T14_G_current/ceiling.json`)

| threshold | spans reachable | ceiling Recall |
| --- | --- | --- |
| 0.20 | 40 / 44 | 0.9091 |
| 0.30 | 40 / 44 | 0.9091 |
| 0.40 | 37 / 44 | 0.8409 |
| **0.50** | 35 / 44 | **0.7955** |
| 0.60 | 25 / 44 | 0.5682 |
| 0.70 | 24 / 44 | 0.5455 |

Best achievable overlap: mean 0.696, median 0.745, min 0.046, max 1.000.
Zero gold documents are absent from the corpus and zero are unresolvable — so a
low measured Recall is a retrieval failure, not a missing-document artefact.

Reproduce:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.oracle \
  --output evaluation/runs/<label>/ceiling.json
```

---

## 3. How "before" and "after" are produced

A "before" number is only trustworthy if it was measured against the **same
corpus** as the "after" number. Checking out an old commit does not give that:
the corpus grew between 2026-08-03 and 2026-08-06 (commit `523fc4c` backfilled
10-Q chunks), so an old-commit run would differ by both code *and* data.

`evaluation/ablate.py` instead disables one query-understanding feature at a
time **in the live tree**, keeping the database, the chunking, the embedding
model and everything downstream byte-identical. Four cumulative configurations:

| run | year notation | company shorthand | company scope | meaning |
| --- | --- | --- | --- | --- |
| `R1_baseline` | off | off | off | pre-`5422531` query understanding |
| `R2_yearfix` | **on** | off | off | + FY22 / Q2 2023 / two-digit year parsing |
| `R3_shorthand` | on | **on** | off | + `JnJ`→JNJ, `Costco`→COST aliases |
| `R4_current` | on | on | **on** | + detected company hard-scopes the pool (HEAD + working tree) |

All four were executed 2026-08-06 (see `evaluation/runs/<label>/cmd.txt` for the
exact command and wall time). Every one reproduced the earlier same-day runs
`2026-08-06T14_D..G` **bit for bit**, which is the evidence that the harness is
deterministic.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.ablate \
  --off year-notation company-shorthand company-scope --label R1_baseline
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.ablate \
  --off company-shorthand company-scope --label R2_yearfix
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.ablate \
  --off company-scope --label R3_shorthand
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.ir_eval --label R4_current
```

Fixed for every run: `k=10`, `alpha=0.7`, fusion `rrf`, dataset
`evaluation/datasets/financebench_filtered.json`, 35 queries, 0 errors.

---

## 4. Results

### 4.1 Primary protocol — threshold 0.5, `clean` variant, `doc_gate` ON, k=10

This is the number the project stands behind. Ceiling here is **0.7955**.

| run | R@1 | R@3 | R@5 | R@10 | MRR@10 | zero-evidence queries |
| --- | --- | --- | --- | --- | --- | --- |
| R1_baseline | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 19 / 35 |
| R2_yearfix | 0.0571 | 0.0571 | 0.0857 | **0.1143** | 0.0657 | 7 / 35 |
| R3_shorthand | 0.0571 | 0.0571 | 0.0857 | **0.1429** | 0.0689 | 5 / 35 |
| **R4_current** | 0.0571 | 0.0571 | 0.0857 | **0.1429** | 0.0689 | **4 / 35** |

### 4.2 Loose protocol — threshold 0.2, `clean` variant, `doc_gate` ON, k=10

Ceiling here is **0.9091**.

| run | R@1 | R@3 | R@5 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| R1_baseline | 0.0286 | 0.0571 | 0.0571 | **0.0857** | 0.0464 |
| R2_yearfix | 0.0857 | 0.2000 | 0.2714 | **0.3714** | 0.1686 |
| R3_shorthand | 0.1143 | 0.2714 | 0.3571 | **0.4429** | 0.2210 |
| **R4_current** | 0.1143 | 0.2714 | 0.3571 | **0.4714** | 0.2257 |

### 4.3 The other three cells of the protocol cube

Reported so nobody can accuse the table of cherry-picking a favourable cell.
`raw` scores overlap against the un-normalised span text; `doc_gate OFF` credits
a content match from the *wrong filing*, which is precisely the failure this
project exists to catch — those numbers are diagnostics, **not** system results.

| run | protocol | R@1 | R@3 | R@5 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| R1_baseline | 0.2 / raw / gate ON | 0.0571 | 0.0857 | 0.0857 | 0.1429 | 0.0782 |
| R2_yearfix | 0.2 / raw / gate ON | 0.1143 | 0.2286 | 0.3143 | 0.5000 | 0.2154 |
| R3_shorthand | 0.2 / raw / gate ON | 0.1571 | 0.3143 | 0.4143 | 0.5857 | 0.2868 |
| R4_current | 0.2 / raw / gate ON | 0.1571 | 0.3714 | 0.4429 | 0.6143 | 0.3083 |
| R1_baseline | 0.2 / clean / gate OFF | 0.2905 | 0.3762 | 0.4048 | 0.4762 | 0.3737 |
| R2_yearfix | 0.2 / clean / gate OFF | 0.2333 | 0.3905 | 0.4476 | 0.5762 | 0.3674 |
| R3_shorthand | 0.2 / clean / gate OFF | 0.2619 | 0.4190 | 0.4905 | 0.5762 | 0.3919 |
| R4_current | 0.2 / clean / gate OFF | 0.2619 | 0.4190 | 0.4619 | 0.5762 | 0.3910 |
| R1_baseline | 0.2 / raw / gate OFF | 0.4429 | 0.5429 | 0.5714 | 0.6286 | 0.5225 |
| R2_yearfix | 0.2 / raw / gate OFF | 0.3286 | 0.5143 | 0.5857 | 0.7000 | 0.4625 |
| R3_shorthand | 0.2 / raw / gate OFF | 0.3714 | 0.5571 | 0.6143 | 0.7286 | 0.5006 |
| R4_current | 0.2 / raw / gate OFF | 0.4000 | 0.5857 | 0.6143 | 0.7286 | 0.5282 |
| R1_baseline | 0.5 / clean / gate OFF | 0.0857 | 0.1143 | 0.1143 | 0.1143 | 0.0952 |
| R2_yearfix | 0.5 / clean / gate OFF | 0.0857 | 0.0857 | 0.0857 | 0.1143 | 0.0886 |
| R3_shorthand | 0.5 / clean / gate OFF | 0.0857 | 0.0857 | 0.0857 | 0.1429 | 0.0917 |
| R4_current | 0.5 / clean / gate OFF | 0.0857 | 0.0857 | 0.0857 | 0.1429 | 0.0917 |
| R1_baseline | 0.5 / raw / gate ON | 0.0000 | 0.0286 | 0.0286 | 0.0286 | 0.0143 |
| R2_yearfix | 0.5 / raw / gate ON | 0.0857 | 0.1143 | 0.1429 | 0.1714 | 0.1086 |
| R3_shorthand | 0.5 / raw / gate ON | 0.0857 | 0.1143 | 0.1429 | 0.2000 | 0.1117 |
| R4_current | 0.5 / raw / gate ON | 0.0857 | 0.1143 | 0.1429 | 0.2000 | 0.1117 |
| R1_baseline | 0.5 / raw / gate OFF | 0.1524 | 0.2095 | 0.2095 | 0.2381 | 0.1993 |
| R2_yearfix | 0.5 / raw / gate OFF | 0.1810 | 0.2095 | 0.2095 | 0.2667 | 0.2215 |
| R3_shorthand | 0.5 / raw / gate OFF | 0.1810 | 0.2095 | 0.2095 | 0.2667 | 0.2215 |
| R4_current | 0.5 / raw / gate OFF | 0.1810 | 0.2095 | 0.2095 | 0.2667 | 0.2215 |

Full 6-threshold sweep for each run is in `evaluation/runs/<label>/summary.json`
(`grid`); per-query traces in `queries.jsonl`.

### 4.4 Gate diagnostics — questions answered with zero in-scope evidence

`mean_chunks_passing_gate` is the mean number of the k=10 returned chunks that
actually came from the filing the question is about.

| run | mean chunks passing gate | queries with **zero** passing | unresolvable gold doc |
| --- | --- | --- | --- |
| R1_baseline | 0.71 / 10 | 19 / 35 (54%) | 0 |
| R2_yearfix | 4.26 / 10 | 7 / 35 (20%) | 0 |
| R3_shorthand | 5.17 / 10 | 5 / 35 (14%) | 0 |
| **R4_current** | **5.40 / 10** | **4 / 35 (11%)** | 0 |

This is the most defensible result in the report and it is threshold-independent
— it does not depend on the weak-supervision overlap proxy at all. Baseline
returned ten chunks and *none of them were from the right filing* for 19 of 35
questions. The current system does that for 4.

### 4.5 Per-feature attribution

| feature | Δ R@10 @0.5 | Δ R@10 @0.2 | Δ zero-evidence queries |
| --- | --- | --- | --- |
| year notation (R1→R2) | +0.1143 (+4 q) | +0.2857 (+10 q) | −12 |
| company shorthand (R2→R3) | +0.0286 (+1 q) | +0.0715 (+2.5 q) | −2 |
| company scope (R3→R4) | **0.0000** | +0.0285 (+1 q) | −1 |

**Year notation is the whole story.** Company shorthand is a small real gain.
**Company scope shows no improvement at the primary threshold** and moves
exactly one query at 0.2 — that is one question out of 35, well inside noise.
It is kept because it is mechanically sound (restricting the candidate pool to
the detected issuer cannot introduce cross-issuer confusion) and because it
removes one more zero-evidence query, **not** because the benchmark endorses it.
Do not claim company scope improved Recall.

---

## 5. Where "Recall@10 0.07 → 0.34" comes from, exactly

This number appears in earlier drafts. Its exact provenance:

| | |
| --- | --- |
| before | `evaluation/runs/baseline/summary.json` — 2026-08-03 21:41 |
| after | `evaluation/runs/year_fix/summary.json` — 2026-08-03 21:54 |
| threshold | **0.2** |
| variant | **clean** |
| doc_gate | **ON** |
| k | **10** (alpha 0.7, fusion rrf) |
| metric | Recall@10, macro over 35 queries |
| values | **0.0714 → 0.3429** |
| variable changed | year-notation parsing only (`5422531`) |

**Two caveats that must travel with it:**

1. **It is threshold 0.2, not the project's primary 0.5.** At threshold 0.5,
   same variant/gate/k, the same pair is **0.0000 → 0.1143**.
2. **Those two runs predate the corpus.** They were executed 2026-08-03, before
   commit `523fc4c` (2026-08-04) backfilled mislabelled 10-Q chunks. They are
   therefore not comparable to anything in §4 and should not be quoted as
   current. Re-measured on today's corpus, the same one-variable ablation is
   **0.0857 → 0.3714** (`R1_baseline` → `R2_yearfix`), and the *full* current
   system reaches **0.4714**.

**Use the current-corpus numbers.** The 0.07 → 0.34 pair is superseded.

### Statistical honesty

n = 35. One query is worth 2.86 recall points, so no delta smaller than that
exists. 95% Wilson intervals on the headline values:

| value | 95% CI |
| --- | --- |
| 0.0000 (baseline, 0.5) | 0.000 – 0.099 |
| 0.1429 (current, 0.5) | 0.063 – 0.294 |
| 0.0857 (baseline, 0.2) | 0.030 – 0.224 |
| 0.4714 (current, 0.2) | 0.317 – 0.631 |

The 0.2-threshold improvement (0.0857 → 0.4714) has non-overlapping intervals.
The 0.5-threshold improvement (0.0000 → 0.1429) does not overlap either, but
both endpoints are small enough that the *ratio* is not meaningful — quote the
absolute pair, never "2× / 5× better".

---

## 6. What can and cannot go on a résumé

### Safe to write

- *"Cut retrieval-provenance failures from 19/35 to 4/35 FinanceBench questions
  (share of questions where none of the top-10 chunks came from the filing the
  question was about), by fixing fiscal-year and issuer parsing in the query
  layer."*
  Threshold-free, proxy-free, directly measured, largest effect in the report.
- *"Recall@10 0.09 → 0.47 on 35 FinanceBench questions (weak-supervision
  protocol: 0.2 token-overlap, provenance gate on, k=10; oracle ceiling 0.91),
  measured by single-feature ablation on a fixed 469,874-chunk corpus."*
  Every qualifier must ship with the number.
- *"Recall@10 0.00 → 0.14 under the strict 0.5-overlap protocol (ceiling 0.79)."*
  Weaker-sounding and therefore more credible; pair it with the gate number.
- *"Built a deterministic in-process IR harness with an oracle ceiling and a
  single-feature ablation runner, so every reported delta is attributable to one
  code change against an unchanged corpus."*
  This is the actual engineering claim and it is fully supported.

### NOT safe to write

- ❌ **"Recall@10 0.07 → 0.34."** Stale corpus (pre-`523fc4c`). Superseded by
  0.0857 → 0.3714. See §5.
- ❌ Any number without its threshold, variant, doc_gate state and k. The same
  system reads anywhere from 0.14 to 0.73 Recall@10 depending on the cell.
- ❌ The `doc_gate OFF` numbers (up to 0.7286) as system performance. They score
  a right-looking chunk from the wrong filing as a hit.
- ❌ **"Company-scoped retrieval improved Recall."** It did not, at the primary
  threshold. See §4.5.
- ❌ Any ratio/multiple framing ("5× recall"). Denominators of 0.00–0.09 on
  n=35 make ratios meaningless.
- ❌ Anything about version/amendment governance being *validated*. See §7.
- ❌ Framing 35 questions as a benchmark result. It is a filtered slice of
  FinanceBench restricted to the 10 issuers in the corpus, not FinanceBench.

---

## 7. Known limitations

1. **Version governance has never fired in the live corpus.** The corpus contains
   **0 of 1,662** filings that are `10-K/A` or `10-Q/A` amendments. The
   section-aware filtering logic has unit coverage plus a repository fixture
   based on GameStop's real March 2024 10-K/10-K/A metadata; that pair proves why
   unchanged original sections must remain available when an amendment changes
   only Item 5. It has still **never been triggered by an amended document loaded
   into this corpus**. Do not claim live-corpus validation.

2. **The relevance signal is weak supervision, not gold labels.** FinanceBench
   gives an evidence *page/table*, not a chunk id. A "hit" is a token-overlap
   proxy. It can credit vocabulary overlap without the fact, and it can miss a
   correct paraphrase. Absolute levels are soft. The bias is held constant
   across configurations, which is what makes the A/B deltas usable and the
   absolute numbers not.

3. **n = 35.** One query = 2.86 recall points. Confidence intervals are wide
   (§5). The company-scope result in particular is a one-query difference.

4. **The ceiling is far from reached.** At threshold 0.5 the oracle allows
   0.7955 and the system delivers 0.1429 — 18% of what is reachable. At 0.2:
   0.4714 of 0.9091, 52%. The retriever, not the corpus, is the bottleneck; no
   gold document is missing and none is unresolvable.

5. **Ablation ≠ git history.** `ablate.py` re-creates the *old behaviour* of one
   feature in the current tree. It reproduces the pre-fix regexes and flags
   faithfully, but it is a reconstruction, not a checkout. The tradeoff was
   deliberate: it buys an identical corpus on both sides, which a checkout
   cannot.

6. **This report measures retrieval only.** A separate deterministic answer
   scorer now exists for answer correctness, citation validity, supporting-chunk
   provenance, and abstention, but no recorded 35-question generation run is
   reported here yet. Semantic citation entailment also remains unmeasured.

7. **Single-issuer, single-period questions only.** The filtered set covers 10
   issuers over 2018–2023. No cross-company comparison, no cross-year trend
   questions, no questions requiring two filings.

8. **No hyperparameter tuning was performed on these 35 questions.** `k=10`,
   `alpha=0.7` and the RRF fusion strategy were fixed before these runs and held
   constant across all four configurations. Nothing in §4 is a tuned result. The
   corollary is that nothing in §4 is a *tuned-optimal* result either.

---

## 8. R5 table-row child experiment (2026-08-09)

Failure analysis attributed 19 of 30 reachable strict misses to table
candidate/context failures. R5 tested an evaluation-only dense arm over 130,084
row children derived from 29,570 table parents. Each child inherited the table
title and available header/year rows; retrieval still returned the original
parent chunk so citations and the overlap protocol remained comparable.

| metric | feature off | feature on |
| --- | ---: | ---: |
| strict 0.5 clean gated Recall@10 | 0.1429 | 0.1429 |
| loose 0.2 clean gated Recall@10 | 0.4714 | 0.5143 |
| zero-in-scope-evidence queries | 4/35 | 4/35 |
| median latency | 169 ms | 1,351 ms |
| p95 latency | 461 ms | 2,374 ms |

The experiment failed its predeclared acceptance criteria: strict Recall did not
reach 0.2500, no strict miss became a hit, and p95 latency increased by roughly
5.15x instead of staying within +50%. Two questions became new loose-threshold
hits and none were lost, so the representation has some signal, but the arm is
**rejected as a production default** and remains behind a default-off flag.

Artifacts: `evaluation/runs/R5_table_child_off_postcode/` and
`evaluation/runs/R5_table_child_on/`. The complete decision rule and failure
breakdown are in `evaluation/NEXT_RETRIEVAL_EXPERIMENT.md`.

---

## 9. Metadata coverage abstention seed (2026-08-09)

The cosine gate was strong enough to allow two requests whose retrieved text was
semantically similar but whose metadata made an answer impossible: Costco
FY2099 and OpenAI's FY2023 SEC annual filing. A deterministic pre-generation
coverage check now rejects a definite issuer or fiscal-year miss before falling
through to the unchanged cosine thresholds.

The no-filter retrieval-only run used nine hand-selected records: seven
answerable FinanceBench questions and the two controls above.

| metric | cosine only | metadata + cosine |
| --- | ---: | ---: |
| correct answerability decisions | 7/9 | 9/9 |
| answerable questions allowed | 7/7 | 7/7 |
| should-abstain controls refused | 0/2 | 2/2 |

The refusal statuses are auditable: Costco FY2099 is `year_mismatch`; OpenAI is
`unknown_company`. Retrieval ranking and the cosine thresholds were unchanged.

This is **not an answer-generation result**. Retrieval-only mode deliberately
records empty answers when a question is allowed, and no `SGLANG_BASE_URL` was
configured for this run. The sample is also tiny and hand-selected, so 9/9 is a
regression result for these cases, not a broad abstention-accuracy claim.

Artifacts: `evaluation/datasets/answer_eval_seed.json`,
`evaluation/runs/R5_answer_seed/metadata-gate.jsonl`, and
`evaluation/runs/R5_answer_seed/metadata-gate-summary.json`. Design and risk
analysis: `evaluation/NEXT_ABSTAIN_EXPERIMENT.md`.
