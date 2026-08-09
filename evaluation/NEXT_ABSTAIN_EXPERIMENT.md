# Next abstain experiment: deterministic metadata coverage before generation

## Outcome (2026-08-09)

Implemented and retained. The production pre-generation gate now checks
deterministic issuer and explicit-year coverage before delegating to the
unchanged cosine gate. API, MCP, and the answer-prediction collector use the
same decision path and expose stable diagnostic statuses.

On the nine-record no-filter seed, correct answerability decisions moved from
7/9 to 9/9. All seven answerable records remained allowed; Costco FY2099 was
refused as `year_mismatch`, and OpenAI's FY2023 SEC-filing request was refused as
`unknown_company`. Retrieval-only mode was used because `SGLANG_BASE_URL` was
unset, so this result says nothing about generated-answer or citation quality.

The seed is deliberately a regression test, not a representative benchmark.
The gate remains conservative: unresolved-company detection requires explicit
SEC-filing intent and a possessive proper name, and year coverage accepts an
explicit requested year found in a retrieved passage even when the filing's own
fiscal-year label differs (for example, a comparative column).

## Scope

This note diagnoses the two no-filter false allows in
`evaluation/datasets/answer_eval_seed.json`. It proposes one narrow experiment;
it does not change retrieval, ranking, or the calibrated cosine thresholds.

The invariant to test is:

> A high semantic similarity is not sufficient when every retrieved passage
> contradicts an explicit company or fiscal-year constraint in the question.

The check belongs after retrieval and before both `evaluate_confidence()` and
the LLM call. It should inspect query constraints against `ChunkResult.company`
and `ChunkResult.fiscal_year`, return a normal `AbstainDecision` on a definite
coverage miss, and otherwise defer unchanged to the existing cosine gate.

## Reproduction against the live corpus (2026-08-09)

Both queries were run through `retrieve(k=10, alpha=0.7)` without seed filters,
after `load_known_tickers()` populated the query-parser caches.

### `abstain_future_costco_2099`

Question: `What were Costco's total assets at the end of FY2099?`

- `detect_company_in_query()` returns `COST`.
- `detect_years_in_query()` returns `["2099"]`.
- `detect_filing_type_in_query()` returns `None`; the weaker annual hint returns
  `10-K`.
- Live corpus metadata has 4,115 Costco chunks and fiscal years 2017 through
  2026; it has no Costco FY2099 chunk.
- Company scoping works: all ten returned chunks are `COST`.
- The retrieved fiscal years are 2023, 2024, 2020, 2021, 2020, 2024, 2025,
  2022, 2021, and 2025. None is 2099.
- The cosine gate nevertheless allows generation: top-1 `0.5724`, mean top-3
  `0.5715`, both above the configured `0.42` / `0.40` floors.

This is a solvable metadata mismatch. The parser already knows both requested
constraints. The failure is that a detected year is only a retrieval boost;
no post-retrieval check verifies that the resulting evidence covers it.

### `abstain_private_openai_2023`

Question: `According to its SEC annual filing, what was OpenAI's net income in FY2023?`

- `detect_company_in_query()` returns `None`.
- `detect_years_in_query()` returns `["2023"]`.
- Both explicit filing-type detection and the filing-type hint return `10-K`.
- The live corpus has 50 distinct tickers and 469,874 searchable chunks. No
  ticker or company name contains `OpenAI`.
- With no detected company, retrieval is unscoped. The ten returned chunks span
  AMD, GOOGL, CVX, DG, META, COP, and BAC; all are FY2023 10-K passages.
- The year and filing type therefore look perfectly covered even though the
  issuer is wrong. The cosine gate allows generation: top-1 `0.5708`, mean
  top-3 `0.5350`.

This is not solvable by checking currently parsed metadata. `None` currently
means both “the question names no issuer” and “the question names an issuer that
is absent from the corpus.” Unknown-entity detection must distinguish those
states before a company-coverage gate can reject this query.

## Smallest proposed gate

Add one deterministic pre-generation coverage decision with two deliberately
conservative rules.

1. **Known-company mismatch:** when the existing parser resolves a ticker,
   require at least one returned chunk whose `company` equals that ticker.
   This also makes the current unscoped fallback safe when scoped retrieval
   returns nothing.
2. **Year mismatch:** when the query contains one or more explicit years,
   require at least one returned chunk whose `fiscal_year` is in that set.
   Use intersection, not “every requested year,” because one annual filing often
   contains comparative values for prior years. Costco FY2099 then fails, while
   a FY2023-versus-FY2022 question may legitimately be supported by a FY2023
   filing.

To cover the OpenAI control without introducing general-purpose NER, extend the
query parse result with a narrow third state: `company_mentioned_unresolved`.
The first experiment should set it only when all of the following are true:

- the normal ticker/name/alias resolver found no company;
- the question contains explicit SEC-filing intent (`SEC`, `10-K`, `10-Q`, or
  `annual filing` / `quarterly filing`); and
- a possessive proper-name candidate such as `OpenAI's` or `OpenAI’s` is present
  and is not a generic stopword (`company`, `issuer`, `management`, `filing`).

When that state is true, abstain before generation with a reason that the named
issuer is not represented in the indexed filing corpus. Do not infer an unknown
entity from arbitrary capitalised words or from a bare unresolved token.

Conceptually:

```python
coverage = parse_coverage_constraints(query)
companies = {chunk.company for chunk in chunks}
years = {chunk.fiscal_year for chunk in chunks if chunk.fiscal_year is not None}

if coverage.company_mentioned_unresolved:
    return abstain("The named issuer is not represented in the indexed filings.")
if coverage.company and coverage.company not in companies:
    return abstain("No retrieved filing passage matches the requested company.")
if coverage.years and not (set(coverage.years) & years):
    return abstain("No retrieved filing passage matches the requested fiscal year.")
return evaluate_confidence(chunks)
```

Filing type should remain out of the first hard gate. The current parser
explicitly distinguishes exact filing-type language from weaker annual/quarter
hints, and hardening inferred hints at the same time would expand the experiment
beyond the two observed failures.

## False-positive risks and containment

- **Comparative questions:** requiring all mentioned years would incorrectly
  reject questions answered by comparative columns in one filing. Intersection
  coverage avoids that known failure mode.
- **Period-label mismatch:** `fiscal_year` is filing metadata, not proof that a
  chunk contains every comparative period requested. The proposed rule only
  catches a total miss; it must not claim full answer support.
- **Unknown-company over-detection:** possessives can be ordinary language
  (`management's`) or titles (`Company's`). SEC-intent conjunction plus a small
  generic stoplist keeps the experiment narrow. Ambiguous candidates must defer
  to the cosine gate rather than reject.
- **Alias gaps:** a real indexed issuer may be phrased with an unregistered
  nickname. The existing resolver remains authoritative; unresolved rejection
  is limited to the strict SEC-intent + possessive pattern, and acceptance tests
  must include known full names, tickers, and aliases.
- **Empty results:** preserve the existing `No matching passages were retrieved.`
  behavior; the coverage gate need not replace it.
- **Caller-supplied filters:** explicit API/MCP filters should remain
  authoritative. The first implementation should define precedence and test a
  query/filter conflict rather than silently combining them.

## Tests and acceptance criteria

Unit tests:

1. `Costco FY2099` parses to known company `COST`, year `2099`, and abstains on
   chunks from COST FY2020-2026 even when cosine scores exceed both thresholds.
2. `Costco FY2021` with at least one COST FY2021 chunk reaches the cosine gate.
3. A known-company result set containing only another ticker abstains before the
   cosine gate.
4. `OpenAI's ... SEC annual filing ... FY2023` produces the unresolved-company
   state and abstains before the cosine gate.
5. `What was the company's FY2023 net income?` does not produce an unresolved
   company.
6. `What was management's FY2023 outlook?` does not produce an unresolved
   company.
7. Known possessive forms (`Costco's`, `Amazon's`, and a full registered company
   name) resolve normally and are not marked unresolved.
8. A FY2023-versus-FY2022 query with only a matching FY2023 filing reaches the
   cosine gate; a result set matching neither year abstains.
9. No-year and no-company questions preserve current behavior exactly.
10. Empty retrieval preserves the existing empty-result refusal reason.

Evaluation acceptance:

- Both no-filter controls change from false allow to correct abstention.
- All currently answerable records in `answer_eval_seed.json` remain allowed by
  the coverage gate (their final answer correctness is scored separately).
- Run the full FinanceBench retrieval set and require zero new refusals caused
  solely by the coverage gate for records whose gold evidence is present in the
  returned chunks.
- Log the parsed constraints, observed company/year sets, and deterministic
  refusal reason so every new abstention is auditable.
- Do not retune cosine thresholds during this experiment; success must be
  attributable to metadata coverage rather than threshold drift.
