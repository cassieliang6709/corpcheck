# Claim verification demo

`POST /claims/verify` takes a sentence of news or memo text, splits it into
atomic claims, retrieves SEC evidence through the same `retrieve()` the serving
path uses, and returns a verdict with the evidence it was bound to. No LLM is
involved in the decision.

This page records an adversarial run against the live local corpus (1,662
filings / 469,874 chunks / 50 issuers) on 2026-08-20, because the first version
of the verifier passed the happy path and still got the important case wrong.

## Preflight

```bash
docker compose -f docker/docker-compose.yml up -d db
.venv/bin/uvicorn corpcheck.api.main:app --host 127.0.0.1 --port 8000
curl -fsS http://127.0.0.1:8000/health
```

## The five probes

Each request is the same shape; only `source_text` changes.

```bash
curl -fsS -X POST http://127.0.0.1:8000/claims/verify \
  -H 'content-type: application/json' \
  -d '{
        "source_text": "Apple reported total net sales of $383.3 billion in fiscal 2023.",
        "company_name": "Apple",
        "as_of": "2026-08-20T00:00:00+00:00"
      }'
```

| # | Claim | Truth | Verdict |
| --- | --- | --- | --- |
| 1 | Apple total net sales `$383.3B`, FY2023 | accurate | `verified` |
| 2 | Apple total net sales `$900B`, FY2023 | false | `refuted` |
| 3 | Apple total net sales above `$500B`, FY2028 | unknowable | `not_yet_decidable` |
| 4 | Apple repurchased `$3.7B` of stock, Q2 FY2023 | false — $3.7B was the dividend line; repurchases were $19.1B | `refuted` |
| 5 | Apple total net sales `$383.3B`, FY**2019** | false — wrong year | `insufficient_evidence` |

Probe 1 returns the sentence the number actually lives in:

> …Fiscal Year Highlights The Company's total net sales were **$383.3 billion**…

Probe 2 returns that same sentence as *counter*-evidence: the filing says
$383.3 billion, the claim says $900 billion.

## Why probe 4 exists

Probes 1–3 were passing while the verifier was still wrong. Probe 4 is the one
that exposed it.

The source sentence in Apple's 10-Q reads:

> The Company repurchased **$19.1 billion** of its common stock and paid
> dividends and dividend equivalents of **$3.7 billion** during the period.

A claim of "repurchased $3.7 billion" is false — $3.7 billion is the dividend
figure. The first verifier returned **`verified`**, with `reason_code:
evidence_binds_value_and_metric` while `metric` was `null`. It was asking only
whether the number appeared somewhere in a retrieved chunk.

For a system whose entire purpose is checking financial claims, a false
`verified` is worse than finding nothing.

## What the fix changed

| Failure | Rule now applied |
| --- | --- |
| Unknown metric matched every number | Metric is extracted from a shared lexicon; an unidentified metric **fails closed** |
| Any number in the chunk could bind | A number binds to the metric named **before** it, within 120 characters |
| Movements scored as levels | "net sales decreased 3% or $11.0 billion" is recognised as a change, not a balance |
| Bare integers scored as money | Units must be comparable; week counts and note numbers are excluded |
| Share counts scored against dollars | Monetary claims only bind to monetary figures |
| Rounding read as contradiction | Compared at the precision the claim was written at: `$383.3 billion` matches `$383,285 million` |
| Same fiscal year treated as same instant | The `as_of` cutoff uses `filed_date` at day granularity |
| `refuted` unreachable | Metric + period bound and value disagrees → `refuted` |

`conflicting` is now reserved for in-scope evidence that disagrees with itself,
rather than being the catch-all for "the number wasn't found".

## Verdict taxonomy

| Verdict | Meaning |
| --- | --- |
| `verified` | Metric, period and value all bound to retrieved evidence |
| `refuted` | Metric and period bound; the value disagrees |
| `conflicting` | In-scope evidence supports and contradicts the same claim |
| `insufficient_evidence` | Nothing in scope bound to the claim's metric |
| `not_yet_decidable` | Prediction; the evidence does not exist yet |
| `non_verifiable` | Opinion or unfalsifiable statement |

## Known limitations

Say these before an interviewer finds them.

- **Probe 4's counter-evidence is an authorisation, not the actual buyback.**
  The verdict is right and the figure is a real repurchase-related dollar
  amount, but the ideal receipt would cite the $19.1 billion actually
  repurchased. Evidence *selection* is weaker than evidence *binding*.
- **Probe 5 answers `insufficient_evidence`, not `refuted`.** No FY2019 chunk
  was retrieved, so there was nothing to contradict. That is honest, but a
  stronger system would retrieve the FY2019 filing and refute outright.
- **The metric lexicon is hand-built.** It covers the common income-statement
  and capital-return lines. An unlisted metric fails closed to
  `insufficient_evidence` — safe, but it is a coverage limit, not a solved
  problem.
- **The quarter in `FY2023Q2` is parsed but not enforced** against chunk
  metadata, because quarter provenance is not reliably populated. Only the
  fiscal year gates retrieval today.
- **This is a retrieval-and-binding check, not entailment.** It proves a number
  attached to a metric in a filing, not that the filing's prose supports the
  claim's full meaning.

## Regression coverage

`tests/test_claim_verification.py` pins each failure above, including the
dividend-versus-repurchase case, the change-amount case, the share-count case
and the day-granular cutoff. `tests/test_claim_normalizer.py` pins the
tokenizer-split decimal (`"$ 98. 0 billion"` must not yield `0`).
