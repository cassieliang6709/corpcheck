# Hugging Face release checklist

CorpCheck currently uses a public embedding base model and an optional public
generation model, but it does not yet contain CorpCheck-trained weights. The
first Hub release should therefore be a **Space plus evaluation artifact**, not
a misleading model repository.

## Proposed public assets

| Asset | Proposed repository | What it proves |
| --- | --- | --- |
| Space | `cassieliang6709/corpcheck` | The evidence-first product flow and product boundary |
| Dataset | `cassieliang6709/corpcheck-eval` | Reproducible schemas, split metadata, exclusions, and run summaries |
| Model | `cassieliang6709/corpcheck-reranker` | Create only after trained weights pass held-out gates |

## Before publishing

- [ ] Run `hf auth login` and confirm with `hf auth whoami`.
- [ ] Confirm the Hugging Face namespace and public repository names.
- [ ] Audit every uploaded file for `.env`, API keys, database URLs, SEC contact
  headers, local absolute paths, and non-redistributable benchmark text.
- [ ] Pin the GitHub commit used by the Space and evaluation card.
- [ ] Record the corpus snapshot date and whether results are live or recorded.
- [ ] Link the upstream base-model pages and preserve their licenses.
- [ ] State that generation is disabled in the current public demo.

## Suggested creation commands

Run only after authentication and the content audit:

```bash
hf repos create cassieliang6709/corpcheck --type space --space-sdk static
hf repos create cassieliang6709/corpcheck-eval --type dataset
```

Do not create `corpcheck-reranker` until a local release directory contains
actual trained weights, tokenizer/config files, training provenance, evaluation
results, and a complete model card.

## Space behavior

The first public Space should be static and honest:

- Default to the same recorded evidence states as the landing page.
- Label the verification date and corpus snapshot beside every recorded result.
- Disable arbitrary custom queries when no live API is configured.
- Link to the full GitHub source and the exact evaluation protocol.
- Never put API credentials in browser JavaScript.

A live API can be connected later only after authentication, rate limiting,
health monitoring, CORS restrictions, and a cost ceiling are in place.

## Model card requirements for the future reranker

The model card must include:

- Base model and license.
- Training data provenance and redistribution status.
- Negative-sampling strategy, especially wrong-year, wrong-issuer, wrong-form,
  and superseded-section negatives.
- Train/validation/test split boundaries and leakage checks.
- Baseline and candidate metrics on the same fixed corpus.
- Refusal and provenance regression checks.
- Known limitations and unsupported uses.
- Minimal inference example that produces a ranking score without requiring the
  CorpCheck database.
