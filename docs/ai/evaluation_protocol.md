# AI Scout evaluation protocol

Status: Production deterministic regression protocol (`ai-scout-golden-v2`)

AI Scout is evaluated as a governed analytics interface, not as a general
chatbot. FootyScout's persisted analytics and packaged model metadata are the
source of truth. Generated language explains and combines that evidence, but it
may not calculate replacement values.

## Golden cases

`backend/evals/ai_scout_golden.jsonl` is the versioned behavioral specification.
Each record includes a distinct question, expected intent and entity behavior,
allowed/required/forbidden tools, deterministic arguments where applicable,
methodology sources, expected clarification or insufficient-evidence behavior,
and forbidden claims.

The versioned file contains cases for:

- entity resolution and player search;
- player profiles, comparisons, and leaderboards;
- similarity and archetypes;
- team intelligence, Role Fit, and role recommendations;
- model and metric methodology;
- unsupported, ambiguous, and adversarial requests.

The cases are executable specifications for deterministic workflow evaluation.
Normal unit tests remain provider-free; live evaluations require explicit
authorization and write generated artifacts beneath the ignored results tree.

## Evaluation layers

### Deterministic contracts

Programmatic tests must verify:

- Pydantic structured-output validity and rejection of extra fields;
- entity resolution, ambiguity, and not-found behavior;
- tool selection allowlists and hard result limits;
- REST/tool parity against the same database fixtures;
- exact numeric values, ordering, nulls, reliability fields, and warnings;
- production versus experimental model labels;
- the request-time import boundary against `backend/analytics`.

Numbers and structured claims have exact answers and must be checked exactly or
with an explicitly justified floating-point tolerance. An LLM judge is not an
acceptable substitute for these assertions.

### Planner metrics

Provider-backed planner evaluations measure:

- intent classification accuracy;
- entity-resolution and clarification accuracy;
- required, allowed, and forbidden tool-selection accuracy;
- exact and field-level argument accuracy;
- structured-output validity;
- tool ordering and step count;
- task completion and appropriate unsupported behavior.

Planner evaluation must be run against a pinned model identifier, prompt version,
tool-registry version, and analytics snapshot. Normal unit tests should remain
provider-free; networked evaluations should run in an explicitly authorized job.

### Retrieval metrics

Methodology retrieval must be evaluated separately from structured analytics.
Measure source recall, source precision, production-status accuracy, relevant
section selection, and citation correctness. Experimental GRU/Transformer
documents must never displace the packaged production XGBoost metadata.

Do not introduce embeddings or pgvector until a lexical/section baseline exists
and measured retrieval failures justify the additional system.

### Grounding metrics

Every numeric claim must link to a field in the immutable tool-result ledger.
Every methodology claim must link to a versioned source section. Measure:

- exact numeric-grounding rate;
- unsupported numeric-claim count;
- citation entailment and citation correctness;
- reliability/sample-warning preservation;
- directionality correctness, especially lower-is-closer Role Fit distance;
- forbidden-claim violation rate.

The release goal for supported cases is 100% structured validity, 100% numeric
grounding, and zero forbidden-claim violations in the golden set.

## Forbidden-claim testing

The suite must explicitly reject claims that:

- style similarity measures player quality or predicts transfer success;
- Role Fit predicts transfer success, future performance, or lineup selection;
- archetypes are quality ratings;
- observed team style proves coaching intent;
- the Transformer or GRU is the production possession-value model;
- AI-generated values replace FootyScout xPass, xG, possession value, attacking
  impact, percentiles, archetypes, similarity, Role Fit, or recommendations;
- FootyScout contains transfer value, wage, contract, injury, or availability
  evidence that is not present in the product.

## Prompt regression and release comparison

Future changes to a prompt, provider model, tool description, tool schema,
methodology corpus, or orchestration policy require a baseline-versus-candidate
run over the same golden set. Store aggregate and per-case results, including:

- correctness metrics;
- p50/p95 latency;
- prompt and completion tokens;
- estimated cost;
- tool-call count and total agent steps;
- retries, validation failures, and safe refusals.

Regressions in grounding, forbidden claims, or schema validity block release even
when fluency improves.

## Use of model-based judges

A focused model-based judge may supplement human review for clarity, relevance,
faithfulness, concision, usefulness, completeness, and context relevance. It is
disabled by default and must never be the only evaluator. Intent, tools, arguments,
retrieval sources, numbers, citations, directionality, and prohibited claims are
deterministic properties and must remain programmatically asserted.
