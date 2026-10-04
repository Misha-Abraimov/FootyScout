# AI Scout observability

Phase 7 turns the diagnostics already produced by the grounded AI Scout workflow into a stable operational trace:

```text
AI Scout run
  -> canonical workflow result and diagnostics
  -> AIScoutRunTrace normalization
  -> TraceSink
  -> optional JSONL
  -> operational aggregate report
  -> Phase 8 deterministic evaluator and optional focused judges
```

The trace schema is `ai-scout-trace-v1`. It records run and model identity, prompt versions, routing, safe tool summaries, retrieval behavior, evidence IDs and counts, validation/repair outcomes, stage timings, provider usage, optional estimated cost, and normalized errors. Existing workflow diagnostics remain unchanged and are the source of truth.

No application-version or Git-SHA mechanism currently exists. Code SHA is therefore intentionally deferred rather than shelling out to Git on production requests; model, provider, prompt, trace-schema, environment, and validation-mode versions remain explicit.

## Privacy and safety

Traces do not contain the raw user question. They store its SHA-256 hash and character length, plus an optional evaluation `case_id`. Web queries receive the same hash-and-length treatment. Traces do not copy evidence bodies, web snippets, provider request or response payloads, hidden prompts, raw SQL, credentials, headers, API keys, or chain-of-thought. Safe normalized analytics arguments are bounded, and sensitive argument keys are redacted.

Evidence IDs are retained for local correlation with the immutable evidence ledger. The ledger remains the owner of retrieved content.

## Persistence

Production persistence is disabled by default:

```dotenv
AI_SCOUT_TRACE_ENABLED=false
AI_SCOUT_TRACE_PATH=
```

Set both values explicitly to append one JSON object per line to a local JSONL file. A sink write failure is logged as a bounded observability error and does not change the completed AI Scout answer. Available sinks are `NullTraceSink`, `JsonlTraceSink`, and `InMemoryTraceSink` for tests.

For a one-off local run, the existing smoke-test command accepts trace metadata without exposing it to the planner or synthesizer:

```powershell
python -m scripts.run_ai_scout_workflow --question "What does Role Fit mean?" --case-id role_fit_001 --trace-path evals/results/ai_scout_traces.jsonl
```

The public `POST /api/ai-scout` response is unchanged and never exposes trace fields.

## Timings and usage

Canonical timings map the existing workflow stages to planner, preparation, analytics, methodology, current-context preparation, web, web normalization, context assembly, synthesis, validation, and total latency. An unexecuted stage is `null`; zero is reserved for a measured value.

Planner and synthesis input, output, and total tokens remain separate. Overall usage is derived only when the usage needed for an executed LLM stage is available. Missing usage remains `null`, never zero.

## Cost estimates

The reviewed, versioned pricing registry is packaged at `backend/app/runtime_metadata/ai_pricing.json`. An exact provider/model match plus complete input/output usage is required for a complete estimate; cached input is priced separately when the provider reports it. Unknown prices or missing usage produce `null`, not `$0`, while observed lower-bound usage and coverage remain explicit. Web-search cost is not estimated.

## Completion semantics

`terminal_status` retains the answer outcome. `workflow_completed` is true for answered, clarification, unsupported, and insufficient-evidence outcomes, because those can be correct guarded completions. `provider_failure` independently records planner, synthesis, or web provider failures. `success` means the workflow completed without a provider failure. A validation block is separately classified and is not mislabeled as a provider failure.

## Aggregation

The aggregate utility reports operational—not answer-quality—metrics: status/error/repair/block rates; mean, p50, p95, and maximum latency; token and optional cost totals; tool frequencies and failures; web usage/failures; evidence counts; validation outcomes/rules; and model/prompt-version usage.

P50 uses the standard median. P95 uses the deterministic nearest-rank convention: `sorted_values[ceil(0.95 * n) - 1]`. Empty metric populations return `null`.

Summarize any trace file with:

```powershell
python -m scripts.summarize_ai_scout_traces --input evals/results/ai_scout_traces.jsonl
python -m scripts.summarize_ai_scout_traces --input evals/results/ai_scout_traces.jsonl --json
```

Production should use balanced validation as configured by deployment. Evaluation/regression runs use strict validation, attach stable case IDs, and write explicit JSONL artifacts under the ignored `backend/evals/results/` directory.

Phase 8 adds the versioned 85-case deterministic regression suite and optional focused LLM judges. Judge output remains separate from deterministic pass/fail metrics and is disabled unless explicitly requested.
