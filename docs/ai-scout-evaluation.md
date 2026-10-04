# AI Scout Phase 8 Evaluation

Phase 8 evaluates the complete AI Scout workflow without changing production answers:

```text
85 governed golden cases
        ↓
AI Scout finite workflow
        ↓
AIScoutRunTrace (Phase 7)
        ↓
deterministic evaluators + optional focused LLM judges
        ↓
judge calibration evidence
        ↓
JSON/JSONL artifacts + aggregate Markdown report
```

The checked-in golden set is `backend/evals/ai_scout_golden.jsonl`, version
`ai-scout-golden-v2`. Its validator requires exactly 85 cases, stable unique IDs,
deterministic ordering, intentional duplicate-question tags, valid categories/tags/enums,
and non-conflicting tool and methodology expectations. The set retains the original case
IDs while covering entity resolution, profiles, comparisons, similarity, archetypes,
leaderboards, team intelligence, Role Fit, methodology, current-world retrieval, mixed
analytics/methodology/web questions, safe terminals, and adversarial grounding cases.

## Deterministic metrics

Deterministic scoring uses the canonical normalized intent and safe trace fields. It never
asks a judge to infer facts already present in the trace.

- Expected-status accuracy: correct terminal status / all completed cases.
- Planner-intent accuracy: correct canonical post-normalization intent / all cases.
- Entity-resolution accuracy: correct stable ID or expected safe terminal / applicable cases.
- Tool-selection accuracy: cases containing every required tool, no forbidden tool, and no
  tool outside the required/optional sets / all cases.
- Required-tool recall: required tool occurrences present / required tool occurrences.
- Forbidden-tool violation rate: cases using a forbidden tool / cases with forbidden-tool
  expectations.
- Tool-argument accuracy: cases satisfying all bounded argument assertions / cases with
  assertions.
- Methodology-retrieval accuracy: required topics present and forbidden topics absent /
  cases with methodology expectations.
- Web-routing accuracy: required searches executed with the expected category, or forbidden
  searches omitted / cases with a required or forbidden expectation.
- Evidence-category accuracy: required categories present and forbidden categories absent /
  applicable cases.
- Citation-validity case rate: cases whose cited IDs exist and were supplied / applicable
  cases. Citation-reference validity is also reported per reference.
- Historical/current authority accuracy: applicable current-world cases passing the
  deterministic web-authority guard / applicable cases.
- Unsupported-claim violation rate: applicable cases with a known forbidden behavior /
  applicable cases.
- Deterministic overall pass rate: cases where every applicable deterministic check passes /
  completed cases. Judge labels are never included.

N/A cases are excluded from applicable-metric denominators, not counted as successes.
Argument assertions support exact equality, set equality, case-insensitive equality, numeric
ranges, null expectations, and containment. No executable expressions are accepted.

## Focused LLM judges

Judges run after the product workflow and cannot change answers, tools, repair, routing, or
production behavior. They are disabled by default and require an explicit model. Missing or
failed judge calls leave deterministic results intact and reduce judge coverage rather than
creating a negative quality label.

| Criterion | Labels | Prompt version |
| --- | --- | --- |
| Relevance | relevant, partially_relevant, irrelevant | `judge-relevance-v2` |
| Completeness | complete, partially_complete, incomplete | `judge-completeness-v2` |
| Faithfulness | faithful, partially_faithful, unfaithful | `judge-faithfulness-v1` |
| Clarity | clear, unclear | `judge-clarity-v1` |
| Concision | concise, verbose | `judge-concision-v1` |
| Usefulness | useful, partially_useful, not_useful | `judge-usefulness-v1` |
| Context relevance | relevant, partially_relevant, irrelevant | `judge-context-relevance-v2` |

Each criterion is a separate structured call with a short rationale and issue list. The
faithfulness judge receives the controlled question, production-presented answer, and a
bounded, structure-aware sanitized evidence bundle. Deterministic grounding checks continue
to use the internal answer and canonical evidence-ledger IDs. Evaluation artifacts retain both
forms.
Context relevance receives only the question and bounded evidence—not the generated answer—so
it cannot grade answer completeness or writing quality. Other judges receive the controlled
question and answer plus optional reference guidance.
All evaluated text is marked as untrusted data so embedded instructions are ignored. Prompts
do not request or persist chain-of-thought.

LLM-as-a-judge is a proxy for human evaluation and is not an objective metric like
deterministic accuracy.

## Judge calibration

Calibration examples live at `backend/evals/ai_scout_judge_calibration.jsonl`, version
`ai-scout-judge-calibration-v2`. The original 28 reviewed examples are development data after
their first diagnostic run. The seven formerly held-out examples are explicitly marked
`previously_evaluated=true`. Twenty-one fresh held-out examples—three per criterion—were
human-reviewed before their first judge evaluation and are now frozen.

1. Validate and review every candidate's label and notes.
2. Set reviewed examples to `human_approved=true`.
3. Run development-split calibration and inspect accuracy, macro-F1, per-class
   precision/recall/F1, support, and confusion matrices.
4. Freeze judge prompt versions.
5. Approve and evaluate held-out examples.

Statuses are `needs_human_review` (fresh held-out candidates remain unapproved), `unreviewed`
(no approved evaluated labels), `insufficient` (development coverage below the documented
minimum), `development_only`, and `held_out_evaluated`. Only the last status means every
criterion has an evaluated approved fresh held-out example. The system intentionally defines
no arbitrary universal passing threshold.

Each criterion/split reports factual `metric_computable`, `example_count`, `label_support`, and
`label_coverage` fields. One example can make a metric computable but is never described as
statistically sufficient. Calibration artifacts use schema
`ai-scout-judge-calibration-report-v2` and record the safe judge provider/model, prompt
versions, timestamps, approved split counts, token usage, latency, pricing version, and cost
when available. Missing usage or verified pricing remains `null`.

Per-class F1 is calculated directly from confusion counts as
`2TP / (2TP + FP + FN)`. Support-aware macro-F1 averages every label with positive
ground-truth support, including supported labels whose F1 is zero; labels with neither support
nor predictions remain not applicable. Judge criteria can have materially different human
agreement. Consult each criterion's held-out calibration results before treating it as reliable;
FootyScout does not impose or claim a universal calibration pass threshold.

Recompute corrected metrics from the frozen recorded predictions without a provider call:

```powershell
python -m scripts.recompute_judge_calibration --input evals/results/judge-calibration-v2.json --output evals/results/judge-calibration-v2-corrected.json
```

Keep the historical artifact unchanged. Future judge-enabled evaluation commands should pass
`--calibration-report evals/results/judge-calibration-v2-corrected.json`.

## Running evaluations

The canonical regression mode is `strict`; production remains `balanced`. Use a unique
output directory or `--resume`. Resume requires an identical golden version, models,
validation mode, and judge configuration, and skips compatible completed case IDs. It refuses
to append to incompatible artifacts.

The output directory contains:

- `run_config.json`: immutable compatible-run configuration.
- `cases.jsonl`: typed per-case expectations, safe trace, deterministic result, and optional
  judge result.
- `traces.jsonl`: Phase 7 privacy-bounded traces.
- `judge_results.jsonl`: separate judge labels/errors and judge usage.
- `evaluation.json` and `summary.json`: aggregate machine-readable result.
- `summary.md`: human-readable deterministic, judge, operational, and failure analysis.

Partial judge labels remain separate from deterministic failures and appear in the summary's
`Judge Quality Warnings` section with their case IDs.

Failure groups include intent, entity resolution, tool selection, tool arguments,
methodology, web routing, citations, authority, unsupported claims, focused judge failures,
and provider failures.

## Privacy, pricing, and limitations

Artifacts use Phase 7 hashes and safe argument summaries: they do not store secrets, raw
provider payloads, request headers, chain-of-thought, or arbitrary raw user content. Golden
questions are controlled repository fixtures. Stable case IDs are attached only to traces and
evaluation artifacts; they are never inserted into planner or synthesis prompts.

Product workflow and judge tokens, latency, and cost are aggregated separately. Both use the
versioned pricing-registry architecture. The reviewed application registry is packaged at
`backend/app/runtime_metadata/ai_pricing.json`; unrecognized provider/model pairs or incomplete
usage remain `null`, with observed lower-bound totals and coverage reported separately. A run may
instead receive a reviewed registry through `--pricing-registry path/to/pricing.json`. The pricing
version becomes part of the resume compatibility contract; evaluation never infers or scrapes
prices during a run.

Direct judges remain model-based proxies and can be biased or inconsistent. Calibration
measures agreement only on the reviewed examples. Deterministic validators cover governed
failure classes but cannot prove that every possible unsupported claim is absent. Pairwise
model/prompt comparison and automatic prompt optimization are intentionally deferred.
