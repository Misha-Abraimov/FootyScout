# Grounded AI Scout response workflow

## Architecture

The grounded response path is deliberately bounded:

```text
question
  -> LangGraph routing
  -> existing ScoutPlanner
  -> LLMPlannerDecision
  -> deterministic PlanBuilder and strict ScoutPlan
  -> deterministic normalization and entity resolution
  -> registered FootyScout analytics tools
  -> immutable EvidenceLedger
  -> exact-topic curated methodology retrieval
  -> bounded current-context decision
  -> optional one-shot web search and normalized web evidence
  -> deterministic context selection and sufficiency check
  -> LangChain OpenAI structured synthesis
  -> deterministic finding detection
  -> strict or balanced validation policy
  -> optional one-pass claim removal
  -> grounded answer
```

### LangGraph

LangGraph owns state, stage ordering, and finite conditional routing. The graph has no
agent loop and cannot select arbitrary tools. It routes planner failures, clarification,
unsupported requests, execution failures, insufficient evidence, synthesis, and final
citation validation. Current-web routing adds no ReAct or retry loop: the graph can make
at most one bounded search request before context assembly.

### LangChain

LangChain builds the synthesis messages and provides the OpenAI chat-model abstraction
and structured-output boundary. It is not used as a ReAct agent. It never chooses which
FootyScout tool runs; the validated `ScoutPlan` remains authoritative.

### Custom FootyScout code

The existing planner DTO, deterministic `PlanBuilder`, `ScoutPlan`, normalization,
entity resolution, tool registry, domain validation, evidence ledger, context selector,
and citation validator remain application-owned. This preserves the tested planning and
analytics contracts.

### PostgreSQL

PostgreSQL remains the structured source for player, team, and persisted football
analytics. The LLM cannot generate SQL and receives only selected tool evidence.

### Curated methodology service

Methodology is retrieved by an allowlisted exact topic. Each document includes a stable
source identifier, path, section, production status, content, and limitations. The
service reuses the existing governed model metadata and methodology tool where possible.
Only topics relevant to the validated intent and executed analysis are selected.

### Optional current-web service

Current external facts are a third evidence category, separate from analytics and
methodology. A narrow deterministic guard detects explicit current-language requests.
When enabled, a provider-neutral adapter performs one bounded search, normalizes the
results into FootyScout schemas, selects a small relevant set, and appends immutable web
records to the ledger. Web reports cannot alter persisted FootyScout analytics.

### Why there is no vector database

The current methodology corpus is small, structured, and maps directly to supported
intent and tool enums. Deterministic topic selection is simpler to audit and prevents
unrelated model documentation from entering the synthesis context. Embeddings and a
vector database would add operational complexity without a measured retrieval problem.

## Grounding rules

- Final synthesis sees only the question, category-separated selected evidence, and
  explicit limitations.
- Numeric and factual claims must cite current-run evidence IDs.
- Declared and inline evidence IDs are checked against the immutable ledger.
- The synthesis model emits evidence IDs only. After validation, application code resolves
  cited methodology records and derives stable methodology source IDs in citation order.
- Cited web evidence IDs are resolved by application code to title, validated URL, domain,
  publication date when supplied, and source-quality category. The model never authors
  source URLs.
- The minimal `LLMGroundedAnswer` DTO is therefore distinct from the enriched final
  `GroundedScoutAnswer` application DTO.
- Role Fit is never converted to a 0–100 score or presented as transfer-success
  prediction.
- Production and experimental model status remain distinct.
- Clarification, unsupported, and insufficient-evidence paths produce deterministic safe
  responses without calling the synthesis model.

## Observability

Each workflow result records planner and synthesis provider/model identifiers, prompt
versions, available token usage, stage timings, executed tools, generated/supplied/cited
evidence IDs, terminal status, and sanitized errors. No secrets, headers, environment
variables, or raw provider requests are stored.

Web diagnostics additionally record whether search was required and enabled, the bounded
normalized query, provider, result counts, selected evidence IDs/domains/dates, latency,
failure stage, sanitized provider error, and degraded status.

## Validation policy

Detection and response policy are separate. The deterministic validator produces immutable
findings using the established provenance, current-world authority, and Role Fit language
rules. `strict` mode blocks every finding and remains the application default for regression
and evaluation. `balanced` mode remains strict on broken provenance but can remove a bounded
semantic claim, revalidate once, and preserve the rest of a grounded answer. Production must
set `AI_SCOUT_VALIDATION_MODE=balanced`; evaluation and regression use `strict`.

The public `POST /api/ai-scout` response never includes the full evidence ledger, planner
internals, prompts, token usage, detailed guardrail findings, or provider responses. It
contains only the answer contract and safe source summaries needed for user presentation.

## Role Fit cohort ranking

Role Fit rank is empirical rather than an invented distance band. A comparison cohort shares
`target_team_id`, `position_group`, `calculation_scope`, and
`is_target_team_player`. Persisted Role Fit rows are already the eligible/available rows.
Ordering is ascending raw distance with ascending `player_id` as the deterministic tie-break.
The recommendations endpoint uses the same canonical cohort helper. Rank communicates
relative stylistic resemblance only, never player quality, transfer success, or future
performance.
