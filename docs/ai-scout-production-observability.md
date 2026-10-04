# AI Scout production observability and reliability

AI Scout keeps its existing planner, deterministic tools, evidence ledger, grounded
synthesis, and strict answer validation. The production controls in this document are
additive and fail closed for authorization while failing open for telemetry export.

## Trace topology

Each HTTP request receives a UUID correlation ID. A valid incoming `X-Request-ID` is
preserved; otherwise FootyScout generates one and returns it in the response header. The
same value is the AI Scout run ID, the OpenTelemetry `ai.correlation_id`, and LangSmith
run metadata.

LangSmith is opt-in through `LANGSMITH_TRACING_ENABLED`. Native LangGraph and LangChain
runs use safe tags and metadata. `LANGSMITH_HIDE_INPUTS` and `LANGSMITH_HIDE_OUTPUTS`
are forced on so raw questions, prompts, evidence, and answers are not sent as trace
payloads. Missing credentials disable LangSmith rather than blocking a request.

OpenTelemetry is independently opt-in through `OTEL_ENABLED`. OTLP/HTTP export,
FastAPI request instrumentation, SQLAlchemy instrumentation, orchestration spans,
graph-stage spans, and deterministic tool spans are enabled only when an endpoint is
configured. Initialization, span creation, and export failures are isolated from the
answer path.

No trace metadata contains API keys, authorization headers, raw prompts, arbitrary
query text, or full retrieved documents. User questions and web queries are represented
in local operational traces by a SHA-256 hash and length.

## Measurements and pricing

The local JSONL trace schema records planner and synthesis input/output/total tokens
separately, stage latency, total request and AI latency, logical provider call counts,
tool usage, web usage, validation outcome, terminal state, and safe error classes.
Aggregation reports nearest-rank p95, median p50, mean/median tokens, and mean/median
cost.

Costs are calculated only when both token counts and an exact provider/model entry are
available. The versioned registry lives at
`backend/app/runtime_metadata/ai_pricing.json`; each entry records its effective date,
units, cached-input price where published, and source URL. Unknown usage or pricing
produces `null`, never a guessed cost.

## Security boundaries

Requests are typed, limited to 2,000 characters, and reject unsafe control/surrogate
characters. Planner output must pass strict typed schemas, deterministic entity
resolution, per-intent tool allowlists, and bounded tool argument models. The executor
checks the intent allowlist again immediately before registry dispatch. No registered
tool exposes arbitrary Python, SQL, shell, filesystem, or URL execution.

External web text is serialized to synthesis with the explicit boundary
`UNTRUSTED_EXTERNAL_DATA_NOT_INSTRUCTIONS`. It remains evidence data and cannot change
tool permissions, prompts, citation policy, or strict validation. Immutable evidence
IDs, authority rules, unsupported-claim checks, and citation validation remain
unchanged.

## Reliability

OpenAI planner, Anthropic planner, OpenAI synthesis, and Exa calls have bounded explicit
timeouts (Anthropic can intentionally retain its SDK-managed non-streaming timeout).
SDK retries remain bounded to zero through two. Exa uses the shared transient classifier
and retries only timeouts, connection failures, HTTP 429, and selected 5xx responses
with capped exponential backoff, jitter, and `Retry-After` support.

Optional planner fallback is configured with `AI_SCOUT_FALLBACK_PROVIDER`. It runs only
after a transient primary error and preserves the same typed planner contract. It never
runs for authentication, malformed requests, structured-output validation, ordinary
4xx responses, clarification, or unsupported requests. Fallback attempt/provider/reason/
success are stored in tracing. Synthesis does not silently cross providers because no
second provider currently implements the grounded structured synthesis contract.

`AI_SCOUT_MAX_OUTPUT_TOKENS` bounds planner and synthesis output. Web unavailability
continues through the existing controlled degradation path: current-world claims are
omitted unless sufficient authoritative evidence survives.

## Baseline status

The canonical final deterministic benchmark is run
`eval-20261004T184153Z-0857a20d`: 83/85 (97.6%) deterministic passes, 100% citation
validity, 108/108 valid citation references, 0/51 unsupported-claim violations, and
0/19 forbidden-tool violations. It recorded 554,414 product LLM tokens, including
173,133 planner tokens, at an estimated cost of $0.2884926. Local-run latency was
11.873 seconds p50 and 23.558 seconds p95, with planner p95 of 7.047 seconds. These are
benchmark observations, not a hosted-service SLA. Generated run artifacts stay under
the ignored `backend/evals/results/` tree rather than source control.
