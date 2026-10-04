# Controlled current-web context

## Purpose and authority boundaries

AI Scout uses optional web search only for time-sensitive facts that FootyScout's
historical event corpus cannot establish, such as a current club, injury or availability
update, recent reporting, or current manager.

The three evidence categories have different authority:

- **Analytics** is authoritative for FootyScout-computed xPass, xG, possession value,
  attacking impact, profiles, percentiles, archetypes, similarity, team intelligence,
  Role Fit, and recommendations.
- **Methodology** explains how those governed calculations should be interpreted.
- **Web** provides current external context. It never recalculates, overrides, mutates,
  or updates analytics.

Pure current-information requests bypass analytics execution and methodology retrieval
after bounded plan preparation. Their substantive synthesis context contains web evidence
only, so historical cohort metadata cannot be presented as a current-world fact. Mixed
requests retain their requested analytics and methodology alongside separately labeled
web evidence.

## Provider abstraction and Exa

`WebSearchProvider` accepts a FootyScout-owned `WebSearchRequest` and returns a normalized
`WebSearchResponse`. The first adapter, `ExaWebSearchProvider`, uses Exa's official
`POST /search` API with automatic search, a hard result limit, optional publication-date
filtering, and bounded highlights. Provider response objects never cross the adapter.

The implementation uses the documented REST surface through `httpx` rather than coupling
workflow code to an SDK. This keeps explicit timeout behavior and future provider
replacement inside one adapter.

## Currentness and query construction

A narrow deterministic guard recognizes explicit current-language categories after the
planner runs. It does not search ordinary historical analytics or methodology questions.
The user's explicit entity in the current-information clause takes precedence over
planner metadata; resolved planner/evidence identity is a fallback for mixed questions
that use pronouns.

The application—not the model—constructs one normalized query from the resolved subject
and an allowlisted category suffix. It performs at most one search pass with a maximum of
five returned results. There is no browser, page clicking, arbitrary fetch, ReAct loop,
or autonomous query refinement.

Words such as “latest,” “current,” and “recent” stay in the normalized query and do not
create a publication-date cutoff. `startPublishedDate` is sent only when the user states
an actual bounded window, such as “in the last 7 days,” “since September 1,” or “news
from 2026.” Freshness-sensitive injury, availability, latest-news, and transfer searches
instead use the application-controlled `contents.maxAgeHours` cache-age limit. Current
club and manager lookups omit that freshness control so older authoritative pages are not
excluded. Deprecated live-crawl options are not used.

## Selection, evidence, and provenance

Search results are normalized to validated URLs, optional publication dates, concise
highlights, retrieval timestamps, and deterministic source-quality categories
(`official`, `reputable_media`, or `other`). Selection rejects results unrelated to the
resolved subject, deduplicates URLs, prefers source quality and recency, encourages domain
diversity, and supplies at most three records to synthesis.

The small reputable-media registry includes established general and sports-news
publishers; it is a classification aid, not a domain allowlist. Exa still searches
broadly.

Each selected result becomes an immutable `web` `EvidenceRecord`. Analytics,
methodology, and web records stay separate in selected context. The synthesis model cites
ledger IDs only; after citation validation, application code derives the final web source
title, URL, domain, publication date, and quality classification.

## Failure and degraded behavior

- A mixed analytics/current request may return the grounded FootyScout analysis with an
  explicit limitation when search fails or yields no relevant evidence.
- A current-only request without adequate web evidence returns
  `insufficient_evidence`; no historical fact is substituted and an empty search is never
  interpreted as proof of a negative.
- With web search disabled, existing analytics and methodology behavior is unchanged and
  no Exa credential is required.

## Security

Web snippets are untrusted evidence, never instructions. Text is control-character
sanitized and length-bounded. Search accepts no model-provided headers, cookies,
credentials, methods, code, files, URLs to fetch, or scraping instructions. API keys and
authorization headers are excluded from evidence, diagnostics, traces, and answers;
provider failures expose only sanitized messages and safe metadata.

## Local configuration

Web search is disabled by default. Add values to a local, uncommitted `backend/.env` or
set them in the current shell:

```dotenv
AI_SCOUT_WEB_SEARCH_ENABLED=true
AI_SCOUT_WEB_SEARCH_PROVIDER=exa
EXA_API_KEY=
AI_SCOUT_WEB_MAX_RESULTS=5
AI_SCOUT_WEB_TIMEOUT_SECONDS=15
AI_SCOUT_WEB_CONTENT_MAX_AGE_HOURS=24
```

An Exa key is required only when the Exa provider is enabled. The existing planner and
synthesis provider credentials are also required for the end-to-end smoke script.

## Offline validation

Normal tests use `FakeWebSearchProvider` and make no provider requests:

```powershell
Set-Location backend
python -m pytest tests/ai/test_web_context.py
```

## One explicit real smoke test

Do not echo keys. After setting `OPENAI_API_KEY` and `EXA_API_KEY` in the current shell:

```powershell
Set-Location backend
$env:AI_SCOUT_PROVIDER = "openai"
$env:AI_SCOUT_WEB_SEARCH_ENABLED = "true"
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "Set OPENAI_API_KEY first." }
if ([string]::IsNullOrWhiteSpace($env:EXA_API_KEY)) { throw "Set EXA_API_KEY first." }
python -m scripts.run_ai_scout_workflow --question "What is the latest reliable update on Florian Wirtz?"
```

This command is intentionally manual because it invokes paid external providers.
