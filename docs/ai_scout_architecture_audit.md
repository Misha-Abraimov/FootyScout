# FootyScout AI Scout architecture audit

Status: design only
Date: 2026-09-27

This report defines a foundation for a future AI Scout without implementing an
LLM, LangGraph, RAG, MCP, a chatbot, or any new production behavior. The
existing analytics, persisted data, and FastAPI contracts remain the source of
truth.

> **ML calculates. Tools retrieve. RAG explains. The LLM combines. LangGraph
> controls. Evaluation proves it works.**

## Executive recommendation

Build AI Scout as a thin, typed orchestration layer over FootyScout's existing
read-only application services. It must never import training pipelines at
request time, execute generated SQL, or derive model values itself. The first
implementation phase should contain no LLM at all: extract reusable query
services from the existing routers, define validated intent and tool contracts,
and create a golden evaluation corpus. This establishes the deterministic
boundary that a later planner can call safely.

The current repository already exposes nearly all required information. The
main architectural gap is that several SQLAlchemy queries live directly in
routers. Those queries should be moved, without behavioral changes, into a
small `app/services` layer so both the REST endpoints and future tools reuse the
same code.

## A. Current architecture relevant to AI Scout

### Request-time application

- `backend/app/main.py` constructs the FastAPI application, configures CORS,
  includes all public routers, and exposes `/health`.
- `backend/app/config.py` loads typed settings with `pydantic-settings`, including
  the database URL, app metadata, environment, and allowed origins.
- `backend/app/database.py` owns the SQLAlchemy engine and session factory.
- `backend/app/dependencies.py::get_db` provides one database session per
  request and closes it after use.
- `backend/app/models.py` defines the persisted product data.
- `backend/app/schemas.py` defines constrained response types and public enums.
- `backend/app/routers/` contains read-only HTTP endpoints and, in several
  cases, their SQLAlchemy query construction.
- `backend/app/presenters.py`, `backend/app/intelligence.py`,
  `backend/app/archetypes.py`, and `backend/app/team_intelligence.py` already
  provide reusable response-building logic.
- `backend/app/runtime_metadata.py` loads packaged, immutable model metadata
  from `backend/app/runtime_metadata/*.json` without importing scientific
  dependencies.

This is a deliberately lightweight production path. `backend/pyproject.toml`
keeps FastAPI, SQLAlchemy, Alembic, psycopg, Pydantic settings, and Uvicorn in
the request-time dependency set. Pandas, scikit-learn, PyTorch, XGBoost, and
the other analytics packages live in extras and are not required by the Vercel
backend.

### Offline analytics

`backend/analytics/` is the training, evaluation, feature-engineering, and
artifact-generation boundary. Important governed contracts include:

- `pass_features.py::MODEL_FEATURE_COLUMNS` and its leakage denylist for xPass.
- `shot_features.py::XG_MODEL_FEATURE_COLUMNS` and its leakage denylist for xG.
- `action_value_features.py` for possession-state and attacking-impact inputs.
- `player_feature_registry.py::FEATURE_REGISTRY` for style and performance
  metric definitions, reliability thresholds, directionality, and radar use.
- `player_similarity.py::SIMILARITY_FEATURE_COLUMNS` for the frozen six-feature,
  same-position similarity space.
- `player_archetypes.py` for the production archetype feature contract.
- `team_role_research.py::FIT_FEATURES` and `team_intelligence.py` for the
  frozen Role Fit space and Bayer Leverkusen target-team implementation.

These modules must not be imported by the future request-time AI layer. Their
outputs are already materialized in PostgreSQL and packaged metadata. The
future tools must read those outputs only.

### Persistence and data loading

- Alembic migrations under `backend/alembic/versions/` evolve the schema.
- `backend/scripts/load_database.py` validates generated Parquet artifacts and
  replaces the application snapshot.
- PostgreSQL is exposed on host port `5433` in local Docker configuration.
- Production uses Neon through the same SQLAlchemy models and query contracts.
- No analytics job runs as part of an API request.

### Frontend

- `frontend/app/` uses the Next.js App Router.
- `frontend/app/layout.tsx` supplies the global shell, navigation, and footer.
- `frontend/components/Navbar.tsx` defines the first-class product navigation.
- `frontend/lib/api.ts` is the typed API client and contains the browser-relative,
  local-development, and Vercel server-side base-URL resolution rules.
- `frontend/lib/types.ts` mirrors the FastAPI response models.
- Pages exist for player search and profiles, comparison, leaderboards,
  archetypes, model methodology, team intelligence, and scouting recommendations.
- Vitest and Testing Library tests are colocated with pages/components or live
  under `frontend/lib/` for client behavior.

The future page belongs at `/ai-scout` as a peer of Players, Team Intelligence,
and Scouting. It should be added to the main navigation only after the grounded
backend flow and evaluation gate exist.

## B. Existing capabilities that can be reused

### Capability matrix

| Capability | Existing endpoint/function | Validated inputs | Returned evidence | Source | AI-tool suitability and gap |
|---|---|---|---|---|---|
| Player search | `GET /api/players`; `routers/players.py::list_players` | Search text, exact team, `GK/DEF/MID/FWD`, minimum pass attempts, allowlisted sort, order, limit, offset | Total, pagination, identity, team, position, pass attempts, headline passing metrics | `Player`, `PlayerProfile` | Suitable. Extract query to a service; cap AI results below the public maximum. Name results may require disambiguation by stable `player_id`. |
| Player passing profile | `GET /api/players/{player_id}`; `get_player_profile_or_404`; `presenters.py::player_profile` | Positive player ID | Identity, sample/reliability and passing aggregates | `Player`, `PlayerProfile` | Suitable. Reuse presenter and query service. Never recompute xPass aggregates. |
| Player intelligence | `GET /api/players/{player_id}/intelligence`; `intelligence.py::get_intelligence_response` | Positive player ID | Style/performance metrics, eligibility, same-position percentiles, radar axes, archetype | `PlayerIntelligenceProfile`, `PlayerPercentile`, `PlayerArchetype` | Suitable as part of a player dossier. Preserve null percentiles and peer/sample limitations. |
| Player shooting | `GET /api/players/{player_id}/shooting` | Positive player ID | Shot volume, goals, xG, finishing and reliability fields | `PlayerShootingProfile` | Suitable as optional player evidence. It should not be requested for unrelated style questions. |
| Player attacking impact | `GET /api/players/{player_id}/attacking` | Positive player ID | Pass, carry, progressive and pressure action-value summaries plus support | `PlayerAttackingProfile` | Suitable. Values are persisted calculations and must not be recreated by the LLM. |
| Similar players | `GET /api/players/{player_id}/similar`; `routers/players.py::get_similar_players` | Player ID; limit 1–10 | Ranked same-position candidates, style similarity, raw distance, feature contributions and support context | `PlayerSimilarity`, aliased `Player` | Suitable. Must state that similarity is not quality, potential, future performance, or transfer success. Extract query to service. |
| Player event evidence | `GET /api/players/{id}/passes`, `/shots`, `/actions` | Bounded filters and pagination; maximum 500 rows | Persisted per-event xPass, xG, attacking value, geometry and context | `Pass`, `Shot`, `AttackingAction` | Useful only for explicit evidence/drill-down questions. Do not expose arbitrary bulk exports to the LLM. Add an AI-specific smaller cap and summary-first behavior. |
| Comparison | `GET /api/compare?player_ids=a,b`; `routers/compare.py::compare_players` | Exactly two unique positive player IDs | Two player identities, passing/attacking/intelligence metrics and comparison payload | Player/profile/intelligence tables | Suitable. Current response is not a full shooting or Role Fit comparison; the tool must not imply it is. Extract orchestration/query logic to service. |
| Leaderboard/search by metric | `GET /api/leaderboard`; `routers/leaderboard.py::get_leaderboard` | Allowlisted metric, optional position/team, bounded limit | Ranked eligible players with metric values and sample context | `Player`, `PlayerProfile`, `PlayerAttackingProfile` | Suitable for bounded shortlist requests. Tool must use the existing metric enum and reliability rules rather than accepting column names. |
| Archetype catalogue | `GET /api/archetypes`; `archetypes.py::archetype_catalogue_response` | None | Archetype definitions, features, counts and representatives | Runtime metadata plus `PlayerArchetype` | Suitable as methodology/catalogue evidence. Archetype is style grouping, not player quality. |
| Team intelligence | `GET /api/teams/{team_id}/intelligence`; `team_intelligence.py::get_team_intelligence` | Existing team ID | Team style, qualification/support, warnings, and positional roles | `TeamStyleProfile`, `TeamRoleProfile` | Suitable. Current production scope contains only the qualified target team; absence for other teams must be explicit. |
| Team roles | `GET /api/teams/{team_id}/roles` and `/roles/{position_group}`; `team_intelligence.py::get_team_roles` | Team ID and outfield position group | Observed pooled role profile, support, feature values and warnings | `TeamRoleProfile` | Suitable, usually folded into the team-intelligence tool. Goalkeepers are unsupported. |
| Role Fit | `GET /api/players/{id}/role-fit?target_team_id=...`; `team_intelligence.py::player_role_fit_response` | Player ID and target team ID | Role distance, fit index/rank where applicable, external/current-player status, support and warnings | `PlayerRoleFit`, player and team-role records | Suitable. It measures similarity to an observed positional role; it is not a probability of transfer success or future performance. |
| Role recommendations | `GET /api/teams/{id}/roles/{position}/recommendations`; `team_intelligence.py::scouting_recommendations_response` | Team ID, `DEF/MID/FWD`, bounded limit | External same-position candidates ordered by ascending role distance with fit/support context | `PlayerRoleFit`, `Player` | Suitable for controlled shortlists. It covers style fit only and must retain limited-sample warnings. |
| xPass methodology | `GET /api/model`; `routers/model_info.py` | None | Selected model, features, validation/test/OOF metrics, calibration and methodology | Packaged `pass_model_metadata.json` | Suitable for methodology, not player-data queries. |
| xG methodology | `GET /api/models/xg`; `routers/xg_model_info.py` | None | Model, features, grouped evaluation, calibration and methodology | Packaged `xg_model_metadata.json` | Suitable for methodology. Provider xG is diagnostic only and not a feature or label. |
| Possession value / attacking-impact methodology | `GET /api/models/action-value`; `routers/action_value_info.py` | None | Current XGBoost state model, feature and evaluation metadata, action-value methodology | Packaged `action_value_model_metadata.json` | Suitable for methodology. The Transformer remains an offline experiment, not production. |
| Product metadata | `GET /api/meta` | None | Teams, position groups, player counts and reliability counts | `Player` aggregates | Useful for entity discovery and user-facing filter context, but not necessarily a separate LLM tool. |

### Service-layer work needed before tool exposure

The presenters and intelligence modules are already reusable. The following
queries should be extracted from routers into pure request-time services that
accept a SQLAlchemy `Session` and validated domain inputs:

- player search/profile/similarity/event queries from
  `backend/app/routers/players.py`;
- comparison assembly from `backend/app/routers/compare.py`;
- leaderboard filtering/ranking from `backend/app/routers/leaderboard.py`;
- role-recommendation SQL from `backend/app/routers/teams.py`.

REST routers should call the extracted services, and AI tools should call those
same services. AI tools should not call the application's own HTTP endpoints,
and they should not duplicate SQL.

## C. Database models and ownership boundaries

| Model/table | Authoritative information | Intended AI access |
|---|---|---|
| `Player` | Stable identity, current team/position group, match and pass sample sizes, reliability | Through player services only |
| `PlayerProfile` | Aggregated xPass and passing metrics | Through player profile/leaderboard services |
| `PlayerSimilarity` | Directed, ranked same-position style neighbors, score/index, raw distance, dimension contributions and support | Through similarity service only |
| `Pass` | Every product pass with OOF expected completion and pre-outcome context | Through bounded evidence service only |
| `Shot` | Product shots with OOF xG where eligible and shot context/outcome | Through bounded evidence service only |
| `PlayerShootingProfile` | Player-level xG/shooting aggregates | Through profile service |
| `AttackingAction` | Pass/carry start and end possession values, attacking value, risk and context | Through bounded evidence service only |
| `PlayerAttackingProfile` | Player-level attacking-impact aggregates and support | Through profile/leaderboard service |
| `PlayerIntelligenceProfile` | Unified style and performance metrics | Through intelligence service |
| `PlayerPercentile` | Metric-specific eligibility, peer count, raw values and same-position percentile | Through intelligence service; never infer missing percentiles |
| `PlayerArchetype` | Production archetype assignment, confidence/distance and support | Through intelligence/archetype service |
| `TeamStyleProfile` | Qualified team style profile and support | Through team intelligence service |
| `TeamRoleProfile` | Observed team-position role profile and support | Through team role service |
| `PlayerRoleFit` | Persisted external or leave-self-out role-fit result, distance, index/rank and warnings | Through Role Fit/recommendation services |

There is no general model-metadata database table. Production model metadata is
packaged as JSON under `backend/app/runtime_metadata/` and loaded through
`backend/app/runtime_metadata.py`. This split should remain explicit:

- player/team results come from PostgreSQL;
- model methodology and immutable evaluation metadata come from packaged JSON;
- long-form explanations come from a later curated documentation index.

The future AI layer should never query these tables with model-generated SQL.
Even internal tool implementations should prefer the shared service layer so
reliability filters, order direction, join behavior, and response semantics stay
consistent with the product.

## D. Documentation suitable for a future RAG corpus

### Recommended curated sources

- `README.md`: product architecture, scope, routes, local/deployment workflow,
  limitations, and public terminology.
- `backend/analytics/README.md`: xPass, xG, possession value, attacking impact,
  intelligence profiles, archetypes, similarity, and Role Fit methodology.
- `docs/experiments/gru_possession_value.md`: GRU experiment design and result.
- `docs/experiments/transformer_possession_value.md`: causal Transformer
  experiment and the decision to retain XGBoost in production.
- `backend/scripts/README.md`: data ingestion and reproducibility context.
- Selected, human-authored metric definitions derived from
  `backend/analytics/player_feature_registry.py` in a future generated glossary.
- Packaged runtime metadata for exact model names, versions, features and
  evaluation numbers, accessed structurally rather than embedded as prose.

### Sources not suitable for direct RAG ingestion

- raw StatsBomb events, Parquet datasets, binary model artifacts, and database
  dumps;
- source code as a substitute for methodology documentation;
- tests, build output, cache directories, or generated frontend bundles;
- secrets and environment files;
- experimental notes without an explicit production/experimental status.

### Required provenance fields

Each future document chunk should carry `document_id`, source path, section,
content version or Git commit, last-reviewed date, production status, and a
stable citation label. Retrieval must never blur experimental Transformer
results with the current production XGBoost possession-value model.

A small corpus can initially use deterministic section-level lexical retrieval.
`pgvector` is not justified until retrieval evaluation demonstrates that the
curated corpus and PostgreSQL full-text or lexical search are insufficient.

## E. Proposed controlled AI tool set

The smallest useful set is nine tools. These are domain tools, not one tool per
HTTP route, and none accepts SQL, arbitrary field names, or unbounded limits.

### 1. `search_players`

- **Purpose:** Resolve a user name and produce a bounded candidate list, or
  build a simple player shortlist using existing filters and sort fields.
- **Input:** `query: str | None`, `team: str | None`,
  `position_group: PositionGroup | None`, `min_pass_attempts: int >= 0`,
  `sort_by: PlayerSortField`, `sort_order: SortOrder`, `limit: int = 10` with a
  hard maximum of 20.
- **Output:** total matching count and a list of stable player identities,
  team, position, samples, reliability and selected headline metric values.
- **Reuse:** player-list query from `routers/players.py::list_players`, the
  `PlayerSortField` allowlist, `_player_filters`, and `PlayerSummary`.
- **Limitations:** name substring matching is not entity resolution; duplicate
  names require the user to choose by ID/team/position. It cannot search players
  outside the loaded product cohort.
- **Use when:** resolving “Xhaka,” finding midfielders, or applying existing
  bounded filters.
- **Do not use when:** the user already supplied a stable player ID or asks a
  methodology-only question.

### 2. `get_player_dossier`

- **Purpose:** Return authoritative, compact player evidence for explanation.
- **Input:** `player_id: int > 0`; optional sections drawn from an enum
  (`passing`, `shooting`, `attacking_impact`, `intelligence`) with a maximum of
  four.
- **Output:** identity plus selected existing response objects:
  `PlayerProfileResponse`, `ShootingProfileResponse`,
  `AttackingProfileResponse`, and `PlayerIntelligenceResponse`.
- **Reuse:** `get_player_profile_or_404`, presenter helpers,
  `get_intelligence_response`, and the shooting/attacking profile queries.
- **Limitations:** missing or ineligible values remain null; the output must
  retain reliability/sample fields. It does not include raw events by default.
- **Use when:** explaining a player's evidence, strengths, profile or archetype.
- **Do not use when:** ranking a whole cohort, asserting future quality, or
  inferring transfer success.

This subsumes a separate `get_player_archetype` tool because archetype evidence
already belongs to the intelligence response.

### 3. `compare_players`

- **Purpose:** Compare exactly two resolved players using the existing product
  comparison contract.
- **Input:** `player_ids: tuple[int, int]`, distinct and positive.
- **Output:** `ComparisonResponse`, including the current passing, attacking and
  intelligence evidence and its sample context.
- **Reuse:** `routers/compare.py::compare_players`, extracted into a service.
- **Limitations:** the current response is not a complete shooting or Role Fit
  comparison. Cross-position percentiles come from different peer groups and
  must be described accordingly.
- **Use when:** “Compare player A and player B.”
- **Do not use when:** the names are unresolved/ambiguous, or when two players
  have not been selected.

### 4. `get_similar_players`

- **Purpose:** Retrieve the persisted, same-position style-neighbor ranking.
- **Input:** `player_id: int > 0`, `limit: int = 6` with maximum 10.
- **Output:** query-player identity, feature contract/version, support warning,
  and ranked `SimilarPlayerResponse` records.
- **Reuse:** `/api/players/{id}/similar` query and response contract.
- **Limitations:** style similarity is not a quality score, future-performance
  forecast, transfer recommendation, or proof that players are interchangeable.
  Scores are meaningful only under the frozen cohort/features and support tier.
- **Use when:** asking who plays similarly to a resolved player.
- **Do not use when:** asking who is “better,” who should be bought, or who fits
  a particular team role; use Role Fit for the last case.

### 5. `get_leaderboard`

- **Purpose:** Produce an eligible, bounded ranking for one existing metric.
- **Input:** `metric: LeaderboardMetric`, optional `position_group` and exact
  `team`, `limit: int = 10` with maximum 25.
- **Output:** metric metadata, reliability rule, rank, identity, value and sample
  evidence in `LeaderboardResponse` form.
- **Reuse:** `routers/leaderboard.py::get_leaderboard` and its allowlisted
  metric-to-column mapping and thresholds.
- **Limitations:** it supports only the existing enum. A high rank in one metric
  is not an overall player rating. Do not silently relax eligibility thresholds.
- **Use when:** “Which defenders have the strongest passing vs. expected?”
- **Do not use when:** the user asks for a compound tactical filter that is not
  represented by current product metrics.

### 6. `get_team_intelligence`

- **Purpose:** Return an observed team style and its available positional roles.
- **Input:** `team_id: int > 0`; optional outfield `position_group` to narrow the
  role output.
- **Output:** `TeamIntelligenceResponse` or the existing role response, including
  support, feature values and warnings.
- **Reuse:** `team_intelligence.py::get_team_intelligence` and
  `get_team_roles`.
- **Limitations:** current production scope is the qualified Bayer Leverkusen
  profile and three outfield roles. It is descriptive of observed event data,
  not a universal tactical model or coaching-intent claim.
- **Use when:** explaining the target team's observed style or role profile.
- **Do not use when:** the user asks about an unsupported team or goalkeeper
  role; return a clear insufficient-evidence result.

### 7. `get_role_fit`

- **Purpose:** Retrieve one player's persisted match to an observed team role.
- **Input:** `player_id: int > 0`, `target_team_id: int > 0`.
- **Output:** `RoleFitResponse`, including position, raw lower-is-closer distance,
  fit index/rank where available, current/external status, support, feature
  differences and warnings.
- **Reuse:** `team_intelligence.py::player_role_fit_response` and
  `/api/players/{id}/role-fit`.
- **Limitations:** Role Fit measures resemblance to an observed positional
  playing style. It is not a probability of transfer success, future performance,
  lineup selection, or adaptation. Current-team rows use leave-self-out roles.
- **Use when:** asking why a specific player fits a supported team role.
- **Do not use when:** the player is unresolved, the team has no profile, or the
  question asks for a candidate list.

### 8. `get_role_recommendations`

- **Purpose:** Return the existing external, same-position shortlist for a
  supported team role.
- **Input:** `team_id: int > 0`, `position_group: DEF | MID | FWD`,
  `limit: int = 10` with maximum 20.
- **Output:** `ScoutingRecommendationsResponse`, including role definition,
  ranked candidates, lower-is-closer distance, fit index, support and warnings.
- **Reuse:** `team_intelligence.py::scouting_recommendations_response` and the
  role-recommendation query in `routers/teams.py`.
- **Limitations:** ranking is style fit within the loaded cohort, not a transfer
  decision. It excludes current-team players from external recommendation rank
  and has no goalkeeper role.
- **Use when:** asking for candidates for a known supported team-position role.
- **Do not use when:** ranking players on general quality, price, availability,
  injury, contract, or future success—none of those data exist here.

### 9. `get_methodology`

- **Purpose:** Retrieve exact production model metadata or cited explanatory
  documentation for one allowlisted topic.
- **Input:** `topic: MethodologyTopic` such as `xpass`, `xg`,
  `possession_value`, `attacking_impact`, `percentiles`, `archetypes`,
  `similarity`, or `role_fit`; optional `question: str` only for later document
  retrieval.
- **Output:** production status, structured metadata when available, cited
  document excerpts, source identifiers and limitations.
- **Reuse:** the three model-information routers/runtime metadata, archetype
  catalogue, and later curated sections from the documentation inventory.
- **Limitations:** experimental results must be labeled experimental. It cannot
  calculate player values or answer entity-specific questions. The Transformer
  must never be described as the production possession-value model.
- **Use when:** asking what a metric means, how a method works, or why a model
  was selected.
- **Do not use when:** a structured player/team tool can supply the requested
  numeric evidence.

Raw pass/shot/action retrieval is intentionally not a general Phase 1 tool. If
evaluation later proves that event-level explanation is necessary, add one
bounded `get_player_event_evidence` tool with an event-type enum, existing
filters, summary fields, a small hard limit, and no free-form ordering.

## F. Proposed structured intent and query plan

The planner output should be a discriminated Pydantic union, not a loose JSON
dictionary. The code below is a proposed contract, not an implementation.

```python
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class IntentKind(StrEnum):
    PLAYER_SEARCH = "player_search"
    PLAYER_PROFILE = "player_profile"
    PLAYER_COMPARISON = "player_comparison"
    SIMILAR_PLAYERS = "similar_players"
    LEADERBOARD = "leaderboard"
    TEAM_ANALYSIS = "team_analysis"
    ROLE_FIT = "role_fit"
    ROLE_RECOMMENDATIONS = "role_recommendations"
    METHODOLOGY = "methodology"


class EntityRef(BaseModel):
    player_id: int | None = Field(default=None, gt=0)
    player_name: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def exactly_one_reference(self):
        if (self.player_id is None) == (self.player_name is None):
            raise ValueError("Provide exactly one player ID or player name")
        return self


class SearchPlayersIntent(BaseModel):
    kind: Literal[IntentKind.PLAYER_SEARCH]
    query: str | None = Field(default=None, min_length=1, max_length=200)
    team: str | None = Field(default=None, min_length=1, max_length=200)
    position_group: PositionGroup | None = None
    min_pass_attempts: int = Field(default=0, ge=0)
    sort_by: PlayerSortField = PlayerSortField.PLAYER_NAME
    sort_order: SortOrder = SortOrder.ASC
    limit: int = Field(default=10, ge=1, le=20)


class PlayerProfileIntent(BaseModel):
    kind: Literal[IntentKind.PLAYER_PROFILE]
    player: EntityRef
    sections: list[PlayerDossierSection] = Field(
        default_factory=lambda: [PlayerDossierSection.INTELLIGENCE],
        min_length=1,
        max_length=4,
    )


class PlayerComparisonIntent(BaseModel):
    kind: Literal[IntentKind.PLAYER_COMPARISON]
    players: tuple[EntityRef, EntityRef]


class SimilarPlayersIntent(BaseModel):
    kind: Literal[IntentKind.SIMILAR_PLAYERS]
    player: EntityRef
    limit: int = Field(default=6, ge=1, le=10)


class LeaderboardIntent(BaseModel):
    kind: Literal[IntentKind.LEADERBOARD]
    metric: LeaderboardMetric
    position_group: PositionGroup | None = None
    team: str | None = Field(default=None, min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=25)


class TeamAnalysisIntent(BaseModel):
    kind: Literal[IntentKind.TEAM_ANALYSIS]
    team_id: int = Field(gt=0)
    position_group: OutfieldPositionGroup | None = None


class RoleFitIntent(BaseModel):
    kind: Literal[IntentKind.ROLE_FIT]
    player: EntityRef
    target_team_id: int = Field(gt=0)


class RoleRecommendationsIntent(BaseModel):
    kind: Literal[IntentKind.ROLE_RECOMMENDATIONS]
    team_id: int = Field(gt=0)
    position_group: OutfieldPositionGroup
    limit: int = Field(default=10, ge=1, le=20)


class MethodologyIntent(BaseModel):
    kind: Literal[IntentKind.METHODOLOGY]
    topic: MethodologyTopic
    question: str | None = Field(default=None, max_length=500)


ScoutIntent = Annotated[
    SearchPlayersIntent
    | PlayerProfileIntent
    | PlayerComparisonIntent
    | SimilarPlayersIntent
    | LeaderboardIntent
    | TeamAnalysisIntent
    | RoleFitIntent
    | RoleRecommendationsIntent
    | MethodologyIntent,
    Field(discriminator="kind"),
]
```

Enums such as `PositionGroup`, `PlayerSortField`, and `LeaderboardMetric` should
reuse `backend/app/schemas.py`. New enums must contain public concepts rather
than SQL column names.

A later multi-tool plan should also be typed and bounded:

```python
class ToolName(StrEnum):
    SEARCH_PLAYERS = "search_players"
    GET_PLAYER_DOSSIER = "get_player_dossier"
    COMPARE_PLAYERS = "compare_players"
    GET_SIMILAR_PLAYERS = "get_similar_players"
    GET_LEADERBOARD = "get_leaderboard"
    GET_TEAM_INTELLIGENCE = "get_team_intelligence"
    GET_ROLE_FIT = "get_role_fit"
    GET_ROLE_RECOMMENDATIONS = "get_role_recommendations"
    GET_METHODOLOGY = "get_methodology"


class ScoutPlan(BaseModel):
    intent: ScoutIntent
    calls: list[ValidatedToolCall] = Field(min_length=1, max_length=6)
    requires_disambiguation: bool = False
    clarification: str | None = Field(default=None, max_length=300)
```

`ValidatedToolCall.arguments` should itself be a discriminated union tied to
the selected tool. Cross-field validators should enforce distinct comparison
IDs, outfield-only role queries, and the relevant result caps. The executor must
reject any undeclared tool, extra field, arbitrary sort expression, or unknown
metric.

Entity resolution is a separate deterministic step. A unique search result can
be promoted to a stable ID; zero results should yield “not found”; multiple
plausible results should request disambiguation rather than guessing.

## G. Proposed backend architecture

```text
backend/app/
├── services/
│   ├── players.py              # extracted ORM queries and response assembly
│   ├── comparisons.py
│   ├── leaderboards.py
│   ├── teams.py
│   └── methodology.py
├── ai/
│   ├── schemas.py              # intents, plans, evidence and answer contracts
│   ├── policy.py               # limits, forbidden claims, production labels
│   ├── entity_resolution.py
│   ├── tools/
│   │   ├── registry.py         # fixed name -> typed executor mapping
│   │   ├── players.py
│   │   ├── teams.py
│   │   └── methodology.py
│   ├── grounding/
│   │   ├── evidence.py         # immutable tool-result ledger
│   │   └── citations.py        # numeric/doc claim provenance
│   ├── retrieval/              # later, only after corpus evaluation
│   │   ├── corpus.py
│   │   ├── index.py
│   │   └── retriever.py
│   ├── orchestration/          # later; LangGraph only if justified
│   │   ├── state.py
│   │   └── graph.py
│   ├── prompts/                # versioned planner/answer policies
│   └── evaluation/
│       ├── cases.py
│       ├── runner.py
│       └── metrics.py
├── routers/
│   └── ai_scout.py             # added only when an LLM-backed API is ready
└── runtime_metadata/            # unchanged existing model source of truth

backend/evals/
├── ai_scout_golden.jsonl
└── methodology_queries.jsonl

docs/ai/
├── methodology_glossary.md
├── limitations.md
└── evaluation_protocol.md
```

Key boundaries:

1. `services/` performs deterministic, read-only domain queries. It may depend
   on models, schemas, presenters and runtime metadata.
2. `ai/tools/` adapts those services to small typed tool contracts. It may not
   import `backend/analytics`.
3. `ai/grounding/` records every returned numeric field and citation before any
   prose is produced.
4. `ai/retrieval/` supplies methodology text only. It never supplies player
   metrics that already exist structurally.
5. `ai/orchestration/` is optional. A direct plan-execute-answer flow should be
   preferred until evaluation shows a need for stateful retries or branching.
6. Training and artifact generation remain in `backend/analytics/` and
   `backend/scripts/`.

No MCP server is needed for the product path. MCP would be useful only if
FootyScout intentionally exposes the same controlled tools to external clients;
it is not required for an in-application AI Scout.

## H. Proposed frontend integration

After the backend and evaluation gates exist:

- add `frontend/app/ai-scout/page.tsx` as the first-class route;
- add a client conversation component under
  `frontend/components/ai-scout/` for input, clarification, evidence panels,
  citations, pending/error states, and bounded conversation history;
- add typed request/response contracts to `frontend/lib/types.ts` and methods to
  `frontend/lib/api.ts` (or a focused `ai-api.ts` if streaming requires it);
- add “AI Scout” to `frontend/components/Navbar.tsx` near Team Intelligence and
  Scouting;
- keep provider API keys exclusively on the backend;
- send only the user's question and an opaque conversation/run ID from the
  browser;
- render computed analytics separately from generated interpretation, with
  visible sample/reliability warnings and source links;
- make clarification a normal state, especially for ambiguous player names;
- expose no client-side generic tool executor.

The first UI should be task-oriented, not an unbounded chat surface. Suggested
starter prompts should map to supported intents, and unsupported requests
should produce a precise capability limitation rather than an improvised answer.

The client must preserve the existing `frontend/lib/api.ts` URL behavior:
browser calls use relative `/api/...`, local server-side calls may use the local
configured backend, and Vercel production SSR prefers the canonical production
project URL.

## I. Evaluation-first architecture

### Golden set

Create 75–100 versioned cases before connecting a production LLM. Store each as
JSONL with:

- stable case ID and category;
- natural-language question;
- expected structured intent;
- expected entity-resolution outcome;
- allowed/required/forbidden tools;
- expected normalized arguments;
- exact numeric facts or database fixture references;
- required methodology source IDs;
- forbidden claims;
- whether clarification or insufficient-evidence behavior is expected;
- maximum agent steps and latency/cost budgets.

Suggested distribution:

| Category | Approximate cases |
|---|---:|
| Entity resolution and player search | 12 |
| Player profiles and comparisons | 15 |
| Similarity and archetypes | 12 |
| Team intelligence, Role Fit and recommendations | 18 |
| Model/metric methodology | 15 |
| Unsupported, ambiguous and adversarial requests | 13 |

Include exact production caveats: similarity is not quality; Role Fit is not
transfer success; archetypes are not ratings; team intelligence is observational;
the Transformer is experimental; and AI-generated numbers may never replace
persisted model outputs.

### Evaluation layers

1. **Schema/unit evaluation:** Pydantic validity, rejected extras, numeric bounds,
   distinct IDs, allowed enums and deterministic service outputs.
2. **Planner evaluation:** intent classification, tool selection, arguments,
   ordering, entity resolution, and clarification accuracy.
3. **Retrieval evaluation:** source recall/precision, production-status accuracy,
   section citation correctness and unsupported-source rejection.
4. **Grounding evaluation:** every numeric claim matches a tool result exactly;
   every methodology claim maps to a cited chunk; no computed value originates
   in generated text.
5. **Answer evaluation:** task completion, faithful comparison direction,
   reliability warnings, limitations, and forbidden-claim violations.
6. **Operational evaluation:** latency percentiles, token usage, estimated cost,
   number of steps/tool calls, errors and retries.

### Metrics and release gates

- intent classification accuracy;
- exact and field-level tool-selection accuracy;
- exact and field-level argument accuracy;
- structured-output validity rate;
- entity-resolution/disambiguation accuracy;
- retrieval recall/precision and citation correctness;
- numeric grounding exact-match rate;
- forbidden-claim violation rate;
- task-completion rate;
- p50/p95 latency;
- prompt/completion tokens and estimated cost;
- mean/max tool-call and agent-step counts.

Recommended production gates are 100% schema validity, 100% numeric grounding
on supported questions, zero forbidden-claim violations in the golden set, and
explicit human review for changes to prompts, tools, source documents, models,
or provider versions. Record baseline and candidate results side-by-side to
catch prompt regressions.

Do not use an LLM judge as the sole ground truth. Intent, tools, arguments,
numbers and citations are programmatically testable. An LLM judge may supplement
human review for tone or completeness only.

### Trace model

Each evaluation and production run should have a `run_id` and capture:

- prompt/policy version and provider model identifier;
- validated intent and plan;
- tool names, sanitized arguments, timing and result hashes;
- evidence IDs attached to answer claims;
- retrieved document IDs/sections/versions;
- token/cost/latency/step counts;
- outcome, validation failure or safe refusal.

Raw secrets must never be logged, and user text retention should be minimized
and documented.

## J. Security and reliability constraints

1. **No generated SQL.** The LLM can select only registered tools with validated
   Pydantic arguments. Tool implementations use predefined SQLAlchemy queries.
2. **Read-only data access.** The AI database identity should be read-only if a
   separate production credential is introduced. No tool performs inserts,
   updates, migrations, model training or snapshot loading.
3. **Hard bounds.** Apply per-tool result limits, input-length limits, a maximum
   of six tool calls initially, request timeouts, token budgets, and provider
   retry limits.
4. **Stable entity IDs.** Resolve names to IDs deterministically; request user
   clarification for ambiguity. Never guess a player/team.
5. **Typed allowlists.** Metrics, positions, sort fields, sections, methodology
   topics and filters are enums. Reject extra arguments.
6. **Evidence ledger.** Numeric claims must point to a returned tool field.
   Methodology claims must point to a versioned document source.
7. **Data/interpretation separation.** UI and response schemas distinguish
   `computed_evidence`, `methodology_sources`, `interpretation`, `limitations`
   and `insufficient_evidence`.
8. **Reliability preserved.** Never coerce null/ineligible metrics to zero or
   omit sample thresholds, peer counts, limited tiers, leave-self-out status, or
   model-production status.
9. **Prompt-injection resistance.** Retrieved documents and database strings are
   evidence, never instructions. System policy and tool registry cannot be
   changed by user or retrieved content.
10. **Secret isolation.** Provider and database credentials remain backend-only,
    are read from environment variables, and are excluded from prompts, tool
    outputs, client bundles and logs.
11. **Graceful failure.** Define typed `not_found`, `ambiguous`, `unsupported`,
    `insufficient_evidence`, `tool_error` and `budget_exceeded` outcomes.
12. **No unsupported extrapolation.** The system cannot claim transfer value,
    availability, wages, injuries, future performance, causal tactics, or player
    quality when FootyScout has no such evidence.
13. **Abuse and cost controls.** Before a public endpoint, add authentication or
    an anonymous quota, rate limiting, concurrency limits and request-size caps.
14. **Version pinning.** Answer traces record the analytics snapshot, runtime
    metadata version, retrieval corpus version and prompt version.

## K. Phased implementation plan

### Phase 1 — deterministic foundation and evaluation fixtures

No LLM, RAG, LangGraph, MCP, provider SDK or frontend page.

1. Extract router-owned read queries into `backend/app/services/` without
   changing endpoint schemas or behavior.
2. Add `backend/app/ai/schemas.py` containing the typed intent, entity reference,
   evidence, tool input/output and safe-error contracts.
3. Add a fixed in-process tool registry and deterministic executors that call
   the shared services.
4. Add policy constants for result/step limits and required disclaimers.
5. Create 75–100 golden-case skeletons, beginning with fully asserted cases for
   every tool, ambiguity path and forbidden claim.
6. Add unit/contract tests that compare REST and tool outputs from identical
   database fixtures, proving there is one calculation/query path.
7. Do not expose a public generic tool-execution endpoint. Test tools directly.

Exit criteria: all existing API tests remain unchanged and green; tool outputs
are deterministic and typed; no analytics imports appear under `app/ai`; and
the golden corpus covers every supported/unsupported intent.

### Phase 2 — structured natural-language planning

1. Add one backend-only provider SDK (likely the official `openai` package) and
   use structured output against `ScoutIntent`/`ScoutPlan`.
2. Implement entity-resolution and clarification before execution.
3. Execute the bounded plan, create an immutable evidence ledger, and produce a
   structured answer object.
4. Run the planner golden set offline in CI or a separately authorized eval job;
   keep ordinary unit tests provider-free.

Start with a direct state machine. Do not add LangGraph unless the evaluated
workflow needs durable multi-step state, conditional recovery, or human-in-the-
loop branches that are materially clearer than ordinary Python.

### Phase 3 — curated methodology retrieval

1. Normalize the approved documentation into versioned section chunks.
2. Add deterministic lexical/PostgreSQL full-text retrieval and citation output.
3. Evaluate retrieval on methodology cases.
4. Consider embeddings/pgvector only if measured retrieval failures justify the
   operational cost.

### Phase 4 — grounded synthesis and controlled orchestration

1. Generate explanations strictly from the evidence ledger and cited chunks.
2. Add claim-to-source validation and forbidden-claim checks.
3. Introduce LangGraph only if Phase 2/3 evidence shows a need for bounded
   routing, retries or clarification state.
4. Add run tracing, prompt versions and regression comparisons.

### Phase 5 — first-class frontend

1. Add the `/ai-scout` page, navigation item and typed client.
2. Render clarifications, computed evidence, interpretation, citations,
   limitations, loading and safe-error states distinctly.
3. Add accessibility, responsive behavior and client tests.
4. Keep all provider access server-side.

### Phase 6 — deployment hardening and monitored release

1. Add rate limits, quotas/auth policy, timeouts, concurrency controls and
   retention policy.
2. Run the full golden suite against the deployed candidate.
3. Establish latency, token, cost, tool-error and safety dashboards.
4. Canary the feature, review traces, and require evaluation approval for prompt,
   tool, model or corpus changes.

## Exact anticipated repository changes

### Files to create

Phase 1:

- `backend/app/services/__init__.py`
- `backend/app/services/players.py`
- `backend/app/services/comparisons.py`
- `backend/app/services/leaderboards.py`
- `backend/app/services/teams.py`
- `backend/app/services/methodology.py`
- `backend/app/ai/__init__.py`
- `backend/app/ai/schemas.py`
- `backend/app/ai/policy.py`
- `backend/app/ai/entity_resolution.py`
- `backend/app/ai/tools/__init__.py`
- `backend/app/ai/tools/registry.py`
- `backend/app/ai/tools/players.py`
- `backend/app/ai/tools/teams.py`
- `backend/app/ai/tools/methodology.py`
- `backend/tests/ai/test_schemas.py`
- `backend/tests/ai/test_entity_resolution.py`
- `backend/tests/ai/test_tools.py`
- `backend/tests/ai/test_policy.py`
- `backend/evals/ai_scout_golden.jsonl`
- `docs/ai/evaluation_protocol.md`
- `docs/ai/limitations.md`

Later phases:

- `backend/app/ai/grounding/evidence.py`
- `backend/app/ai/grounding/citations.py`
- `backend/app/ai/retrieval/corpus.py`
- `backend/app/ai/retrieval/index.py`
- `backend/app/ai/retrieval/retriever.py`
- `backend/app/ai/orchestration/state.py`
- `backend/app/ai/orchestration/graph.py` only if justified
- `backend/app/ai/prompts/*`
- `backend/app/ai/evaluation/cases.py`
- `backend/app/ai/evaluation/runner.py`
- `backend/app/ai/evaluation/metrics.py`
- `backend/app/routers/ai_scout.py`
- `frontend/app/ai-scout/page.tsx`
- `frontend/components/ai-scout/*`
- corresponding backend/frontend tests and a curated methodology glossary

### Files to modify

Phase 1 service extraction:

- `backend/app/routers/players.py`
- `backend/app/routers/compare.py`
- `backend/app/routers/leaderboard.py`
- `backend/app/routers/teams.py`

The changes should be mechanical delegation to shared services with the public
API unchanged.

Later phases only:

- `backend/app/main.py` to include the AI router;
- `backend/app/config.py` for backend-only provider/evaluation settings;
- `backend/pyproject.toml` for only the dependencies justified by that phase;
- `frontend/lib/types.ts` and `frontend/lib/api.ts` for typed AI contracts;
- `frontend/components/Navbar.tsx` for the new route;
- `.github/workflows/ci.yml` only if a separate opt-in evaluation job is added;
- `README.md` and analytics documentation to describe the released capability.

Existing analytics files, generated model artifacts, PostgreSQL snapshot data,
and runtime model metadata should not be modified by this work.

## Eventual dependencies

- **Phase 1:** none; FastAPI, Pydantic, SQLAlchemy and pytest already cover the
  deterministic contracts and tests.
- **Phase 2:** one official provider SDK, likely `openai`, in the backend runtime
  dependency set. Avoid adding LangChain solely as an SDK wrapper.
- **Phase 3:** initially none if using a small curated lexical index or PostgreSQL
  full-text search. Consider `pgvector` and an embedding SDK only after measured
  need.
- **Phase 4:** `langgraph` only if evaluated stateful orchestration requirements
  justify it. Do not add it for a linear plan-execute-answer path.
- **Observability:** prefer existing platform logs and a small structured trace
  schema first. Add OpenTelemetry or a hosted tracing SDK only after defining
  retention, privacy and operational requirements.

No PyTorch, XGBoost, pandas or training dependency should be added to the
production request path for AI Scout.

## Database migrations

- **Phase 1:** none.
- Stateless Phase 2 can also require no migration.
- A later observability/feedback phase may add `ai_runs`, `ai_tool_calls` and
  optional `ai_feedback` tables. Store sanitized structured traces and hashes,
  not secrets or unrestricted prompt content.
- A vector column/table and `pgvector` extension should be a separate, optional
  migration only if retrieval evaluation justifies semantic search.
- Conversation persistence should not be added by default; define retention and
  user privacy requirements first.

## Risks and architectural issues found

1. **Router-owned queries:** several core queries are embedded in routers. Tool
   code would otherwise duplicate SQL or call internal HTTP, so service
   extraction is the most important Phase 1 change.
2. **Split sources of truth:** player/team analytics are in PostgreSQL, model
   metadata is packaged JSON, and methodology is in Markdown. Evidence records
   need explicit source type and version.
3. **Terminology drift:** internal field names retain technical terms while the
   frontend deliberately uses simpler product copy. AI answers need a governed
   terminology map without renaming API fields.
4. **Entity ambiguity:** substring player search can return multiple names. A
   future planner must not choose silently.
5. **Sparse reliability:** null percentiles, position peer thresholds, limited
   match support and style-profile eligibility are meaningful, not missing data
   to be imputed.
6. **Directionality risk:** similarity index is higher-is-closer while raw RMS
   distance and Role Fit distance are lower-is-closer. Tools and explanations
   must preserve the correct direction.
7. **Narrow team scope:** production team intelligence currently covers one
   qualified team and three outfield role profiles. The AI must not imply
   league-wide team-role coverage.
8. **Comparison scope:** the comparison endpoint does not include every shooting
   or Role Fit field. A response must not claim completeness beyond its schema.
9. **Experiment/production confusion:** GRU and Transformer experiments exist,
   but XGBoost remains the production possession-value model.
10. **No public AI abuse controls yet:** authentication, quotas, rate limits,
    provider timeouts and cost budgets are required before a public LLM endpoint.
11. **Test working-directory sensitivity:** repository-root `.env` variables can
    be interpreted by backend settings when commands are launched from the root.
    Backend commands should continue running from `backend/`, as CI does.
12. **Document freshness:** methodology can become stale after an analytics
    change. Corpus versions and review dates must be tied to releases.

## Recommended next implementation

Proceed only with Phase 1 after review. Specifically, extract the existing
read-only domain queries into shared services, add the typed intent/tool/evidence
contracts, implement the nine deterministic tool adapters, and establish golden
fixtures and parity tests. Do not yet add an LLM, retrieval index, LangGraph,
MCP, a public AI endpoint, or frontend UI.

This order proves that AI Scout can retrieve every allowed fact from the same
production logic as the existing application before any generated language is
introduced.
