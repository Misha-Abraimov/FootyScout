"""Code-level limits and governed domain caveats for deterministic AI tools."""

PLAYER_SEARCH_MAX_RESULTS = 20
SIMILAR_PLAYERS_MAX_RESULTS = 10
LEADERBOARD_MAX_RESULTS = 25
RECOMMENDATIONS_MAX_RESULTS = 20
MAX_TOOL_CALLS = 6
MAX_ENTITY_STRING_LENGTH = 200
MAX_METHODOLOGY_QUESTION_LENGTH = 500

SUPPORTED_METHODOLOGY_TOPICS = frozenset(
    {
        "xpass",
        "xg",
        "possession_value",
        "attacking_impact",
        "player_profiles",
        "percentiles",
        "archetypes",
        "similarity",
        "team_intelligence",
        "role_fit",
        "role_recommendations",
    }
)

SIMILARITY_LIMITATIONS = (
    "Similarity describes same-position playing style, not player quality.",
    "Similarity does not predict future performance or transfer success.",
    "Similarity is not team-role fit.",
)

ROLE_FIT_LIMITATIONS = (
    "Role Fit supports eligible outfield positions only.",
    "Role Fit does not predict transfer success or future performance.",
    "Role Fit is not lineup selection or a tactical guarantee.",
)

ARCHETYPE_LIMITATIONS = ("Archetypes group playing style; they are not player-quality ratings.",)

TEAM_INTELLIGENCE_LIMITATIONS = (
    "Team intelligence describes observed event data, not coaching intent.",
    "Current production team-intelligence scope is limited to qualified teams.",
)

POSSESSION_VALUE_LIMITATIONS = (
    "XGBoost is the current production possession-value model.",
    "The GRU and PyTorch causal Transformer are offline experiments, not production models.",
)

METHODOLOGY_SOURCE_REFERENCES = {
    "xpass": ("backend/app/runtime_metadata/pass_model_metadata.json", "production metadata"),
    "xg": ("backend/app/runtime_metadata/xg_model_metadata.json", "production metadata"),
    "possession_value": (
        "backend/app/runtime_metadata/action_value_model_metadata.json",
        "production possession-value metadata",
    ),
    "attacking_impact": (
        "backend/analytics/README.md",
        "Possession value and attacking impact",
    ),
    "player_profiles": (
        "backend/analytics/README.md",
        "Unified player intelligence profiles",
    ),
    "percentiles": (
        "backend/analytics/README.md",
        "Unified player intelligence and same-position percentiles",
    ),
    "archetypes": (
        "backend/analytics/README.md",
        "Production player-style archetypes",
    ),
    "similarity": (
        "backend/analytics/README.md",
        "Same-position player similarity",
    ),
    "team_intelligence": (
        "backend/analytics/README.md",
        "Team intelligence and positional-role profiles",
    ),
    "role_fit": (
        "backend/analytics/README.md",
        "Team intelligence and Role Fit",
    ),
    "role_recommendations": (
        "backend/analytics/README.md",
        "Scouting recommendations",
    ),
}
