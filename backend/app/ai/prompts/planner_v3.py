"""Version 3 semantic refinements for bounded FootyScout planning."""

from app.ai.prompts.planner_v2 import PLANNER_PROMPT as PLANNER_V2_PROMPT

PLANNER_PROMPT_VERSION = "planner-v3"

PLANNER_PROMPT = PLANNER_V2_PROMPT + """
The following intent and ambiguity refinements are authoritative. The primary intent is
the user's final analytical goal, not the first supporting operation needed to reach it.
A request for a player's profile remains player_profile, a comparison remains
player_comparison, and a similarity request remains similar_players even when a name
must first be searched and resolved. Use player_search only when returning matching
players is itself the user's final goal.

For player_search, partial names, case differences, and multiple matching players are
normal results. Preserve the user's raw search string and do not request clarification
merely because a search may return several players. For player_profile,
player_comparison, similar_players, role_fit, and other analyses requiring one specific
player, preserve the final analytical intent and let deterministic entity resolution
request clarification when a reference is ambiguous.

If the user does not explicitly state a numeric result count, emit the existing 0
sentinel in limit. Do not infer or copy a default count from another intent.
Deterministic code owns intent-specific defaults and hard maximums. If the user does
state a result count, copy that count into limit.
"""
