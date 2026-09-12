"""Frozen V4 team-intelligence feature contract used by the production API.

The analytics implementation owns how these values are calculated. Keeping the
published names here prevents the request-time API bundle from importing the
scientific Python stack solely to serialize already-persisted database values.
"""

FIT_FEATURES = (
    "expected_completion_rate",
    "pressure_pass_rate",
    "progressive_pass_rate",
    "long_pass_rate",
    "positive_forward_distance_per_100_passes",
    "carry_share_of_actions",
)

DESCRIPTIVE_FEATURES = (
    *FIT_FEATURES,
    "average_forward_distance",
    "final_third_entries_per_100_passes",
    "progressive_carry_rate",
    "progressive_action_rate",
    "pressure_action_rate",
    "shots_per_match",
    "xg_per_shot",
    "xg_per_match",
    "attacking_value_per_100_actions",
)
