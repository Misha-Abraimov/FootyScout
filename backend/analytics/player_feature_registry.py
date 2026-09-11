"""Authoritative V3 player-intelligence feature contract.

Style describes observed action tendencies; performance describes observed execution or
model-derived output. A high style value is not automatically better.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MIN_POSITION_PEERS = 10


@dataclass(frozen=True)
class PlayerFeature:
    feature_name: str
    label: str
    family: str
    source: str
    required_sample_type: str
    sample_column: str
    minimum_sample: int
    unit: str
    directionality: str
    eligible_for_style_clustering: bool
    eligible_for_style_similarity: bool
    eligible_for_ability_similarity: bool
    radar: bool = False
    stability_note: str | None = None


PASS = 100
SUBSET = 20
SHOT = 20
ACTION = 63
VALUE_PASS = 36
CARRY = 29


FEATURE_REGISTRY = (
    PlayerFeature("expected_completion_rate", "Expected completion", "style", "player_profiles", "passes", "pass_attempts", PASS, "rate", "higher means safer attempted passes", True, True, False),
    PlayerFeature("pressure_pass_rate", "Under-pressure pass rate", "style", "player_profiles", "passes", "pass_attempts", PASS, "rate", "higher means more passes attempted under pressure", True, True, False, radar=True),
    PlayerFeature("progressive_pass_rate", "Progressive pass rate", "style", "player_profiles", "passes", "pass_attempts", PASS, "rate", "higher means more progressive passes attempted", True, True, False, radar=True),
    PlayerFeature("long_pass_rate", "Long-pass rate", "style", "player_profiles (derived)", "passes", "pass_attempts", PASS, "rate", "higher means more long passes attempted", True, True, False, radar=True),
    PlayerFeature("average_forward_distance", "Average forward distance", "style", "player_profiles", "passes", "pass_attempts", PASS, "distance", "higher means more forward distance per attempted pass", False, True, False),
    PlayerFeature("positive_forward_distance_per_100_passes", "Positive forward distance / 100 passes", "style", "player_profiles", "passes", "pass_attempts", PASS, "distance_per_100", "higher means more positive progression volume", True, True, False),
    PlayerFeature("final_third_entries_per_100_passes", "Final-third entries / 100 passes", "style", "player_profiles", "passes", "pass_attempts", PASS, "per_100_passes", "higher means more final-third entries", True, True, False),
    PlayerFeature("carry_share_of_actions", "Carry share of actions", "style", "player_attacking_profiles (derived)", "actions", "actions", ACTION, "rate", "higher means carries form a larger action share", True, True, False),
    PlayerFeature("progressive_carry_rate", "Progressive carry rate", "style", "attacking_actions (derived)", "carries", "carries", CARRY, "rate", "higher means more carries are progressive", False, True, False, radar=True),
    PlayerFeature("progressive_action_rate", "Progressive action rate", "style", "attacking_actions (derived)", "actions", "actions", ACTION, "rate", "higher means more actions are progressive", False, True, False),
    PlayerFeature("pressure_action_rate", "Under-pressure action rate", "style", "attacking_actions (derived)", "actions", "actions", ACTION, "rate", "higher means more actions occur under pressure", False, True, False),
    PlayerFeature("shots_per_match_observed", "Shots / observed match", "style", "player_shooting_profiles (derived)", "shots", "shots", SHOT, "per_match", "higher means more shooting involvement in observed matches", False, True, False, radar=True),
    PlayerFeature("xg_per_shot", "xG / shot", "style", "player_shooting_profiles", "shots", "shots", SHOT, "rate", "higher means higher-quality modeled shot opportunities", False, True, False),
    PlayerFeature("completion_above_expected_pp", "Completion above expected", "performance", "player_profiles", "passes", "pass_attempts", PASS, "percentage_points", "higher means better observed execution versus xPass", False, False, True, radar=True),
    PlayerFeature("pressure_above_expected_pp", "Under-pressure completion above expected", "performance", "player_profiles", "pressure passes", "pressure_attempts", SUBSET, "percentage_points", "higher means better observed under-pressure execution", False, False, True),
    PlayerFeature("progressive_above_expected_pp", "Progressive completion above expected", "performance", "player_profiles", "progressive passes", "progressive_attempts", SUBSET, "percentage_points", "higher means better observed progressive-pass execution", False, False, True),
    PlayerFeature("long_pass_above_expected_pp", "Long-pass completion above expected", "performance", "player_profiles", "long passes", "long_pass_attempts", SUBSET, "percentage_points", "higher means better observed long-pass execution", False, False, True),
    PlayerFeature("goals_minus_xg", "Goals above expected", "performance", "player_shooting_profiles", "shots", "shots", SHOT, "goals", "higher means more goals than modeled expectation", False, False, True),
    PlayerFeature("attacking_value_per_100_actions", "Attacking value / 100 actions", "performance", "player_attacking_profiles", "actions", "actions", ACTION, "value_per_100", "higher means more observed possession-value gain", False, False, False, stability_note="Weak split-half stability in the current 34-match product cohort."),
    PlayerFeature("pass_value_per_100_passes", "Pass value / 100 passes", "performance", "player_attacking_profiles", "passes", "passes", VALUE_PASS, "value_per_100", "higher means more observed pass value", False, False, False, stability_note="Weak split-half stability in the current 34-match product cohort."),
    PlayerFeature("carry_value_per_100_carries", "Carry value / 100 carries", "performance", "player_attacking_profiles", "carries", "carries", CARRY, "value_per_100", "higher means more observed carry value", False, False, True, radar=True, stability_note="Moderate split-half stability; descriptive rather than latent ability."),
    PlayerFeature("progressive_value_per_100_actions", "Progressive value / 100 actions", "performance", "player_attacking_profiles", "actions", "actions", ACTION, "value_per_100", "higher means more observed value from progressive actions", False, False, False, stability_note="Weak split-half stability in the current 34-match product cohort."),
    PlayerFeature("pressure_value_per_100_actions", "Under-pressure value / 100 actions", "performance", "player_attacking_profiles", "actions", "actions", ACTION, "value_per_100", "higher means more observed value from under-pressure actions", False, False, False, stability_note="Weak split-half stability in the current 34-match product cohort."),
)

FEATURE_BY_NAME = {feature.feature_name: feature for feature in FEATURE_REGISTRY}
STYLE_FEATURES = [f.feature_name for f in FEATURE_REGISTRY if f.family == "style"]
PERFORMANCE_FEATURES = [f.feature_name for f in FEATURE_REGISTRY if f.family == "performance"]
STYLE_CLUSTERING_FEATURES = [
    f.feature_name for f in FEATURE_REGISTRY if f.eligible_for_style_clustering
]
RADAR_FEATURES = [f.feature_name for f in FEATURE_REGISTRY if f.radar]


def registry_records() -> list[dict[str, object]]:
    return [asdict(feature) for feature in FEATURE_REGISTRY]


def validate_registry() -> None:
    names = [feature.feature_name for feature in FEATURE_REGISTRY]
    if len(names) != len(set(names)):
        raise ValueError("Player feature names must be unique")
    if {feature.family for feature in FEATURE_REGISTRY} != {"style", "performance"}:
        raise ValueError("Player features must be explicitly classified")
