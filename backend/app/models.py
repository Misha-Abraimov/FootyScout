"""SQLAlchemy persistence models for FootyScout analytics outputs."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Player(Base):
    """One unique StatsBomb player and current display metadata."""

    __tablename__ = "players"
    __table_args__ = (
        CheckConstraint("matches_observed >= 0", name="ck_players_matches_nonnegative"),
        CheckConstraint("pass_attempts >= 0", name="ck_players_attempts_nonnegative"),
        CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_players_position_group",
        ),
        Index("ix_players_player_name", "player_name"),
        Index("ix_players_team_name", "team_name"),
        Index("ix_players_position_group", "position_group"),
    )

    player_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    position: Mapped[str] = mapped_column(String(100), nullable=False)
    position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    overall_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)

    profile: Mapped[PlayerProfile] = relationship(
        back_populates="player",
        cascade="all, delete-orphan",
        uselist=False,
    )
    passes: Mapped[list[Pass]] = relationship(back_populates="player")
    shots: Mapped[list[Shot]] = relationship(back_populates="player")
    shooting_profile: Mapped[PlayerShootingProfile | None] = relationship(
        back_populates="player",
        cascade="all, delete-orphan",
        uselist=False,
    )
    attacking_actions: Mapped[list[AttackingAction]] = relationship(back_populates="player")
    attacking_profile: Mapped[PlayerAttackingProfile | None] = relationship(
        back_populates="player", cascade="all, delete-orphan", uselist=False
    )
    intelligence_profile: Mapped[PlayerIntelligenceProfile | None] = relationship(
        back_populates="player", cascade="all, delete-orphan", uselist=False
    )
    percentiles: Mapped[list[PlayerPercentile]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )
    archetype: Mapped[PlayerArchetype | None] = relationship(
        back_populates="player", cascade="all, delete-orphan", uselist=False
    )
    role_fits: Mapped[list[PlayerRoleFit]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )
    similarities: Mapped[list[PlayerSimilarity]] = relationship(
        back_populates="player",
        cascade="all, delete-orphan",
        foreign_keys="PlayerSimilarity.player_id",
    )
    similar_to: Mapped[list[PlayerSimilarity]] = relationship(
        back_populates="similar_player",
        cascade="all, delete-orphan",
        foreign_keys="PlayerSimilarity.similar_player_id",
    )


class PlayerProfile(Base):
    """One-to-one analytical passing profile for a player."""

    __tablename__ = "player_profiles"
    __table_args__ = (
        CheckConstraint(
            "actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_actual_rate",
        ),
        CheckConstraint(
            "expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_expected_rate",
        ),
        CheckConstraint(
            "pressure_actual_completion_rate IS NULL OR "
            "pressure_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_pressure_actual_rate",
        ),
        CheckConstraint(
            "pressure_expected_completion_rate IS NULL OR "
            "pressure_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_pressure_expected_rate",
        ),
        CheckConstraint(
            "progressive_actual_completion_rate IS NULL OR "
            "progressive_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_actual_rate",
        ),
        CheckConstraint(
            "progressive_expected_completion_rate IS NULL OR "
            "progressive_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_expected_rate",
        ),
        CheckConstraint(
            "long_pass_actual_completion_rate IS NULL OR "
            "long_pass_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_long_actual_rate",
        ),
        CheckConstraint(
            "long_pass_expected_completion_rate IS NULL OR "
            "long_pass_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_long_expected_rate",
        ),
        CheckConstraint(
            "pressure_pass_rate BETWEEN 0 AND 1",
            name="ck_profiles_pressure_pass_rate",
        ),
        CheckConstraint(
            "progressive_pass_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_pass_rate",
        ),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"),
        primary_key=True,
    )
    passes_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_completion_rate: Mapped[float] = mapped_column(Float, nullable=False)
    expected_completions: Mapped[float] = mapped_column(Float, nullable=False)
    expected_completion_rate: Mapped[float] = mapped_column(Float, nullable=False)
    completions_above_expected: Mapped[float] = mapped_column(Float, nullable=False)
    completion_above_expected_pp: Mapped[float] = mapped_column(Float, nullable=False)

    pressure_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    pressure_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    pressure_actual_completion_rate: Mapped[float | None] = mapped_column(Float)
    pressure_expected_completion_rate: Mapped[float | None] = mapped_column(Float)
    pressure_completions_above_expected: Mapped[float | None] = mapped_column(Float)
    pressure_above_expected_pp: Mapped[float | None] = mapped_column(Float)
    pressure_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)

    progressive_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    progressive_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    progressive_actual_completion_rate: Mapped[float | None] = mapped_column(Float)
    progressive_expected_completion_rate: Mapped[float | None] = mapped_column(Float)
    progressive_completions_above_expected: Mapped[float | None] = mapped_column(Float)
    progressive_above_expected_pp: Mapped[float | None] = mapped_column(Float)
    progressive_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)

    long_pass_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    long_pass_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    long_pass_actual_completion_rate: Mapped[float | None] = mapped_column(Float)
    long_pass_expected_completion_rate: Mapped[float | None] = mapped_column(Float)
    long_pass_completions_above_expected: Mapped[float | None] = mapped_column(Float)
    long_pass_above_expected_pp: Mapped[float | None] = mapped_column(Float)

    average_forward_distance: Mapped[float] = mapped_column(Float, nullable=False)
    net_forward_distance_per_100_passes: Mapped[float] = mapped_column(Float, nullable=False)
    positive_forward_distance_per_100_passes: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    final_third_entries: Mapped[int] = mapped_column(Integer, nullable=False)
    final_third_entries_per_100_passes: Mapped[float] = mapped_column(Float, nullable=False)

    pressure_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    progressive_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    long_pass_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)

    player: Mapped[Player] = relationship(back_populates="profile")


class PlayerSimilarity(Base):
    """Directed V3.3B playing-style recommendation with sample support."""

    __tablename__ = "player_similarities"
    __table_args__ = (
        CheckConstraint("player_id <> similar_player_id", name="ck_similarities_not_self"),
        CheckConstraint("rms_distance >= 0", name="ck_similarities_distance_nonnegative"),
        CheckConstraint(
            "similarity_score BETWEEN 0 AND 100",
            name="ck_similarities_score_range",
        ),
        CheckConstraint("rank >= 1", name="ck_similarities_rank_positive"),
        CheckConstraint(
            "query_matches_observed >= 0 AND candidate_matches_observed >= 0 "
            "AND pair_support_matches >= 0",
            name="ck_similarities_match_counts_nonnegative",
        ),
        CheckConstraint(
            "pair_support_matches <= query_matches_observed "
            "AND pair_support_matches <= candidate_matches_observed",
            name="ck_similarities_pair_support_weaker",
        ),
        CheckConstraint(
            "sample_support IN ('limited', 'higher')",
            name="ck_similarities_sample_support",
        ),
        UniqueConstraint("player_id", "rank", name="uq_similarities_player_rank"),
        Index("ix_player_similarities_player_id", "player_id"),
        Index("ix_player_similarities_similar_player_id", "similar_player_id"),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"),
        primary_key=True,
    )
    similar_player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    rms_distance: Mapped[float] = mapped_column(Float, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    same_position_group: Mapped[bool] = mapped_column(Boolean, nullable=False)
    position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    similar_position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    closest_feature_1: Mapped[str] = mapped_column(String(100), nullable=False)
    closest_feature_2: Mapped[str] = mapped_column(String(100), nullable=False)
    closest_feature_3: Mapped[str] = mapped_column(String(100), nullable=False)
    query_matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    pair_support_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_support: Mapped[str] = mapped_column(String(20), nullable=False)
    sample_support_explanation: Mapped[str] = mapped_column(String(300), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)
    distance_contributions: Mapped[str] = mapped_column(Text, nullable=False)

    player: Mapped[Player] = relationship(
        back_populates="similarities",
        foreign_keys=[player_id],
    )
    similar_player: Mapped[Player] = relationship(
        back_populates="similar_to",
        foreign_keys=[similar_player_id],
    )


class Pass(Base):
    """One pass with historical out-of-fold expected completion."""

    __tablename__ = "passes"
    __table_args__ = (
        CheckConstraint(
            "expected_completion BETWEEN 0 AND 1",
            name="ck_passes_expected_completion",
        ),
        CheckConstraint("fold >= 1", name="ck_passes_fold_positive"),
        Index("ix_passes_player_id", "player_id"),
        Index("ix_passes_match_id", "match_id"),
        Index("ix_passes_player_match", "player_id", "match_id"),
    )

    pass_index: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    match_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id", ondelete="SET NULL"),
        nullable=True,
    )
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    position: Mapped[str] = mapped_column(String(100), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_completion: Mapped[float] = mapped_column(Float, nullable=False)
    fold: Mapped[int] = mapped_column(Integer, nullable=False)

    start_x: Mapped[float] = mapped_column(Float, nullable=False)
    start_y: Mapped[float] = mapped_column(Float, nullable=False)
    end_x: Mapped[float] = mapped_column(Float, nullable=False)
    end_y: Mapped[float] = mapped_column(Float, nullable=False)
    pass_length: Mapped[float] = mapped_column(Float, nullable=False)
    pass_angle: Mapped[float] = mapped_column(Float, nullable=False)
    forward_distance: Mapped[float] = mapped_column(Float, nullable=False)
    lateral_distance: Mapped[float] = mapped_column(Float, nullable=False)
    distance_to_goal_before: Mapped[float] = mapped_column(Float, nullable=False)
    distance_to_goal_after: Mapped[float] = mapped_column(Float, nullable=False)
    distance_toward_goal: Mapped[float] = mapped_column(Float, nullable=False)

    under_pressure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    progressive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pass_height: Mapped[str] = mapped_column(String(100), nullable=False)
    body_part: Mapped[str] = mapped_column(String(100), nullable=False)
    pass_type: Mapped[str] = mapped_column(String(100), nullable=False)
    start_zone: Mapped[str] = mapped_column(String(100), nullable=False)
    end_zone: Mapped[str] = mapped_column(String(100), nullable=False)

    player: Mapped[Player | None] = relationship(back_populates="passes")


class Shot(Base):
    """One product-cohort shot with a production xG estimate when eligible."""

    __tablename__ = "shots"
    __table_args__ = (
        CheckConstraint(
            "expected_goal IS NULL OR expected_goal BETWEEN 0 AND 1",
            name="ck_shots_expected_goal",
        ),
        CheckConstraint("goal IN (false, true)", name="ck_shots_goal_boolean"),
        Index("ix_shots_player_id", "player_id"),
        Index("ix_shots_match_id", "match_id"),
        Index("ix_shots_player_match", "player_id", "match_id"),
    )

    shot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    match_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id", ondelete="SET NULL"), nullable=True
    )
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    period: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    second: Mapped[int] = mapped_column(Integer, nullable=False)
    start_x: Mapped[float] = mapped_column(Float, nullable=False)
    start_y: Mapped[float] = mapped_column(Float, nullable=False)
    distance: Mapped[float] = mapped_column(Float, nullable=False)
    angle: Mapped[float] = mapped_column(Float, nullable=False)
    goal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_goal: Mapped[float | None] = mapped_column(Float)
    body_part: Mapped[str] = mapped_column(String(100), nullable=False)
    shot_type: Mapped[str] = mapped_column(String(100), nullable=False)
    technique: Mapped[str] = mapped_column(String(100), nullable=False)
    play_pattern: Mapped[str] = mapped_column(String(100), nullable=False)
    under_pressure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    first_time: Mapped[bool] = mapped_column(Boolean, nullable=False)
    one_on_one: Mapped[bool] = mapped_column(Boolean, nullable=False)
    open_goal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    penalty: Mapped[bool] = mapped_column(Boolean, nullable=False)
    penalty_shootout: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)

    player: Mapped[Player | None] = relationship(back_populates="shots")


class PlayerShootingProfile(Base):
    """Non-penalty expected-goals aggregate for one product-cohort player."""

    __tablename__ = "player_shooting_profiles"
    __table_args__ = (
        CheckConstraint("shots >= 0", name="ck_shooting_profiles_shots"),
        CheckConstraint("goals >= 0", name="ck_shooting_profiles_goals"),
        CheckConstraint("total_xg >= 0", name="ck_shooting_profiles_total_xg"),
        CheckConstraint("xg_per_shot BETWEEN 0 AND 1", name="ck_shooting_profiles_xg_per_shot"),
        CheckConstraint("goals_per_shot BETWEEN 0 AND 1", name="ck_shooting_profiles_goals_per_shot"),
        CheckConstraint("matches_observed >= 0", name="ck_shooting_profiles_matches"),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    shots: Mapped[int] = mapped_column(Integer, nullable=False)
    goals: Mapped[int] = mapped_column(Integer, nullable=False)
    total_xg: Mapped[float] = mapped_column(Float, nullable=False)
    xg_per_shot: Mapped[float] = mapped_column(Float, nullable=False)
    goals_minus_xg: Mapped[float] = mapped_column(Float, nullable=False)
    goals_per_shot: Mapped[float] = mapped_column(Float, nullable=False)
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    shooting_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)

    player: Mapped[Player] = relationship(back_populates="shooting_profile")


class AttackingAction(Base):
    """One Bundesliga pass/carry with held-out state values before and after."""

    __tablename__ = "attacking_actions"
    __table_args__ = (
        CheckConstraint("state_value_before >= 0", name="ck_actions_before_nonnegative"),
        CheckConstraint("state_value_after >= 0", name="ck_actions_after_nonnegative"),
        CheckConstraint(
            "expected_completion IS NULL OR expected_completion BETWEEN 0 AND 1",
            name="ck_actions_expected_completion",
        ),
        CheckConstraint(
            "pass_risk IS NULL OR pass_risk BETWEEN 0 AND 1", name="ck_actions_pass_risk"
        ),
        CheckConstraint("fold >= 1", name="ck_actions_fold_positive"),
        Index("ix_attacking_actions_player_id", "player_id"),
        Index("ix_attacking_actions_match_id", "match_id"),
        Index("ix_attacking_actions_possession_id", "possession_id"),
        Index("ix_attacking_actions_action_type", "action_type"),
        Index("ix_attacking_actions_attacking_value", "attacking_value"),
        Index("ix_attacking_actions_pass_index", "pass_index"),
    )

    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pass_index: Mapped[int | None] = mapped_column(
        ForeignKey("passes.pass_index", ondelete="SET NULL"), nullable=True, unique=True
    )
    match_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    possession_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_index: Mapped[int] = mapped_column(Integer, nullable=False)
    player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id", ondelete="SET NULL"), nullable=True
    )
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    action_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_x: Mapped[float] = mapped_column(Float, nullable=False)
    start_y: Mapped[float] = mapped_column(Float, nullable=False)
    end_x: Mapped[float] = mapped_column(Float, nullable=False)
    end_y: Mapped[float] = mapped_column(Float, nullable=False)
    state_value_before: Mapped[float] = mapped_column(Float, nullable=False)
    state_value_after: Mapped[float] = mapped_column(Float, nullable=False)
    attacking_value: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    under_pressure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    progressive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_completion: Mapped[float | None] = mapped_column(Float)
    pass_risk: Mapped[float | None] = mapped_column(Float)
    risk_reward_category: Mapped[str | None] = mapped_column(String(40))
    fold: Mapped[int] = mapped_column(Integer, nullable=False)

    player: Mapped[Player | None] = relationship(back_populates="attacking_actions")


class PlayerAttackingProfile(Base):
    """Bundesliga pass/carry possession-value aggregate for one product-cohort player."""

    __tablename__ = "player_attacking_profiles"
    __table_args__ = (
        CheckConstraint("actions >= 0 AND passes >= 0 AND carries >= 0", name="ck_attacking_profile_counts"),
        CheckConstraint(
            "positive_value_action_rate BETWEEN 0 AND 1", name="ck_attacking_profile_positive_rate"
        ),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    actions: Mapped[int] = mapped_column(Integer, nullable=False)
    passes: Mapped[int] = mapped_column(Integer, nullable=False)
    carries: Mapped[int] = mapped_column(Integer, nullable=False)
    total_attacking_value: Mapped[float] = mapped_column(Float, nullable=False)
    attacking_value_per_100_actions: Mapped[float] = mapped_column(Float, nullable=False)
    total_pass_value: Mapped[float] = mapped_column(Float, nullable=False)
    pass_value_per_100_passes: Mapped[float | None] = mapped_column(Float)
    total_carry_value: Mapped[float] = mapped_column(Float, nullable=False)
    carry_value_per_100_carries: Mapped[float | None] = mapped_column(Float)
    positive_value_actions: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_value_action_rate: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_action_value: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_value_per_100_actions: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_action_value: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_value_per_100_actions: Mapped[float] = mapped_column(Float, nullable=False)
    attacking_value_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pass_value_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    carry_value_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)

    player: Mapped[Player] = relationship(back_populates="attacking_profile")


class PlayerIntelligenceProfile(Base):
    """V3.1 position context for one product-cohort player."""

    __tablename__ = "player_intelligence_profiles"
    __table_args__ = (
        CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_intelligence_profiles_position_group",
        ),
        CheckConstraint(
            "matches_observed >= 0", name="ck_intelligence_profiles_matches"
        ),
        Index("ix_intelligence_profiles_position_group", "position_group"),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)

    player: Mapped[Player] = relationship(back_populates="intelligence_profile")


class PlayerPercentile(Base):
    """One metric-specific same-position percentile for a V3.1 player."""

    __tablename__ = "player_percentiles"
    __table_args__ = (
        CheckConstraint("family IN ('style', 'performance')", name="ck_percentiles_family"),
        CheckConstraint(
            "peer_position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_percentiles_position_group",
        ),
        CheckConstraint(
            "percentile IS NULL OR percentile BETWEEN 0 AND 100",
            name="ck_percentiles_range",
        ),
        CheckConstraint(
            "peer_count >= 0 AND sample_count >= 0", name="ck_percentiles_counts"
        ),
        Index("ix_player_percentiles_player_id", "player_id"),
        Index("ix_player_percentiles_position_group", "peer_position_group"),
        Index("ix_player_percentiles_metric_name", "metric_name"),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    family: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_value: Mapped[float | None] = mapped_column(Float)
    percentile: Mapped[float | None] = mapped_column(Float)
    peer_position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    peer_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    eligibility_reason: Mapped[str] = mapped_column(String(100), nullable=False)

    player: Mapped[Player] = relationship(back_populates="percentiles")


class PlayerArchetype(Base):
    """One eligible player's V3.2C position-relative style assignment."""

    __tablename__ = "player_archetypes"
    __table_args__ = (
        CheckConstraint(
            "archetype_id IN ('direct_progressor', 'safe_circulator')",
            name="ck_player_archetypes_semantic_id",
        ),
        CheckConstraint(
            "position_group IN ('DEF', 'MID', 'FWD')",
            name="ck_player_archetypes_outfield_position",
        ),
        CheckConstraint(
            "centroid_distance >= 0 AND second_centroid_distance >= centroid_distance",
            name="ck_player_archetypes_distances",
        ),
        CheckConstraint(
            "separation_margin BETWEEN 0 AND 1",
            name="ck_player_archetypes_separation",
        ),
        CheckConstraint("eligible = true", name="ck_player_archetypes_eligible"),
        Index("ix_player_archetypes_archetype_id", "archetype_id"),
        Index("ix_player_archetypes_position_group", "position_group"),
        Index("ix_player_archetypes_raw_cluster", "raw_cluster_id"),
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    archetype_id: Mapped[str] = mapped_column(String(40), nullable=False)
    archetype_name: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    centroid_distance: Mapped[float] = mapped_column(Float, nullable=False)
    second_centroid_distance: Mapped[float] = mapped_column(Float, nullable=False)
    separation_margin: Mapped[float] = mapped_column(Float, nullable=False)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    expected_completion_rate_position_z: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_pass_rate_position_z: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_pass_rate_position_z: Mapped[float] = mapped_column(Float, nullable=False)
    long_pass_rate_position_z: Mapped[float] = mapped_column(Float, nullable=False)
    positive_forward_distance_per_100_passes_position_z: Mapped[float] = mapped_column(
        Float, nullable=False
    )
    carry_share_of_actions_position_z: Mapped[float] = mapped_column(Float, nullable=False)

    player: Mapped[Player] = relationship(back_populates="archetype")


class TeamStyleProfile(Base):
    """One production-qualified observed team-style profile."""

    __tablename__ = "team_style_profiles"
    __table_args__ = (
        CheckConstraint("matches_observed >= 1", name="ck_team_style_matches"),
        CheckConstraint("passes >= 0 AND carries >= 0 AND actions >= 0", name="ck_team_style_counts"),
    )

    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)
    sample_scope: Mapped[str] = mapped_column(String(500), nullable=False)
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    contributors: Mapped[int] = mapped_column(Integer, nullable=False)
    passes: Mapped[int] = mapped_column(Integer, nullable=False)
    carries: Mapped[int] = mapped_column(Integer, nullable=False)
    actions: Mapped[int] = mapped_column(Integer, nullable=False)
    shots: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_completion_rate: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    long_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    positive_forward_distance_per_100_passes: Mapped[float] = mapped_column(Float, nullable=False)
    carry_share_of_actions: Mapped[float] = mapped_column(Float, nullable=False)
    average_forward_distance: Mapped[float] = mapped_column(Float, nullable=False)
    final_third_entries_per_100_passes: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_carry_rate: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_action_rate: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_action_rate: Mapped[float] = mapped_column(Float, nullable=False)
    shots_per_match: Mapped[float] = mapped_column(Float, nullable=False)
    xg_per_shot: Mapped[float] = mapped_column(Float, nullable=False)
    xg_per_match: Mapped[float] = mapped_column(Float, nullable=False)
    attacking_value_per_100_actions: Mapped[float] = mapped_column(Float, nullable=False)


class TeamRoleProfile(Base):
    """Pooled observed role style for one qualified team and broad position."""

    __tablename__ = "team_role_profiles"
    __table_args__ = (
        CheckConstraint("position_group IN ('DEF', 'MID', 'FWD')", name="ck_team_roles_outfield"),
        CheckConstraint("matches_observed >= 1 AND contributor_count >= 1", name="ck_team_roles_support"),
        UniqueConstraint("team_id", "position_group", name="uq_team_roles_team_position"),
        Index("ix_team_role_profiles_team_id", "team_id"),
    )

    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    position_group: Mapped[str] = mapped_column(String(3), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)
    aggregation_method: Mapped[str] = mapped_column(String(50), nullable=False)
    matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    contributor_count: Mapped[int] = mapped_column(Integer, nullable=False)
    contributors: Mapped[str] = mapped_column(Text, nullable=False)
    passes: Mapped[int] = mapped_column(Integer, nullable=False)
    carries: Mapped[int] = mapped_column(Integer, nullable=False)
    actions: Mapped[int] = mapped_column(Integer, nullable=False)
    shots: Mapped[int] = mapped_column(Integer, nullable=False)
    support_level: Mapped[str] = mapped_column(String(50), nullable=False)
    support_message: Mapped[str] = mapped_column(String(500), nullable=False)
    expected_completion_rate: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    long_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    positive_forward_distance_per_100_passes: Mapped[float] = mapped_column(Float, nullable=False)
    carry_share_of_actions: Mapped[float] = mapped_column(Float, nullable=False)
    expected_completion_rate_z: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_pass_rate_z: Mapped[float] = mapped_column(Float, nullable=False)
    progressive_pass_rate_z: Mapped[float] = mapped_column(Float, nullable=False)
    long_pass_rate_z: Mapped[float] = mapped_column(Float, nullable=False)
    positive_forward_distance_per_100_passes_z: Mapped[float] = mapped_column(Float, nullable=False)
    carry_share_of_actions_z: Mapped[float] = mapped_column(Float, nullable=False)


class PlayerRoleFit(Base):
    """Position-matched RMS style distance to one target-team role."""

    __tablename__ = "player_role_fits"
    __table_args__ = (
        CheckConstraint("position_group IN ('DEF', 'MID', 'FWD')", name="ck_role_fits_outfield"),
        CheckConstraint("role_distance >= 0", name="ck_role_fits_distance"),
        CheckConstraint("sample_support IN ('limited', 'higher')", name="ck_role_fits_support"),
        UniqueConstraint(
            "target_team_id", "position_group", "recommendation_rank",
            name="uq_role_fits_target_position_rank",
        ),
        Index("ix_player_role_fits_target_role", "target_team_id", "position_group"),
        Index("ix_player_role_fits_player_id", "player_id"),
    )

    target_team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    target_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    player_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    position: Mapped[str] = mapped_column(String(100), nullable=False)
    position_group: Mapped[str] = mapped_column(String(3), nullable=False)
    is_target_team_player: Mapped[bool] = mapped_column(Boolean, nullable=False)
    calculation_scope: Mapped[str] = mapped_column(String(60), nullable=False)
    role_distance: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation_rank: Mapped[int | None] = mapped_column(Integer)
    closest_feature_1: Mapped[str] = mapped_column(String(100), nullable=False)
    closest_feature_2: Mapped[str] = mapped_column(String(100), nullable=False)
    closest_feature_3: Mapped[str] = mapped_column(String(100), nullable=False)
    largest_difference: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_gaps: Mapped[str] = mapped_column(Text, nullable=False)
    distance_contributions: Mapped[str] = mapped_column(Text, nullable=False)
    player_matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    player_pass_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    player_carries: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_support: Mapped[str] = mapped_column(String(20), nullable=False)
    sample_support_message: Mapped[str] = mapped_column(String(300), nullable=False)
    role_matches_observed: Mapped[int] = mapped_column(Integer, nullable=False)
    role_contributor_count: Mapped[int] = mapped_column(Integer, nullable=False)
    role_actions: Mapped[int] = mapped_column(Integer, nullable=False)
    role_support_message: Mapped[str] = mapped_column(String(500), nullable=False)
    archetype_id: Mapped[str | None] = mapped_column(String(40))
    archetype_name: Mapped[str | None] = mapped_column(String(100))
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)

    player: Mapped[Player] = relationship(back_populates="role_fits")
