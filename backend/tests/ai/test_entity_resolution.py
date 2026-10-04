"""Deterministic player entity-resolution behavior."""

import pytest
from sqlalchemy.orm import Session

from app.ai.entity_resolution import resolve_player, resolve_team
from app.ai.schemas import EntityRef, EntityResolutionStatus, TeamRef
from app.models import Player
from app.services.players import normalize_player_name


def test_resolves_unique_name_and_stable_id(ai_session: Session) -> None:
    by_name = resolve_player(ai_session, EntityRef(player_name="Bob"))
    by_id = resolve_player(ai_session, EntityRef(player_id=1))
    assert by_name.status is EntityResolutionStatus.RESOLVED
    assert by_name.player_id == 2
    assert by_id.status is EntityResolutionStatus.RESOLVED
    assert by_id.player is not None
    assert by_id.player.player_name == "Alice Playmaker"


def test_ambiguous_name_returns_all_bounded_candidates(ai_session: Session) -> None:
    result = resolve_player(ai_session, EntityRef(player_name="Alice"))
    assert result.status is EntityResolutionStatus.AMBIGUOUS
    assert result.player_id is None
    assert [candidate.player_id for candidate in result.candidates] == [1, 5]


@pytest.mark.parametrize(
    "query",
    [
        "Karim-David Adeyemi",
        "karim-david adeyemi",
        "Karim David Adeyemi",
        "Karim Adeyemi",
        "karim adeyemi",
        "KARIM ADEYEMI",
        "  Karim   Adeyemi  ",
        "Adeyemi",
    ],
)
def test_hyphen_middle_name_variants_resolve_the_same_unique_player(
    ai_session: Session,
    query: str,
) -> None:
    player = ai_session.get(Player, 2)
    assert player is not None
    player.player_name = "Karim-David Adeyemi"
    ai_session.flush()

    result = resolve_player(ai_session, EntityRef(player_name=query))

    assert result.status is EntityResolutionStatus.RESOLVED
    assert result.player_id == 2
    assert result.player is not None
    assert result.player.player_name == "Karim-David Adeyemi"


def test_name_normalization_is_unicode_case_and_punctuation_aware() -> None:
    assert normalize_player_name("  Karim-David   Adeyemi  ") == (
        "karim david adeyemi"
    )
    assert normalize_player_name("O’Connor") == "oconnor"
    assert normalize_player_name("J. Smith") == "j smith"


def test_ambiguous_surname_requires_clarification_but_full_name_resolves(
    ai_session: Session,
) -> None:
    surname = resolve_player(ai_session, EntityRef(player_name="Williams"))
    full_name = resolve_player(ai_session, EntityRef(player_name="Alex Williams"))

    assert surname.status is EntityResolutionStatus.AMBIGUOUS
    assert [candidate.player_name for candidate in surname.candidates] == [
        "Alex Williams",
        "Jordan Williams",
    ]
    assert full_name.status is EntityResolutionStatus.RESOLVED
    assert full_name.player_id == 7


def test_token_subset_requires_every_query_token(ai_session: Session) -> None:
    player = ai_session.get(Player, 2)
    assert player is not None
    player.player_name = "Karim-David Adeyemi"
    ai_session.flush()

    result = resolve_player(ai_session, EntityRef(player_name="Karim Smith"))

    assert result.status is EntityResolutionStatus.NOT_FOUND


@pytest.mark.parametrize("query", ["Granit Xhaka", "granit xhaka", "GRANIT XHAKA"])
def test_existing_case_insensitive_behavior_is_preserved(
    ai_session: Session,
    query: str,
) -> None:
    player = ai_session.get(Player, 2)
    assert player is not None
    player.player_name = "Granit Xhaka"
    ai_session.flush()

    assert resolve_player(ai_session, EntityRef(player_name=query)).player_id == 2


def test_player_existence_is_independent_from_profile_availability(
    ai_session: Session,
) -> None:
    ai_session.add(
        Player(
            player_id=99,
            player_name="Unprofiled Player",
            team_id=999,
            team_name="Example Club",
            position="Center Midfield",
            position_group="MID",
            matches_observed=1,
            pass_attempts=1,
            overall_reliable=False,
        )
    )
    ai_session.flush()

    result = resolve_player(ai_session, EntityRef(player_name="Unprofiled Player"))

    assert result.status is EntityResolutionStatus.RESOLVED
    assert result.player_id == 99


def test_not_found_never_guesses(ai_session: Session) -> None:
    by_name = resolve_player(ai_session, EntityRef(player_name="Nobody Here"))
    by_id = resolve_player(ai_session, EntityRef(player_id=999))
    assert by_name.status is EntityResolutionStatus.NOT_FOUND
    assert by_id.status is EntityResolutionStatus.NOT_FOUND
    assert by_name.candidates == []
    assert by_id.player is None


def test_resolves_loaded_teams_separately_from_analytics_support(
    ai_session: Session,
) -> None:
    resolved = resolve_team(
        ai_session,
        TeamRef(team_name="bayer leverkusen"),
    )
    unsupported = resolve_team(ai_session, TeamRef(team_name="Alpha FC"))
    assert resolved.status is EntityResolutionStatus.RESOLVED
    assert resolved.team_id == 904
    assert resolved.team is not None
    assert resolved.team.team_name == "Bayer Leverkusen"
    assert resolved.analytics_supported is True
    assert unsupported.status is EntityResolutionStatus.RESOLVED
    assert unsupported.team_id == 100
    assert unsupported.analytics_supported is False
