"""Download and normalize StatsBomb Open Data pass events.

This module performs raw-data ingestion only. It intentionally does not derive
model features or train an expected-pass-completion model.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PREFERRED_COUNTRY = "Germany"
PREFERRED_COMPETITION = "1. Bundesliga"
PREFERRED_SEASON = "2023/2024"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "passes.parquet"

PASS_COLUMNS = [
    "match_id",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "minute",
    "second",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle",
    "pass_height",
    "body_part",
    "pass_type",
    "under_pressure",
    "pass_outcome",
    "completed",
]

INTEGER_COLUMNS = ["match_id", "player_id", "team_id", "minute", "second"]
FLOAT_COLUMNS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle",
]
STRING_COLUMNS = [
    "player_name",
    "team_name",
    "position",
    "pass_height",
    "body_part",
    "pass_type",
    "pass_outcome",
]


@dataclass(frozen=True)
class CompetitionSelection:
    competition_id: int
    season_id: int
    country_name: str
    competition_name: str
    season_name: str


def _is_missing(value: object) -> bool:
    """Return whether a scalar value represents missing data."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _number_or_none(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_coordinates(value: object) -> tuple[float | None, float | None]:
    """Safely extract an x/y pair from a StatsBomb location value."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None, None
    return _number_or_none(value[0]), _number_or_none(value[1])


def _entity_value(value: object, key: str) -> object | None:
    if not isinstance(value, Mapping):
        return None
    result = value.get(key)
    return None if _is_missing(result) else result


def completion_from_outcome(outcome: object) -> int:
    """Map a StatsBomb pass outcome to the binary supervised-learning target.

    Inspection of the Open Data event schema confirms that a successful pass
    omits ``pass.outcome``. Any populated outcome (for example ``Incomplete``
    or ``Out``) therefore represents an unsuccessful pass.
    """
    if _is_missing(outcome):
        return 1
    if isinstance(outcome, Mapping):
        has_name = not _is_missing(outcome.get("name"))
        has_id = not _is_missing(outcome.get("id"))
        return 0 if has_name or has_id else 1
    return 0


def normalize_pass_event(event: Mapping[str, object]) -> dict[str, object] | None:
    """Normalize one raw StatsBomb pass event into a flat record."""
    if _entity_value(event.get("type"), "name") != "Pass":
        return None

    pass_data = event.get("pass")
    if not isinstance(pass_data, Mapping):
        pass_data = {}

    start_x, start_y = extract_coordinates(event.get("location"))
    end_x, end_y = extract_coordinates(pass_data.get("end_location"))
    outcome = pass_data.get("outcome")
    under_pressure = event.get("under_pressure")

    return {
        "match_id": event.get("match_id"),
        "player_id": _entity_value(event.get("player"), "id"),
        "player_name": _entity_value(event.get("player"), "name"),
        "team_id": _entity_value(event.get("team"), "id"),
        "team_name": _entity_value(event.get("team"), "name"),
        "position": _entity_value(event.get("position"), "name"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "pass_length": pass_data.get("length"),
        "pass_angle": pass_data.get("angle"),
        "pass_height": _entity_value(pass_data.get("height"), "name"),
        "body_part": _entity_value(pass_data.get("body_part"), "name"),
        "pass_type": _entity_value(pass_data.get("type"), "name"),
        "under_pressure": False if _is_missing(under_pressure) else bool(under_pressure),
        "pass_outcome": _entity_value(outcome, "name"),
        "completed": completion_from_outcome(outcome),
    }


def normalize_pass_events(events: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Extract pass events and return a consistently typed tabular dataset."""
    records = [record for event in events if (record := normalize_pass_event(event)) is not None]
    passes = pd.DataFrame.from_records(records, columns=PASS_COLUMNS)

    for column in INTEGER_COLUMNS:
        passes[column] = pd.to_numeric(passes[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS:
        passes[column] = pd.to_numeric(passes[column], errors="coerce").astype("Float64")
    for column in STRING_COLUMNS:
        passes[column] = passes[column].astype("string")

    passes["under_pressure"] = passes["under_pressure"].fillna(False).astype(bool)
    passes["completed"] = pd.to_numeric(passes["completed"], errors="coerce").astype("Int8")
    return passes


def _canonical_season(value: object) -> str:
    season = str(value).strip().replace("-", "/")
    parts = season.split("/")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return season
    start_year = int(parts[0])
    end_year = int(parts[1])
    if end_year < 100:
        end_year = (start_year // 100) * 100 + end_year
    return f"{start_year:04d}/{end_year:04d}"


def _season_start_year(value: object) -> int:
    canonical = _canonical_season(value)
    first_part = canonical.split("/", maxsplit=1)[0]
    return int(first_part) if first_part.isdigit() else -1


def select_competition(competitions: pd.DataFrame) -> CompetitionSelection:
    """Select 2023/24 Bundesliga when present, otherwise its newest open season."""
    required = {
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
    }
    missing_columns = required.difference(competitions.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"StatsBomb competition data is missing required columns: {names}")
    if competitions.empty:
        raise ValueError("StatsBomb returned no open-data competitions")

    country = competitions["country_name"].astype("string").str.casefold()
    competition = competitions["competition_name"].astype("string").str.casefold()
    bundesliga_mask = (
        country.eq(PREFERRED_COUNTRY.casefold())
        & competition.eq(PREFERRED_COMPETITION.casefold())
    )
    if "competition_gender" in competitions.columns:
        gender = competitions["competition_gender"].astype("string").str.casefold()
        bundesliga_mask &= gender.eq("male")

    bundesliga = competitions.loc[bundesliga_mask].copy()
    if bundesliga.empty:
        raise ValueError("No men's 1. Bundesliga season is available in StatsBomb Open Data")

    preferred = bundesliga[
        bundesliga["season_name"].map(_canonical_season).eq(PREFERRED_SEASON)
    ]
    if not preferred.empty:
        selected = preferred.iloc[0]
    else:
        bundesliga["_season_start"] = bundesliga["season_name"].map(_season_start_year)
        selected = bundesliga.sort_values("_season_start", ascending=False).iloc[0]

    return CompetitionSelection(
        competition_id=int(selected["competition_id"]),
        season_id=int(selected["season_id"]),
        country_name=str(selected["country_name"]),
        competition_name=str(selected["competition_name"]),
        season_name=str(selected["season_name"]),
    )


def available_match_ids(matches: pd.DataFrame) -> list[int]:
    """Return every match ID whose event data is marked available."""
    if "match_id" not in matches.columns:
        raise ValueError("StatsBomb match data is missing the match_id column")

    available = matches
    if "match_status" in matches.columns:
        statuses = matches["match_status"].astype("string").str.casefold()
        available = matches.loc[statuses.eq("available")]

    return [int(match_id) for match_id in available["match_id"].dropna().drop_duplicates()]


def _event_values(raw_events: object) -> Iterable[Mapping[str, object]]:
    if isinstance(raw_events, Mapping):
        values = raw_events.values()
    elif isinstance(raw_events, Iterable) and not isinstance(raw_events, (str, bytes)):
        values = raw_events
    else:
        raise TypeError("StatsBomb returned events in an unsupported format")

    for event in values:
        if isinstance(event, Mapping):
            yield event


def download_competition_passes(
    client: object,
    selection: CompetitionSelection,
) -> tuple[pd.DataFrame, int]:
    """Download and normalize every available match for one competition-season."""
    matches = client.matches(  # type: ignore[attr-defined]
        competition_id=selection.competition_id,
        season_id=selection.season_id,
        fmt="dataframe",
    )
    match_ids = available_match_ids(matches)
    if not match_ids:
        raise RuntimeError("The selected competition-season has no available matches")

    match_frames: list[pd.DataFrame] = []
    for index, match_id in enumerate(match_ids, start=1):
        print(f"Downloading match {index}/{len(match_ids)}: {match_id}")
        try:
            raw_events = client.events(match_id=match_id, fmt="dict")  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"Failed to download events for match {match_id}") from exc
        match_frames.append(normalize_pass_events(_event_values(raw_events)))

    return pd.concat(match_frames, ignore_index=True), len(match_ids)


def save_parquet(passes: pd.DataFrame, output_path: Path) -> None:
    """Write the processed dataset atomically using the PyArrow engine."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        passes.to_parquet(temporary_path, engine="pyarrow", index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def print_summary(
    selection: CompetitionSelection,
    matches_processed: int,
    passes: pd.DataFrame,
    output_path: Path,
) -> None:
    total = len(passes)
    completed = int(passes["completed"].sum()) if total else 0
    incomplete = total - completed
    under_pressure = int(passes["under_pressure"].sum()) if total else 0
    completion_rate = completed / total if total else 0.0
    pressure_rate = under_pressure / total if total else 0.0

    print("\nStatsBomb ingestion complete")
    print(f"Selected competition: {selection.country_name} - {selection.competition_name}")
    print(f"Selected season: {selection.season_name}")
    print(f"Matches processed: {matches_processed:,}")
    print(f"Total passes: {total:,}")
    print(f"Completed passes: {completed:,}")
    print(f"Incomplete passes: {incomplete:,}")
    print(f"Overall completion rate: {completion_rate:.2%}")
    print(f"Unique players: {passes['player_id'].nunique(dropna=True):,}")
    print(f"Unique teams: {passes['team_id'].nunique(dropna=True):,}")
    print(f"Passes under pressure: {under_pressure:,} ({pressure_rate:.2%})")
    print(f"Saved dataset: {output_path}")


def run(output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Discover, download, normalize, save, and summarize open pass data."""
    from statsbombpy import sb

    competitions = sb.competitions(fmt="dataframe")
    selection = select_competition(competitions)
    passes, matches_processed = download_competition_passes(sb, selection)
    save_parquet(passes, output_path)
    print_summary(selection, matches_processed, passes, output_path)
    return passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

