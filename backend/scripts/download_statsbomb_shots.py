"""Download a documented multi-competition StatsBomb Open Data shot corpus."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.shot_features import XG_MODEL_FEATURE_COLUMNS, normalize_shot_events
from scripts.download_statsbomb import available_match_ids, save_parquet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "shots.parquet"


@dataclass(frozen=True)
class CorpusTarget:
    country_name: str
    competition_name: str
    season_name: str
    product_cohort: bool = False


@dataclass(frozen=True)
class CorpusSelection:
    competition_id: int
    season_id: int
    country_name: str
    competition_name: str
    season_name: str
    product_cohort: bool


# Deliberately bounded, auditable men's corpus: 233 accessible matches at audit time.
CORPUS_TARGETS = (
    CorpusTarget("Germany", "1. Bundesliga", "2023/2024", True),
    CorpusTarget("International", "FIFA World Cup", "2022"),
    CorpusTarget("Europe", "UEFA Euro", "2024"),
    CorpusTarget("South America", "Copa America", "2024"),
    CorpusTarget("Africa", "African Cup of Nations", "2023"),
)


def select_corpus(competitions: pd.DataFrame) -> list[CorpusSelection]:
    """Resolve named corpus targets against the live Open Data catalogue."""
    required = {
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
    }
    missing = required.difference(competitions.columns)
    if missing:
        raise ValueError(f"Competition catalogue is missing columns: {sorted(missing)}")
    selections: list[CorpusSelection] = []
    for target in CORPUS_TARGETS:
        rows = competitions.loc[
            competitions["country_name"].astype(str).eq(target.country_name)
            & competitions["competition_name"].astype(str).eq(target.competition_name)
            & competitions["season_name"].astype(str).eq(target.season_name)
        ]
        if "competition_gender" in rows.columns:
            rows = rows.loc[rows["competition_gender"].astype(str).str.casefold().eq("male")]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one accessible catalogue row for {target}, found {len(rows)}"
            )
        row = rows.iloc[0]
        selections.append(
            CorpusSelection(
                competition_id=int(row["competition_id"]),
                season_id=int(row["season_id"]),
                country_name=str(row["country_name"]),
                competition_name=str(row["competition_name"]),
                season_name=str(row["season_name"]),
                product_cohort=target.product_cohort,
            )
        )
    return selections


def _event_values(raw: object) -> list[Mapping[str, Any]]:
    values = raw.values() if isinstance(raw, Mapping) else raw
    if not isinstance(values, (list, tuple)) and not hasattr(values, "__iter__"):
        raise TypeError("StatsBomb returned an unsupported event payload")
    return [value for value in values if isinstance(value, Mapping)]


def download_shot_corpus(client: object, selections: list[CorpusSelection]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total_matches = 0
    for selection in selections:
        matches = client.matches(  # type: ignore[attr-defined]
            competition_id=selection.competition_id,
            season_id=selection.season_id,
            fmt="dataframe",
        )
        match_ids = available_match_ids(matches)
        print(
            f"{selection.competition_name} {selection.season_name}: "
            f"{len(match_ids):,} matches"
        )
        for local_index, match_id in enumerate(match_ids, start=1):
            total_matches += 1
            print(f"  match {local_index}/{len(match_ids)} · {match_id}")
            events = client.events(match_id=match_id, fmt="dict")  # type: ignore[attr-defined]
            frame = normalize_shot_events(
                _event_values(events),
                competition_id=selection.competition_id,
                season_id=selection.season_id,
                product_cohort=selection.product_cohort,
            )
            frames.append(frame)
    shots = pd.concat(frames, ignore_index=True)
    if shots["shot_id"].isna().any() or shots["shot_id"].duplicated().any():
        raise ValueError("Every shot must have one unique StatsBomb event ID")
    if shots["goal"].isna().any():
        raise ValueError("Every shot must have an explicit StatsBomb outcome")
    if shots["match_id"].nunique() != total_matches:
        raise ValueError("Downloaded match count does not match normalized shot matches")
    return shots


def print_audit(shots: pd.DataFrame, selections: list[CorpusSelection]) -> None:
    eligible = shots.loc[shots["model_eligible"]]
    print("\nStatsBomb shot audit")
    for selection in selections:
        subset = shots.loc[
            shots["competition_id"].eq(selection.competition_id)
            & shots["season_id"].eq(selection.season_id)
        ]
        print(
            f"  {selection.country_name} · {selection.competition_name} · "
            f"{selection.season_name}: {subset['match_id'].nunique():,} matches, "
            f"{len(subset):,} shots, {int(subset['goal'].sum()):,} goals"
        )
    print(f"Matches used: {shots['match_id'].nunique():,}")
    print(f"Shots retained: {len(shots):,}")
    print(f"Goals: {int(shots['goal'].sum()):,}")
    print(f"Penalties: {int(shots['penalty'].sum()):,}")
    print(f"Penalty shootout shots: {int(shots['penalty_shootout'].sum()):,}")
    print(f"Eligible non-penalty shots: {len(eligible):,}")
    print(f"Eligible non-penalty goals: {int(eligible['goal'].sum()):,}")
    print("Candidate feature missingness (eligible rows):")
    for column in XG_MODEL_FEATURE_COLUMNS:
        missing = int(eligible[column].isna().sum())
        print(f"  {column}: {missing:,} ({missing / len(eligible):.2%})")
    print(f"Shot outcomes: {sorted(shots['shot_outcome'].dropna().unique())}")


def run(output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    from statsbombpy import sb

    catalogue = sb.competitions(fmt="dataframe")
    selections = select_corpus(catalogue)
    shots = download_shot_corpus(sb, selections)
    save_parquet(shots, output_path)
    print_audit(shots, selections)
    print(f"Saved: {output_path}")
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

