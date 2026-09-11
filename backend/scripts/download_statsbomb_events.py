"""Download and normalize the full 233-match V2.1 StatsBomb event corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analytics.action_value_features import (
    build_possession_states,
    horizon_audit,
    normalize_event,
    save_parquet,
)
from scripts.download_statsbomb import available_match_ids
from scripts.download_statsbomb_shots import _event_values, select_corpus

ROOT = Path(__file__).resolve().parents[2]
EVENTS_PATH = ROOT / "data" / "processed" / "events.parquet"
STATES_PATH = ROOT / "data" / "processed" / "possession_states.parquet"
SHOT_OOF_PATH = ROOT / "data" / "processed" / "shot_oof_predictions.parquet"


def normalize_match_events(
    raw: object, *, competition_id: int, season_id: int, product_cohort: bool
) -> pd.DataFrame:
    records = [
        normalize_event(
            event,
            competition_id=competition_id,
            season_id=season_id,
            product_cohort=product_cohort,
        )
        for event in _event_values(raw)
    ]
    return pd.DataFrame.from_records(records).sort_values("event_index", kind="stable")


def download_events(client: object) -> pd.DataFrame:
    catalogue = client.competitions(fmt="dataframe")  # type: ignore[attr-defined]
    selections = select_corpus(catalogue)
    frames: list[pd.DataFrame] = []
    for selection in selections:
        matches = client.matches(  # type: ignore[attr-defined]
            competition_id=selection.competition_id,
            season_id=selection.season_id,
            fmt="dataframe",
        )
        match_ids = available_match_ids(matches)
        print(f"{selection.competition_name} {selection.season_name}: {len(match_ids):,} matches")
        for number, match_id in enumerate(match_ids, start=1):
            print(f"  match {number}/{len(match_ids)} · {match_id}")
            raw = client.events(match_id=match_id, fmt="dict")  # type: ignore[attr-defined]
            frames.append(
                normalize_match_events(
                    raw,
                    competition_id=selection.competition_id,
                    season_id=selection.season_id,
                    product_cohort=selection.product_cohort,
                )
            )
    events = pd.concat(frames, ignore_index=True)
    if events["match_id"].nunique() != 233:
        raise ValueError(f"Expected the frozen 233-match corpus, found {events['match_id'].nunique()}")
    return events


def run(events_path: Path = EVENTS_PATH, states_path: Path = STATES_PATH) -> pd.DataFrame:
    from statsbombpy import sb

    events = download_events(sb)
    save_parquet(events, events_path)
    shot_oof = pd.read_parquet(SHOT_OOF_PATH)
    states, audit = build_possession_states(events, shot_oof)
    save_parquet(states, states_path)
    print("\nFull-event audit")
    print(f"Matches: {events['match_id'].nunique():,}")
    print(f"Events: {len(events):,}")
    print(f"Possessions: {audit['possessions']:,}")
    print(f"Eligible pre-event states: {audit['states']:,}")
    print(f"OOF shots mapped exactly once: {audit['mapped_oof_shots']:,}")
    print(f"Event types: {sorted(events['event_type'].unique())}")
    print("Horizon candidates:")
    for name, values in horizon_audit(states).items():
        print(f"  {name}: {values}")
    print(f"Saved events: {events_path}")
    print(f"Saved states: {states_path}")
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-output", type=Path, default=EVENTS_PATH)
    parser.add_argument("--states-output", type=Path, default=STATES_PATH)
    args = parser.parse_args()
    run(args.events_output.resolve(), args.states_output.resolve())


if __name__ == "__main__":
    main()
