# Scripts

Command-line utilities for data ingestion, preprocessing, training, and maintenance live here.

## StatsBomb Open Data passes

From the `backend/` directory, run:

```powershell
python -m scripts.download_statsbomb
```

The command discovers the available open-data competitions, prefers the men's
2023/24 Bundesliga, downloads every available match, and writes normalized pass
events to `data/processed/passes.parquet`. It does not create model features.

## StatsBomb Open Data shots

From `backend/`, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_statsbomb_shots
```

The command resolves a deliberately bounded five-competition corpus against the
live StatsBomb catalogue and writes `data/processed/shots.parquet`. It prints the
competition/season, match, shot, goal, penalty, outcome, and candidate-feature
missingness audit before modeling.

## Player-intelligence database snapshot

After running `python -m analytics.player_intelligence`, apply migrations and
reload the full additive snapshot from `backend/`:

```powershell
python -m alembic upgrade head
python -m scripts.load_database
```

The loader validates and inserts `player_intelligence_profiles` and normalized
`player_percentiles` without dropping or replacing the V1/V2 schema objects.
