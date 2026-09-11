# Analytics

This package contains feature engineering, model evaluation, grouped out-of-fold
prediction, and player-level analytics for FootyScout.

## Pass features

With the backend virtual environment active, run from `backend/`:

```powershell
python -m analytics.pass_features
```

The command reads `data/processed/passes.parquet` and writes the leakage-safe,
pre-outcome feature table to `data/processed/pass_features.parquet`. It retains
metadata and the `completed` target but keeps both out of `MODEL_FEATURE_COLUMNS`.

## Expected-pass model benchmark

From `backend/`, reproduce the Logistic Regression, PyTorch MLP, and XGBoost
comparison with:

```powershell
.\.venv\Scripts\python.exe -m analytics.train_pass_model
```

The command keeps matches grouped, selects by validation log loss, evaluates the
frozen selection on the untouched test split, and writes authoritative grouped
OOF predictions. The passing-profile snapshot is regenerated when XGBoost remains
selected; V3 player-intelligence and similarity stages run separately in their
documented order.

## Expected-goals model

After downloading the shot corpus, run from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m analytics.train_xg_model
```

V2.1 uses non-penalty shots only, keeps matches grouped through fixed evaluation
and five-fold OOF prediction, and selects between Logistic Regression and a small
controlled XGBoost candidate set using validation log loss. Penalties (including
shootouts) remain in the normalized and product shot datasets but have no model
prediction and are excluded from player aggregates. A player is marked as having
a reliable shooting sample at 20 non-penalty shots; lower-volume players remain
available with a limited-sample flag.

## Attacking action value

After `scripts.download_statsbomb_events` builds the full 233-match event/state
corpus, run from `backend/`:

```powershell
python -m analytics.train_action_value_model
```

The target is all remaining FootyScout grouped-OOF xG in the supplied StatsBomb
possession. States are defined immediately before each eligible on-ball event.
The pipeline reuses V2.1's match-grouped train/validation/test allocation. Its
upstream xG labels are strictly nested: outer-training shots receive inner
match-grouped OOF xG, while outer-held-out shots are predicted by the frozen xG
specification fitted only on outer-training matches. It selects the controlled
state candidate with validation RMSE, reports the untouched test once, and
produces one non-negative five-fold OOF prediction per state.
Pass/carry action value is `V(after) - V(before)` and is observational rather
than causal. It does not model defensive or off-ball value.

## Player intelligence

Run V3.1 from `backend/`:

```powershell
python -m analytics.player_intelligence
```

The pipeline joins the frozen passing, shooting, and attacking profiles onto
the passing-product player cohort. It writes one unified row per player,
normalized metric percentile rows, an unscaled style-only exploratory matrix,
and an audit report. `player_feature_registry.py` is the authoritative contract
for feature family, source, sample type and threshold, directionality, future
similarity/clustering eligibility, and action-value stability caveats.

Percentiles are tie-aware empirical ranks within eligible same-position peers,
not overall ratings. Higher style percentiles indicate more of a tendency, not
better football performance. Missing or low-sample metrics remain null, and no
percentile is emitted for a position/metric cohort with fewer than 10 eligible
players. Goalkeepers receive no forced outfield radar.

## Style stability and clustering readiness

Run the analysis-only V3.2A audit from `backend/`:

```powershell
python -m analytics.player_style_stability `
  --normalized-output ../data/processed/player_style_features_position_normalized.parquet `
  --normalized-pass-threshold 50 `
  --normalized-carry-threshold 29
```

The pipeline verifies its full-sample recomputation against the frozen V3.1
style matrix, splits each player's observed matches chronologically, and audits
all 25/50/75/100-pass by 15/20/29-carry threshold combinations. It writes raw
expanded and position-normalized analysis matrices plus JSON audit metadata.
These artifacts contain no performance features and are not used by the API,
database, percentiles, or radar. This step does not run PCA or clustering.

Run the analysis-only V3.2B archetype audit from `backend/`:

```powershell
python -m analytics.player_archetype_analysis
```

The command validates the frozen 133-player position-normalized cohort, fits
PCA for diagnostics, and compares K-Means, full-covariance GMM, and Ward
clustering for `k=2..6`. Stochastic candidates receive 25-seed checks, and all
solutions without a sub-10-player cluster receive 100 deterministic 80%
resamples. The selected candidate remains an analysis proposal: no cluster ID
or descriptive name is written to PostgreSQL or exposed through the API/UI.

## Production player archetypes

Run the fixed V3.2C production fit from `backend/`:

```powershell
python -m analytics.player_archetypes
```

This does not repeat the V3.2B clustering search. It fits deterministic
combined-outfield K-Means with `k=2`, a 50-pass threshold, a 29-carry threshold,
and six position-normalized style dimensions. Final-third entries are omitted
because of their weak split-half stability and the stronger six-feature
internal diagnostics observed in V3.2B. The output includes eligible assignment
rows, a joblib model bundle with normalization statistics, and governed JSON
metadata with semantic centroids and separation methodology. Archetypes and
their separation margins describe style only; they are not ratings,
probabilities, or ability tiers.

## Player-style similarity

Run the historical V3.3A audit followed by the frozen V3.3B production stage:

```powershell
python -m analytics.player_similarity_analysis
python -m analytics.player_similarity
```

V3.3A reconstructs the former V1 methodology internally for research comparison;
it does not require or overwrite a legacy production similarity artifact. V3.3B
uses six pre-outcome passing/carrying style dimensions, position-specific
normalization, same-position candidates, and RMS Euclidean distance. The
cohort-calibrated 0–100 index is not a probability or quality score. Sample
support is reported separately and never changes distance, score, or rank.

## Team Intelligence and Role Fit

Run the frozen V4.0 research audit and V4.1–V4.4 production stage from
`backend/`:

```powershell
python -m analytics.team_role_research
python -m analytics.team_intelligence
```

The production team profile covers Bayer Leverkusen's 34-match product sample.
DEF, MID, and FWD roles pool events/actions and use the same six-dimensional,
position-relative coordinate system as V3.3B. External players are compared
with the full same-position role; current Leverkusen players use a leave-self-out
role. Scouting Recommendations sort eligible external players only by ascending
RMS Role Fit distance. Performance, archetype, identity, and sample support are
excluded from fit and ranking.
