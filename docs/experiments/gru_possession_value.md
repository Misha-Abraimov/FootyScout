# Offline GRU possession-value experiment

## Question and frozen baseline

Can causal possession history improve prediction of the **same** pre-event state
target used by the production XGBoost attacking-value model? This is an offline
experiment only. No production weights, predictions, database values, API responses,
or frontend metrics were changed.

The archived production selection is a pseudo-Huber XGBoost regressor with 300
boosting rounds, chosen by validation RMSE. The experiment reconstructs its fixed
test prediction using the original frozen split, nested xG labels, feature contract,
preprocessor, and selected configuration; reconstructed RMSE matches the archived
0.0505564613 value to within 1e-6.

## Exact state and target

`data/processed/possession_states.parquet` has 667,062 eligible **pre-event** states
from 38,419 possessions across 233 matches. The state at event index `t` is immediately
before that event. Its target, `future_oof_xg_same_possession`, is the sum of eligible
FootyScout out-of-fold shot xG from event `t` onward in the same possession. Thus an
eligible shot at `t` belongs to the target, not to the inputs. The target ranges from
0 to 1.012965 and is zero in 84.21% of states; this supports a non-negative output.

Upstream xG is nested exactly as in the existing XGBoost pipeline: inner match-grouped
OOF xG for outer training matches, and outer-train-only xG predictions for outer
holdouts. Provider xG, goals, shot outcome, future xG, and future events are never
input features. Cached nested labels are manifest-validated against the frozen xG
specification. The fixed test matches were not used for neural early stopping.

## Inputs and sequence representation

The 16 inputs are the unchanged production state-feature contract:

- Continuous (12): `ball_x`, `ball_y`, `distance_to_goal`, `angle_to_goal`,
  `period`, `minute`, `score_difference`, `possession_action_number`,
  `possession_elapsed_time`, `possession_start_x`, `possession_start_y`,
  `distance_progressed_from_possession_start`.
- Binary (2): `previous_action_success`, `under_pressure`.
- Categorical (2): `previous_action_type`, `current_play_pattern`.

Numeric median imputation, mean, and standard deviation are fitted on training
states only. Binary missing-value modes and sorted categorical vocabularies are also
fitted on training only. Unseen categories use index 0 (`Unknown`). A small embedding
is useful here because the two categorical inputs are strings rather than already
numeric encodings. No validation/test category is added to a fitted vocabulary.

Rows are grouped by `(match_id, possession_id)` and ordered by authoritative
`event_index`. Each possession is processed once by a **unidirectional** GRU; the
output at `t` uses only features from timesteps `<= t`. Padded positions are masked
from the loss. No separately materialized prefixes or bidirectional recurrence are
used. A current-state MLP receives precisely the same preprocessed features but no
history.

## Architecture, optimization, and split

The GRU uses per-category embeddings (16/10 fitted vocabulary sizes with 8/5
embedding dimensions), a 64-unit input projection, one 64-unit unidirectional GRU
layer, and a 64→32→1 regression head with Softplus. It has **29,043 trainable
parameters**. The MLP ablation has **4,083**.

Both use masked SmoothL1/Huber loss (`beta=0.05`), AdamW (`lr=0.001`, weight decay
`0.0001`), batch size 256 possessions, gradient clip 1.0, seed 42, up to 18 epochs,
and patience 4. The best checkpoint is selected by validation RMSE, matching the
baseline selection metric. The selected epoch count is then refitted on **non-test**
matches using freshly fitted non-test preprocessing, as the frozen XGBoost pipeline
does before its fixed-test evaluation. Both neural models use a non-negative Softplus
output, confirmed against the actual target distribution.

The frozen match allocation is 186 train matches (534,490 states), 23 validation
matches (63,763 states), and 24 untouched test matches (68,809 states). All states
from a match remain in one split; no possession crosses a split. Best validation
epochs were 6 (GRU) and 8 (MLP).

## Held-out results

All three models were evaluated on the **same 68,809 fixed-test pre-event states**
and the same nested xG target. Lower MAE/RMSE is better; higher R² and correlation
is better. MAE is the average absolute error, RMSE emphasizes larger errors, R²
compares squared error with a mean baseline, and Pearson/Spearman measure linear/rank
association rather than calibration.

| Model | MAE | RMSE | R² | Pearson | Spearman | Positive-target MAE | Positive-target RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen XGBoost | 0.023758 | **0.050556** | **0.070794** | 0.267406 | **0.217942** | **0.076756** | **0.124684** |
| Causal GRU | 0.019250 | 0.050686 | 0.066038 | **0.276578** | 0.207701 | 0.083891 | 0.129233 |
| Current-state MLP | **0.018626** | 0.051014 | 0.053899 | 0.271383 | 0.212506 | 0.085858 | 0.130836 |

Validation RMSE was 0.056694 for the archived XGBoost selection, 0.057207 for
the GRU, and 0.057397 for the MLP. Test target mean was 0.014719; mean predictions
were 0.016116 (XGBoost), 0.009424 (GRU), and 0.008073 (MLP). The zero-heavy target
explains why the neural models' lower all-state MAE does not mean better positive
state prediction: they underpredict larger opportunities more strongly. On the
positive-target subset (10,170 states), both neural models are worse on MAE/RMSE.

The sequence history **helped modestly relative to the same-input MLP** on RMSE,
R², Pearson, and positive-target errors; it did not improve the predeclared RMSE
relative to XGBoost. These small differences should not be read as proof of a
generalizable sequence benefit from one split and one modest configuration.

## Downstream attacking-impact check

Analysis-only values use the frozen alignment rule: for a completed pass or carry,
`V(after)` is the next eligible pre-event state in the **same possession**; for a
failed or ending action it is zero. `V(after) - V(before)` was computed for both
model predictions on 41,433 fixed-test passes/carries (23,054 passes; 18,379
carries). No GRU values were written to production.

| Model | Mean impact | Median | Positive actions | 1st pct. | 99th pct. | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | -0.000086 | +0.000074 | 53.72% | -0.033156 | +0.026063 | -0.211525 | +0.195286 |
| GRU | +0.000111 | -0.000092 | 46.69% | -0.019843 | +0.026254 | -0.194484 | +0.263756 |

Action-value Pearson correlation between models was 0.9003. XGBoost mean impact
was +0.001088 for carries and -0.001022 for passes; GRU was +0.000650 and
-0.000320, respectively. The largest absolute disagreement was 0.147893 on a
single pass; the ten largest cases are in `metrics.json`. Correlation is high, but
the positive-action fraction and extremes differ enough that production player
ratings should **not** be swapped without deeper action-level validation.

## Artifacts, runtime, and limitations

Generated, ignored offline artifacts:

- `models/experiments/possession_gru/`: best validation GRU/MLP checkpoints,
  non-test-refit test checkpoints, configuration and metrics, training history,
  training-only/pre-test preprocessing metadata, and static learning/state/action plots.
- `data/processed/experiments/possession_gru/`: one prediction per fixed-test
  state and the corresponding pass/carry action comparisons.

The run used PyTorch 2.14 CPU only (CUDA unavailable). Total elapsed time was
552.9 seconds: GRU validation training 278.7 s, MLP validation training 45.2 s,
XGBoost fixed-test reconstruction 11.5 s, GRU non-test refit 163.4 s, and MLP
non-test refit 31.7 s. The remainder includes loading, preprocessing, evaluation,
and plots. Neural checkpoint reload was verified.

Limitations: one frozen match split, one predeclared GRU size/configuration, a
zero-heavy provider-independent OOF-xG-derived target, imperfect off-ball context,
and limited evidence about action-level reliability. The GRU may benefit from more
careful validation-only loss/output calibration or a richer **causal** sequence
representation, but none was selected using the test set here.

**Recommendation: keep production XGBoost.** The GRU provides no held-out RMSE or
positive-target gain, while downstream impact signs and tails shift. Investigate
further offline only if sequence modeling is still a research priority.
