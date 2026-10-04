# Offline hurdle and causal-Transformer possession-value experiment

## Research question and frozen context

This experiment asks whether explicit zero handling and possession history improve
pre-event possession-value prediction beyond the production XGBoost model. It is
**offline only**: no production model, action-value output, API, database, frontend,
or Vercel dependency was changed. The completed GRU experiment and its artifacts
remain intact.

The exact target is `future_oof_xg_same_possession`: the remaining eligible
FootyScout out-of-fold xG from the current **pre-event** state onward in the same
possession. It is non-negative, and 84.21% of states have a zero target. A shot at
the current event index can contribute to the label, not the input. The frozen
corpus has 667,062 eligible states from 38,419 possessions and 233 matches.

The earlier fixed-test RMSEs were 0.050556 for XGBoost, 0.050686 for a causal
single-head GRU, and 0.051014 for a current-state single-head MLP. Both neural
models underpredicted the fixed-test target mean of 0.014719. A two-head hurdle
model separately predicts `P(target > 0)` and the expected magnitude given a
positive target; their product is the non-negative state value. This avoids a
hard classification threshold at inference.

## Inputs, state alignment, and leakage controls

Both new models use exactly the existing 16 pre-event input fields:

- Numeric: `ball_x`, `ball_y`, `distance_to_goal`, `angle_to_goal`, `period`,
  `minute`, `score_difference`, `possession_action_number`,
  `possession_elapsed_time`, `possession_start_x`, `possession_start_y`, and
  `distance_progressed_from_possession_start`.
- Binary: `previous_action_success` and `under_pressure`.
- Categorical: `previous_action_type` and `current_play_pattern`.

No event ID, match/team/player identifier, future event, later position, goal,
provider xG, future xG, or target is fed to either model. The existing sequence
builder orders `(match_id, possession_id)` by `event_index` and maps exactly one
prediction to every eligible state. The Transformer processes each possession
once, outputting a prediction at every timestep, rather than duplicating prefixes.
Its upper-triangular attention mask blocks timesteps `> t`; a separate key-padding
mask blocks padded states. Sinusoidal positions provide explicit order. An
invariance test changes a later state's inputs and confirms earlier outputs are
unchanged. The current-state fusion path sees only the same timestep's pre-event
features.

Numeric medians, means, and standard deviations; binary missing-value modes;
and categorical vocabularies are fitted on training states only. Unknown
validation/test categories map to index zero. The upstream OOF xG labels use the
same nested match-grouped, manifest-validated construction as the frozen
possession-value training pipeline. The split is unchanged: 186 training matches
(534,490 states), 23 validation matches (63,763 states), and 24 fixed-test
matches (68,809 states). No match crosses splits. The previous fixed-test
XGBoost, MLP, and GRU predictions are joined by unique event ID and verified
against their archived labels and RMSE values; the GRU run is not regenerated.

## Architectures and optimization

The hurdle MLP uses the current-state numeric/binary inputs and learned
categorical embeddings, a 64-unit projection, a 64-unit shared hidden layer,
then a binary logit head and a Softplus conditional-magnitude head. The hybrid
Transformer uses the same input encoder, a 64-dimensional projection with
sinusoidal positions, two pre-norm causal encoder layers, four attention heads,
128-dimensional feed-forward layers, 0.1 dropout, and a separate 64-dimensional
current-state projection. It concatenates the causal history representation and
current-state projection before shared hurdle heads. Both final predictions are
`sigmoid(logit) * softplus(magnitude_logit)`.

Training uses AdamW (learning rate 0.001, weight decay 0.0001), batches of 128
possessions grouped by length to reduce padding, gradient clipping at 1, seed
42, at most 12 epochs, and patience 3. The classification loss is unweighted
`BCEWithLogitsLoss` over eligible states. The magnitude loss is SmoothL1
(`beta=0.05`) over **positive eligible** states only. Both loss weights are 1.
The train-only class imbalance is recorded in metrics; no positive-class BCE
weight was applied because it would distort raw probability calibration. The
primary positive magnitude remains on its original xG scale. Across the
original state corpus, positive targets have median 0.06829, p95 0.29249,
p99 0.46477, maximum 1.01297, and skew 2.50. A separate validation-only
`log1p` magnitude ablation checks this right tail without altering the primary
fixed-test run. Best epochs are selected by validation RMSE of the final value, then
the model and preprocessor are refitted on non-test states for exactly those
epoch counts. The fixed test does not select architecture, loss weights, or
checkpoints.

## Results, ablations, and downstream impact

The primary hurdle MLP has **6,260** trainable parameters; the hybrid
Transformer has **79,092**. Both ran on **CPU**. The hurdle MLP's best epoch was
1 and the Transformer's was 8. Train positive fraction was 15.94%; a balanced
BCE weight would have been 5.27 but was not used. The main run took 3,233.5
seconds (53.9 minutes), including 472.2 seconds of Transformer validation
training and 2,667.2 seconds of non-test refitting.

Validation metrics (same 63,763 states; XGBoost and single-head results are
read from frozen metadata, not recomputed or tuned here):

| Model | RMSE | MAE | Positive-target RMSE | Mean prediction |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | **0.056694** | 0.026774 | **0.137646** | — |
| Single-head current-state MLP | 0.057397 | 0.022300 | 0.142848 | 0.009471 |
| Single-head GRU | 0.057207 | 0.022435 | 0.142403 | 0.009723 |
| Two-head current-state MLP | 0.057328 | 0.025066 | 0.139893 | 0.013247 |
| Hybrid causal Transformer hurdle | 0.056899 | 0.023833 | 0.140637 | 0.011765 |

The XGBoost validation metadata did not record a prediction mean. Validation
target mean is 0.018369.

Fixed-test metrics (same 68,809 states and 10,170 positive targets for all five):

| Model | MAE | RMSE | R² | Pearson | Spearman | Positive MAE | Positive RMSE | Mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.023758 | **0.050556** | **0.070794** | 0.267406 | 0.217942 | **0.076756** | **0.124684** | 0.016116 |
| Single-head MLP | **0.018626** | 0.051014 | 0.053899 | 0.271383 | 0.212506 | 0.085858 | 0.130836 | 0.008073 |
| Single-head GRU | 0.019250 | 0.050686 | 0.066038 | **0.276578** | 0.207701 | 0.083891 | 0.129233 | 0.009424 |
| Hurdle MLP | 0.019550 | 0.051321 | 0.042486 | 0.232353 | 0.193845 | 0.084704 | 0.130503 | 0.009299 |
| Hybrid Transformer hurdle | 0.019518 | 0.050662 | 0.066914 | 0.276327 | **0.218656** | 0.082825 | 0.128874 | 0.009937 |

Fixed-test target mean is 0.014719. Lower all-state MAE for neural models
reflects the zero-heavy target and does not offset their worse RMSE and
positive-target errors. The hurdle MLP marginally improved positive-target
RMSE and mean bias versus the single-head MLP, but worsened overall RMSE,
correlations, and validation RMSE. The hybrid Transformer narrowly improved
the GRU's overall and positive-target RMSE, but the comparison confounds the
hurdle head with the sequence architecture. Neither surpassed XGBoost on the
primary or positive-target RMSE.

Independent fixed-test head diagnostics:

| Hurdle model | ROC-AUC | PR-AUC | Log loss | Brier | 10-bin ECE | Positive magnitude MAE | Positive magnitude RMSE | P-positive mean | Magnitude mean on positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current-state MLP | 0.657457 | 0.297105 | 0.401238 | 0.119443 | 0.033331 | 0.061917 | 0.105113 | 0.136859 | 0.066972 |
| Hybrid Transformer | **0.676786** | **0.328506** | **0.389747** | **0.115987** | **0.022304** | **0.061773** | **0.101816** | 0.131777 | **0.075361** |

Observed positive rate is 0.147800 and observed conditional magnitude mean is
0.099588. Both heads still underpredict on average. Ten-bin calibration
curves and all bucket counts are saved in metrics and figures.

The machine-readable validation and fixed-test tables, independent head
metrics, action-impact diagnostics, runtime, parameter counts, and plotted
figures are saved in `models/experiments/possession_transformer/metrics.json`.
The five-way fixed-test predictions and exact same-possession before/after
action values are saved under
`data/processed/experiments/possession_transformer/`. These local artifacts are
ignored by Git and must be regenerated to reproduce the numeric results.

The baseline ablation is the archived single-head current-state MLP; the new
two-head current-state MLP isolates explicit zero handling. The archived GRU
versus the hybrid Transformer compares different sequence architectures *and*
loss heads, so that difference alone cannot isolate the attention mechanism.
The validation-only Transformer without current-state fusion reached RMSE
0.057161 (73,204 parameters, best epoch 6), versus 0.056899 for the fused
Transformer. This is a modest fusion benefit on validation, not a fixed-test
claim. The validation-only `log1p` positive-magnitude MLP reached RMSE
0.056950 (best epoch 8), versus 0.057328 for the primary untransformed hurdle
MLP. Both optional ablations were run **after** the main fixed-test result was
seen; they were not evaluated on that test set and must be treated as
exploratory, not as a new unbiased model selection. Their histories, metrics,
and checkpoints have separate files in the experiment artifact directory.

Attacking impact is **not** redefined: for each of 41,433 held-out pass/carry
actions (23,054 passes, 18,379 carries), the
existing same-possession transition computes `V(after) - V(before)`. A failed
or terminal action has an after-value of zero; a successful nonterminal action
uses the next eligible pre-event state from its own possession. The saved
diagnostics describe correlations with XGBoost, distributions by action type,
and largest disagreements. The hybrid impact distribution had Pearson
correlation 0.923879 and Spearman 0.753227 with XGBoost; its mean was
-0.000166, median -0.000265, standard deviation 0.008742, and 43.73% of
actions had positive impact. XGBoost's corresponding values were -0.000086,
+0.000074, 0.010726, and 53.72%. Hybrid pass impact mean was -0.000148
(55.50% positive); carry mean was -0.000189 (28.96% positive), versus
XGBoost carry mean +0.001088 (58.27% positive). Hybrid extrema were -0.180353
and +0.207000; the saved JSON lists p01/p99 and the ten largest disagreements.
This consequential difference in carry signs is another reason not to migrate.
None of these experimental values enter production player ratings.

## Limitations and production decision

The target is mostly zero and the positive tail is difficult to predict.
Observational event states do not encode every defensive or tactical context.
One fixed held-out allocation and one compact configuration do not establish
generalization across seasons and leagues. Lower all-state MAE alone is not
evidence of improvement if positive-target RMSE, calibration, or prediction
bias deteriorate. A replacement would require a meaningful held-out improvement
over the XGBoost RMSE of 0.050556 and positive-target RMSE of 0.124684, with
acceptable probability calibration and action-impact behavior. Until separate
review and migration, **keep the production XGBoost model in place**. Further
offline work should prioritize positive-tail calibration and the carry-impact
disagreement before considering a migration.
