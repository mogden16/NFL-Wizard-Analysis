# Phase 6 predeclared comparison protocol

This protocol was written before running any Phase 6 validation experiment.
Only frozen Phase 5 player rows and 2017-2024 sources may be read. The 2025
holdout and September 13, 2026 diagnostic exhibition are sealed. Phase 4.6
prediction/settlement controls are unchanged.

The common formal sample uses 2022 for component fitting, 2023 for separate
chronological calibration and out-of-fold blend-weight selection, and 2024
for validation. The 2021 strict market season is the Phase 4 warmup; using it
as a player-model fit year would give in-sample team expectations. Each 2023
blend-training component prediction must be produced by a model fitted on
2022 only, with no 2023 or later observations in the component fit. Raw
component predictions, before calibrators fitted on 2023, enter the blend.

The frozen Phase 5 hierarchical 2024 log loss is **0.3804**, the promotion
benchmark. A challenger must lower the exact 2024 hierarchical log loss by
at least **0.005 absolute**, with a game-cluster bootstrap 95% confidence
interval for its paired log-loss difference wholly below zero. Its ECE may
not exceed the hierarchy's ECE by more than 0.010, its calibration slope
must lie between 0.8 and 1.2, no position's log loss may worsen by over
0.010, and it must improve in at least two of the three fixed chronological
segments (weeks 2-6, 7-12, and 13+), with no segment worsening by more than
0.005. These thresholds are fixed before viewing Phase 6 results. If none
passes, `PHASE6_CHAMPION` remains `phase5_hierarchical`.

Position models are one logistic model per RB, WR, TE and QB, using a small
predeclared set of role-specific lagged fields. RB/WR/TE each have a controlled
core-usage and core-usage-plus-xTD ablation. The latter is the position
component eligible for the fixed ensemble; ablations do not tune ensemble
composition. Position models may be sigmoid calibrated on 2023 if 2024 log
loss, Brier and ECE all improve. QB passing TDs remain excluded from labels.

One LightGBM model family uses the same disciplined team, role, xTD and
lagged-opponent fields. Three predeclared configurations are evaluated: shallow
conservative, shallow medium, and moderate. The conservative configuration is
the sole tree component eligible for the 2023 out-of-fold blend; choosing a
configuration using 2024 outcomes cannot retroactively affect 2023 OOF
predictions. Each configuration and its metrics are recorded. Sigmoid
calibration may use 2023 only under the same quality rule as above.

The five fixed blend candidates use weights in order (hierarchy, pooled raw
logistic, position raw logistic, conservative tree raw):

1. (1, 0, 0, 0) — hierarchy control
2. (0.75, 0, 0.25, 0)
3. (0.75, 0, 0, 0.25)
4. (0.50, 0.25, 0.25, 0)
5. (0.50, 0.25, 0, 0.25)

Choose one blend on 2023 OOF log loss (Brier tie-break), then evaluate it
once on 2024. Do not fit a meta-model because only one chronological OOF
season exists. No player ATD price, market residual, betting return, or
threshold enters football training or model promotion. Compare T-60 market
prices only on the previously matched 2024 Phase 1 sample.
