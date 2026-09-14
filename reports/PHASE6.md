# Phase 6 — nonlinear and position-specific player models

All Phase 0-5 predictive artifacts and Phase 4.6 replay controls are unchanged. The 2025 holdout and September 13, 2026 exhibition were not loaded. No betting result or threshold informed selection.

## Design and population

The frozen Phase 5 player universe gives 6,441 player-games for 2022 fitting, 6,321 2023 chronological OOF/calibration rows, and 6,320 2024 validation rows. The 2021 strict-market season warms up the Phase 4 team expectation, so there is no prior-market fit for 2021 player rows. All selected team expectations and player opportunity features use earlier games. The prior-game publication time is the documented kickoff +24h assumption, not an observed publication timestamp.
Model choices and promotion gates were fixed in `docs/PHASE6_PROTOCOL.md` before Phase 6 validation. The required absolute log-loss improvement is 0.005 plus a wholly negative game-cluster bootstrap interval, acceptable calibration, position results and three-segment stability.

## Overall 2024 validation

| Model | Player-games | Log loss | Brier | ECE | Cal intercept | Cal slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_td | 6,320 | 0.4068 | 0.1241 | 0.0206 | -0.320 | 0.845 |
| phase5_hierarchical | 6,320 | 0.3804 | 0.1178 | 0.0165 | +0.205 | 1.174 |
| phase5_pooled_logistic | 6,320 | 0.3831 | 0.1189 | 0.0237 | -0.010 | 0.981 |
| position_logistic | 6,320 | 0.3835 | 0.1186 | 0.0143 | +0.009 | 0.979 |
| lightgbm | 6,320 | 0.3807 | 0.1178 | 0.0064 | +0.001 | 0.991 |
| oof_blend | 6,320 | 0.3801 | 0.1178 | 0.0176 | +0.250 | 1.145 |

ROC-AUC and PR-AUC are secondary metrics in `reports/phase6_metrics.json`. Every model's overall, position and probability-bucket calibration tables are stored there. The historical and Phase 5 columns are unchanged frozen benchmarks.

## Role-specific logistic models and xTD ablation

| Position | Fit rows | Fit TD+ | Core log loss | Core+xTD log loss | xTD minus core | Selected log loss | Hierarchy log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RB | 1,689 | 322 | 0.4518 | 0.4534 | +0.0016 | 0.4534 | 0.4447 |
| WR | 2,580 | 373 | 0.4011 | 0.4011 | +0.0000 | 0.3987 | 0.3974 |
| TE | 1,457 | 152 | 0.3223 | 0.3231 | +0.0008 | 0.3231 | 0.3208 |
| QB | 715 | 77 | — | — | — | 0.2988 | 0.2998 |

Position-level probability diagnostics for each serious model:

| Position | Model | Rows | Log loss | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| RB | phase5_hierarchical | 1,584 | 0.4447 | 0.1429 | 0.0252 |
| RB | phase5_pooled_logistic | 1,584 | 0.4562 | 0.1481 | 0.0355 |
| RB | position_logistic | 1,584 | 0.4534 | 0.1457 | 0.0255 |
| RB | lightgbm | 1,584 | 0.4486 | 0.1441 | 0.0253 |
| RB | oof_blend | 1,584 | 0.4454 | 0.1436 | 0.0210 |
| WR | phase5_hierarchical | 2,567 | 0.3974 | 0.1251 | 0.0213 |
| WR | phase5_pooled_logistic | 2,567 | 0.3978 | 0.1247 | 0.0245 |
| WR | position_logistic | 2,567 | 0.3987 | 0.1252 | 0.0157 |
| WR | lightgbm | 2,567 | 0.3980 | 0.1251 | 0.0139 |
| WR | oof_blend | 2,567 | 0.3978 | 0.1251 | 0.0240 |
| TE | phase5_hierarchical | 1,394 | 0.3208 | 0.0936 | 0.0119 |
| TE | phase5_pooled_logistic | 1,394 | 0.3213 | 0.0936 | 0.0217 |
| TE | position_logistic | 1,394 | 0.3231 | 0.0945 | 0.0183 |
| TE | lightgbm | 1,394 | 0.3208 | 0.0937 | 0.0033 |
| TE | oof_blend | 1,394 | 0.3203 | 0.0935 | 0.0168 |
| QB | phase5_hierarchical | 775 | 0.2998 | 0.0853 | 0.0455 |
| QB | phase5_pooled_logistic | 775 | 0.2965 | 0.0851 | 0.0294 |
| QB | position_logistic | 775 | 0.2988 | 0.0849 | 0.0135 |
| QB | lightgbm | 775 | 0.2921 | 0.0832 | 0.0197 |
| QB | oof_blend | 775 | 0.2960 | 0.0847 | 0.0320 |

QB designed-run and scramble flags, WR/TE air-yard history, and route participation are not present as safe lagged fields in the frozen player table. QB carries and red-zone carries are proxies; passing TDs never enter QB labels. All opponent inputs are Phase 2 lagged allowed-volume context, not opponent-adjusted EPA. Logistic models use training-median imputation with missing indicators; LightGBM sees native nulls. The per-field, per-season missingness matrix is in the metrics JSON.

## LightGBM registry

| Configuration | 2024 log loss | Brier | ECE | Calibration |
| --- | ---: | ---: | ---: | --- |
| conservative | 0.3807 | 0.1178 | 0.0072 | sigmoid |
| shallow_medium | 0.3807 | 0.1178 | 0.0064 | sigmoid |
| moderate | 0.3818 | 0.1181 | 0.0116 | sigmoid |

Selected standalone LightGBM configuration: **shallow_medium**. The conservative configuration alone was eligible for OOF blending, regardless of validation ranking.
The tree gain and joint split-path diagnostics in the metrics JSON describe whether predeclared opportunity/environment pairs appeared in learned paths; they do not establish causal interactions.

| Highest gain feature | Split gain |
| --- | ---: |
| last5_touch_share | 6503.7 |
| last5_snap_share | 1831.7 |
| last5_target_share | 1428.5 |
| last5_carry_share | 945.4 |
| expected_team_td | 668.2 |
| last5_total_xtd | 418.3 |
| last5_total_xtd_share | 281.9 |
| implied_team_points | 275.2 |

| Hypothesized pair sharing a tree path | Split-pair count |
| --- | ---: |
| last5_goal_line_opportunity_share / expected_team_td | 5 |
| last5_red_zone_target_share / implied_team_points | 5 |
| last5_carry_share / team_spread | 1 |
| last5_end_zone_target_share / implied_team_points | 1 |
| last5_total_xtd_share / last5_opponent_allowed_red_zone_carries_per_game | 2 |

These counts show that some interactions were available to the fitted tree; they do not show improved 2024 probability quality.

## Chronological OOF blend

All 2023 component predictions came from models fitted on 2022 only. The hierarchy used its 2022 allocation fit; pooled and position logistic and the conservative LightGBM used 2022-only fits. No calibrator trained on 2023 generated a 2023 blend-training input. Blend weights were selected on 2023 OOF log loss and evaluated once on 2024.

| Fixed blend | 2023 OOF log loss | 2024 log loss |
| --- | ---: | ---: |
| hierarchy_control | 0.3722 | 0.3804 |
| hierarchy_position | 0.3708 | 0.3799 |
| hierarchy_tree | 0.3713 | 0.3800 |
| hierarchy_pool_position | 0.3701 | 0.3801 |
| hierarchy_pool_tree | 0.3703 | 0.3800 |

Selected by 2023 OOF: **hierarchy_pool_position**.

## Promotion and uncertainty

| Challenger | Difference vs hierarchy | Game bootstrap 95% CI | Material gate | All gates |
| --- | ---: | ---: | --- | --- |
| position_logistic | +0.0031 | [-0.0003, +0.0066] | fail | fail |
| lightgbm | +0.0003 | [-0.0024, +0.0030] | fail | fail |
| oof_blend | -0.0003 | [-0.0016, +0.0011] | fail | fail |

The full gate outcomes, position changes and early/middle/late validation metrics are in the metrics JSON. The bootstrap resamples 2024 games, preserving within-game player correlation.

## T-60 player market benchmark

Only 223 matched players across 10 frozen Phase 1 games have a strict T-60 player price. These raw implied probabilities are a benchmark only; no price enters football models or blend selection.

| Position | Matched rows | Champion log loss | Raw market log loss | Mean football minus market P |
| --- | ---: | ---: | ---: | ---: |
| RB | 59 | 0.4233 | 0.4269 | -0.069 |
| WR | 95 | 0.4472 | 0.3949 | -0.060 |
| TE | 48 | 0.3456 | 0.3546 | -0.033 |
| QB | 21 | 0.3513 | 0.3497 | -0.064 |

Matched overall: champion 0.4100 vs market 0.3904 log loss. Football probabilities were below raw market implied probabilities in all four positions. Raw implied prices contain bookmaker margin, so this sign alone does not establish systematic overpricing. Position residuals are descriptive; ten games cannot establish a stable pattern or justify a betting threshold.

## Answers and limits

- **Do position models improve on pooled models?** No overall. Position logistic log loss 0.3835 versus frozen pooled logistic 0.3831 and hierarchy 0.3804; see position table.
- **Does LightGBM materially beat hierarchy?** No; it fails at least one predeclared gate.
- **Does xTD add within a position?** No clear 2024 log-loss gain. RB xTD-minus-core log loss +0.0016; WR xTD-minus-core log loss +0.0000; TE xTD-minus-core log loss +0.0008. Negative values favor xTD; small differences are not strong evidence.
- **Does the OOF ensemble materially improve?** No; it fails at least one predeclared gate.
- **Which model advances to Phase 7?** phase5_hierarchical.
- **Does market still outperform best football model?** Yes on 223 matched 2024 T-60 player quotes; this is a narrow sample.

**PHASE6_CHAMPION = phase5_hierarchical**

The 2025 holdout remains unopened. Phase 7 betting simulation was not started. Validation predictions and chronological OOF components are in `reports/phase6_validation_predictions.csv` and `reports/phase6_2023_oof_components.csv`.
