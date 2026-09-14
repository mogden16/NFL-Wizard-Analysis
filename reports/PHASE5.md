# Phase 5 — player anytime-touchdown probability baselines

Phase 1-4.6 inputs and predictive architectures are frozen. No 2025 file or
September 13, 2026 outcome/price/result was read. The 2026 exhibition
remains excluded from all modeling decisions. This phase fits no tree or
ensemble and performs no betting-return or threshold optimization.

## Population and chronology

The prior-game-only universe contains **48,674** player-games in 2017-2024. It uses a player who appeared for the same team in one of the previous three team games and had a carry or target in that prior three-game window; a QB with at least 20 prior snaps also qualifies. No current-game participation, carries, targets, snaps or TD is used to choose a row. Rookie debuts and first-team games are consequently absent. RB, WR, TE and QB are included. FB is omitted because the frozen Phase 2 feature pipeline has no FB rows or comparable lagged red-zone/snap fields; adding an incomplete FB feature definition solely for this phase would make model comparisons inconsistent.
The common strict-market comparison contains **6,441** 2022 base-fit rows, **6,321** 2023 calibration-only rows, and **6,320** 2024 validation rows (TD rate **15.82%**). 2017-2021 contributes to historical scoring priors and coverage diagnostics. Phase 3 chronological xTD is unavailable in 2017; strict T-60 team market inputs begin in 2021, which is a warmup season for the Phase 4 team market model. Team expectations are scored out of sample with an expanding prior-season fit: 2021 for 2022, 2021-2022 for 2023, and 2021-2023 for 2024. The 2025 holdout is not loaded.
Historical prior-game statistics retain the **assumed kickoff + 24h** availability proxy. It is not an observed publication timestamp. Phase 3 xTD was scored with models fit on earlier seasons, and the current game's xTD is excluded.

## 2024 common-row validation

| Probability | Player-games | Log loss | Brier | ECE | Cal intercept | Cal slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_td | 6,320 | 0.4068 | 0.1241 | 0.0206 | -0.320 | 0.845 |
| hierarchical | 6,320 | 0.3804 | 0.1178 | 0.0165 | +0.205 | 1.174 |
| logistic_raw | 6,320 | 0.3839 | 0.1191 | 0.0263 | +0.172 | 1.037 |
| logistic_selected | 6,320 | 0.3831 | 0.1189 | 0.0237 | -0.010 | 0.981 |

Historical TD rate uses the ewma candidate selected on 2024 validation. The logistic calibration choice is sigmoid; its sigmoid was fitted only to 2023 base-model predictions, then compared with raw probabilities on 2024. ROC-AUC and PR-AUC, which are secondary, and full metrics for all four TD-rate windows are in the metrics JSON.

The 2024 market comparison is limited to **223** player-games across ten frozen Phase 1 week-four games. No new historical-odds credits were spent. These already-captured T-60 quotes are a separate raw implied-probability benchmark, never a football-model input; coverage is too small for empirical market calibration.

| Model on quoted subset | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| historical_td | 0.4601 | 0.1443 | 0.0492 |
| hierarchical | 0.4100 | 0.1308 | 0.0288 |
| logistic | 0.4159 | 0.1321 | 0.0482 |
| market_raw | 0.3904 | 0.1217 | 0.0708 |

## Controlled logistic feature-family tests

All six models use identical 2024 rows, the same 2022 base-fit period and the same fixed logistic regularization. The 2023 rows are reserved for calibration; feature-family results below use raw 2024 predictions. The reduced base-fit period is necessary because 2021 has no prior strict-market season from which to generate an out-of-sample team expectation.

| Family | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| A_actual_td_only | 0.4103 | 0.1256 | 0.0168 |
| B_xtd_only | 0.4014 | 0.1239 | 0.0437 |
| C_usage_actual_td | 0.3872 | 0.1207 | 0.0325 |
| D_usage_xtd | 0.3878 | 0.1206 | 0.0309 |
| E_team_usage_xtd | 0.3839 | 0.1191 | 0.0263 |
| F_team_usage_xtd_actual_td | 0.3838 | 0.1190 | 0.0267 |

Selected logistic stability within 2024:

| Segment | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: |
| weeks_2_to_9 | 0.3914 | 0.1217 | 0.0192 |
| weeks_10_plus | 0.3762 | 0.1165 | 0.0275 |

## Acceptance answers

- **Does xTD beat recent actual TD?** No with usage controlled: D minus C log loss +0.00065. xTD alone minus actual TD alone is -0.00893; conclusions differ by comparison.
- **Does hierarchy beat historical TD?** Yes on log loss.
- **Does logistic beat hierarchy?** No on log loss.
- **Overall calibration** Selected logistic ECE 0.0237, intercept -0.010, slope 0.981.
- **Position calibration** RB mean P 21.7% vs actual 21.3%; WR mean P 15.5% vs actual 16.6%; TE mean P 11.6% vs actual 11.3%; QB mean P 10.6% vs actual 10.3%.
- **Does football beat market?** No on the 223-row matched 2024 subset by log loss; this limited sample is not conclusive.
- **Does independent logistic violate team coherence?** No large systematic gap: mean lambda-minus-team -0.033 TD. Mean absolute team-game gap 0.253 TD; 11.1% of team-games differ by over 0.5 TD.
- **Does actual TD add after xTD and role?** F minus E log loss -0.00016; this is too small to justify adding the feature to the selected model.

## Hierarchical allocation and team coherence

Nonnegative allocation weights were fit to 2022 player TD labels by training Brier loss. The fixed feature families were declared before fitting. The weighted role score omits unavailable families, then each team-game normalizes player shares to one. No September 13 result or Phase 4.5 weight was used.

| Lagged feature | Fitted weight |
| --- | ---: |
| last5_total_xtd_share | 0.0460 |
| last5_rushing_xtd_share | 0.0000 |
| last5_receiving_xtd_share | 0.0101 |
| last5_goal_line_opportunity_share | 0.0219 |
| last5_red_zone_target_share | 0.0000 |
| last5_end_zone_target_share | 0.0000 |
| last5_carry_share | 0.3462 |
| last5_target_share | 0.5342 |
| last5_snap_share | 0.0416 |

Across 512 validation team-games, hierarchical player expected TD sums differed from Phase 4 expected team TD by at most 0.000000. The independent logistic model's Poisson-equivalent player lambda sum minus team expectation averaged -0.033 TD, with mean absolute gap 0.253; 11.1% of team-games exceeded a 0.5-TD absolute gap. No coherence adjustment was applied to logistic predictions.

## Calibration by position

| Position | Rows | TD rate | Mean P | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RB | 1,584 | 21.28% | 21.75% | 0.4562 | 0.1481 | 0.0355 |
| WR | 2,567 | 16.60% | 15.52% | 0.3978 | 0.1247 | 0.0245 |
| TE | 1,394 | 11.26% | 11.59% | 0.3213 | 0.0936 | 0.0217 |
| QB | 775 | 10.32% | 10.62% | 0.2965 | 0.0851 | 0.0294 |

## Selected-logistic calibration buckets

| Predicted P | Player-games | Mean P | Actual TD rate |
| --- | ---: | ---: | ---: | ---: |
| <10% | 2,784 | 6.27% | 5.14% |
| 10-20% | 1,935 | 14.24% | 16.12% |
| 20-30% | 841 | 24.45% | 28.42% |
| 30-40% | 371 | 34.34% | 37.47% |
| 40-50% | 202 | 44.59% | 38.12% |
| 50-60% | 104 | 54.79% | 49.04% |
| 60%+ | 83 | 68.14% | 46.99% |

## Logistic coefficients and feature diagnostics

Coefficients are for standardized, imputed inputs. Approximate Fisher intervals ignore model-selection uncertainty and are descriptive. Correlated feature groups can cause unstable individual signs.

| Feature | Standardized coefficient | Approx. 95% interval |
| --- | ---: | ---: |
| expected_team_td | -0.108 | [-0.628, +0.412] |
| implied_team_points | +0.326 | [-0.207, +0.858] |
| home | +0.015 | [-0.061, +0.091] |
| last5_carry_share | +0.103 | [-0.243, +0.449] |
| last5_target_share | +0.103 | [-0.096, +0.302] |
| last5_touch_share | +0.396 | [+0.055, +0.737] |
| last5_snap_share | +0.469 | [+0.318, +0.619] |
| last5_goal_line_opportunity_share | +0.072 | [-0.042, +0.186] |
| last5_inside_10_carries_per_game | -0.051 | [-0.236, +0.134] |
| last5_inside_5_carries_per_game | -0.118 | [-0.307, +0.072] |
| last5_red_zone_target_share | -0.052 | [-0.171, +0.067] |
| last5_end_zone_target_share | +0.015 | [-0.099, +0.129] |
| last5_rushing_xtd | +0.197 | [-0.547, +0.940] |
| last5_receiving_xtd | +0.013 | [-0.736, +0.761] |
| last5_total_xtd | +0.164 | [-0.764, +1.093] |
| last5_total_xtd_per_opportunity | -0.046 | [-0.180, +0.088] |
| last5_rushing_xtd_share | -0.077 | [-0.289, +0.135] |
| last5_receiving_xtd_share | +0.101 | [-0.160, +0.361] |
| last5_total_xtd_share | -0.207 | [-0.472, +0.058] |
| is_rb | +0.056 | [-0.677, +0.789] |
| is_wr | +0.040 | [-0.771, +0.851] |
| is_te | +0.017 | [-0.676, +0.710] |
| is_qb | -0.164 | [-0.694, +0.367] |
| missingindicator_last5_snap_share | -0.096 | [-0.791, +0.599] |
| missingindicator_last5_goal_line_opportunity_share | +0.034 | [-0.053, +0.121] |
| missingindicator_last5_red_zone_target_share | -0.038 | [-0.138, +0.062] |
| missingindicator_last5_end_zone_target_share | -0.019 | [-0.112, +0.074] |
| missingindicator_last5_rushing_xtd | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_receiving_xtd | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_total_xtd | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_total_xtd_per_opportunity | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_rushing_xtd_share | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_receiving_xtd_share | -0.031 | [-1.332, +1.271] |
| missingindicator_last5_total_xtd_share | -0.031 | [-1.332, +1.271] |

Highest absolute feature correlations: missingindicator_last5_rushing_xtd / missingindicator_last5_receiving_xtd 1.00, missingindicator_last5_rushing_xtd / missingindicator_last5_total_xtd 1.00, missingindicator_last5_rushing_xtd / missingindicator_last5_total_xtd_per_opportunity 1.00, missingindicator_last5_rushing_xtd / missingindicator_last5_rushing_xtd_share 1.00, missingindicator_last5_rushing_xtd / missingindicator_last5_receiving_xtd_share 1.00, missingindicator_last5_rushing_xtd / missingindicator_last5_total_xtd_share 1.00, missingindicator_last5_receiving_xtd / missingindicator_last5_total_xtd 1.00, missingindicator_last5_receiving_xtd / missingindicator_last5_total_xtd_per_opportunity 1.00.
The standalone expected-team-TD coefficient is negative while implied points is positive. These two environment signals are correlated, and the expected-team-TD interval crosses zero; the combined effect should not be read as a causal decrease. Goal-line and red-zone target signs are small and uncertain. All xTD missing indicators are identical because the source is joined as one family; their individual coefficients are not separately interpretable.

## Missingness and limitations

Route participation is unavailable and excluded, never imputed as zero. All selected numeric families retain nulls in the feature table; the logistic pipeline adds missing indicators and imputes from the training distribution. FB, rookie debuts, players changing teams midseason before an appearance, and long-inactive players are outside this prior-known universe. Historical pregame active rosters are unavailable, so the prior-appearance rule is a proxy, not proof of actual active status.
The market comparison covers only ten games and may be too small to resolve a close model difference. 2024 is validation/model selection, not an untouched final test. Some xTD feature missingness reflects no earlier scored opportunity rather than zero scoring quality. 2025 remains the untouched historical holdout.

| Feature | 2022 null | 2023 null | 2024 null |
| --- | ---: | ---: | ---: |
| expected_team_td | 0.00% | 0.00% | 0.00% |
| implied_team_points | 0.00% | 0.00% | 0.00% |
| last5_carry_share | 0.00% | 0.00% | 0.00% |
| last5_end_zone_target_share | 2.61% | 2.59% | 1.34% |
| last5_goal_line_opportunity_share | 3.62% | 4.51% | 2.56% |
| last5_inside_10_carries_per_game | 0.00% | 0.00% | 0.00% |
| last5_inside_5_carries_per_game | 0.00% | 0.00% | 0.00% |
| last5_receiving_xtd | 0.17% | 0.90% | 0.40% |
| last5_receiving_xtd_share | 0.17% | 0.90% | 0.40% |
| last5_red_zone_target_share | 0.65% | 0.55% | 0.35% |
| last5_rushing_xtd | 0.17% | 0.90% | 0.40% |
| last5_rushing_xtd_share | 0.17% | 0.90% | 0.40% |
| last5_snap_share | 0.05% | 0.08% | 0.16% |
| last5_target_share | 0.00% | 0.00% | 0.00% |
| last5_total_xtd | 0.17% | 0.90% | 0.40% |
| last5_total_xtd_per_opportunity | 0.17% | 0.90% | 0.40% |
| last5_total_xtd_share | 0.17% | 0.90% | 0.40% |
| last5_touch_share | 0.00% | 0.00% | 0.00% |
| last8_td_rate | 0.00% | 0.00% | 0.00% |

## Representative 2024 observations

These are diagnostic examples, including missed outcomes; a single binary result does not validate an individual probability. Early-week historical TD rates can rest on only one prior game.

| Pattern | Game | Player | Prior last-8 TD rate | Prior last-5 xTD | History P | Hierarchy P | Logistic P | Actual TD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low TD / high xTD, scored | 2024_08_ARI_MIA | Raheem Mostert | 0.0% | 1.082 | 11.3% | 34.3% | 39.2% | 1 |
| Low TD / high xTD, scored | 2024_02_LV_BAL | Davante Adams | 0.0% | 0.741 | 12.2% | 19.6% | 17.6% | 1 |
| Low TD / high xTD, missed | 2024_02_BUF_MIA | Raheem Mostert | 0.0% | 1.385 | 15.3% | 33.9% | 43.6% | 0 |
| Low TD / high xTD, missed | 2024_03_MIA_SEA | Raheem Mostert | 0.0% | 1.385 | 15.3% | 20.6% | 33.2% | 0 |
| High TD / low xTD, missed | 2024_02_BUF_MIA | Khalil Shakir | 100.0% | 0.099 | 37.2% | 16.7% | 12.4% | 0 |
| High TD / low xTD, missed | 2024_02_LAC_CAR | Bryce Young | 100.0% | 0.090 | 32.5% | 15.9% | 7.3% | 0 |
| High TD / low xTD, scored | 2024_02_LV_BAL | Alexander Mattison | 100.0% | 0.040 | 40.3% | 25.3% | 9.1% | 1 |
| High TD / low xTD, scored | 2024_02_SF_MIN | Jalen Nailor | 100.0% | 0.094 | 37.2% | 8.6% | 8.8% | 1 |
| Logistic / hierarchy disagree | 2024_12_SF_GB | Christian McCaffrey | 0.0% | 1.005 | 12.8% | 38.6% | 78.1% | 0 |
| Logistic / hierarchy disagree | 2024_11_SEA_SF | Christian McCaffrey | 0.0% | 0.931 | 15.3% | 48.2% | 84.9% | 0 |

The complete 2024 prediction artifact is `reports/phase5_validation_predictions.csv`; machine-readable metrics and all selected examples are in `reports/phase5_metrics.json`.
