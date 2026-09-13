# Phase 3 completion report

Play-level scoring opportunities span the 2017–2024 regular seasons. 2017 is training-only: it has no earlier season from which to fit a model. Scored and lagged xTD artifacts therefore span 2018–2024. The protected 2025 holdout is never loaded or scored.

Modeled opportunities: 109,480 rushing and 138,559 receiving. Scored opportunities: 217,523; player-game aggregates: 35,722; lagged player-game rows: 54,916.

2017–2022 fit and 2023 calibration checks precede chronological 2024 validation. Historical feature production uses the preregistered logistic baseline, independent of 2024 model-selection outcomes. Each target season is scored by a model fit on earlier seasons only. LightGBM must lower both 2024 log loss and Brier by at least 1% without materially worse calibration to be selected for future use. Accuracy is not a selection criterion.

## Rushing

logistic: log loss 0.089998; Brier 0.023931; calibration gap 0.003546 (14,278 2024 plays, 511 TDs).
lightgbm: log loss 0.089852; Brier 0.023827; calibration gap 0.003122 (14,278 2024 plays, 511 TDs).
Selected: **logistic** — LightGBM did not meet the predeclared material-improvement rule.
Sigmoid calibration applied: **True** (selected only if prior-season late-week log loss and Brier both improve).

| Bin | Plays | Mean xTD | TD rate |
|---|---:|---:|---:|
| 0-1% | 10573 | 0.0033 | 0.0046 |
| 1-2% | 1037 | 0.0135 | 0.0135 |
| 2-5% | 972 | 0.0329 | 0.0319 |
| 5-10% | 562 | 0.0719 | 0.0534 |
| 10-20% | 395 | 0.1377 | 0.1722 |
| 20-35% | 280 | 0.2659 | 0.2714 |
| 35-50% | 155 | 0.4113 | 0.3935 |
| 50%+ | 304 | 0.5763 | 0.5987 |

Leading fitted features: log_yardline, run_location=Unknown, yardline_100, run_gap=end, inside_20, qtr, position=WR, position=QB.
Sanity: {"one_vs_twenty_ratio": 27.063565386601493, "goal_to_go_one_vs_non_goal_one": 1.0260278418397826, "one_substantially_greater_than_twenty": true, "goal_to_go_greater": true}.

## Receiving

logistic: log loss 0.115327; Brier 0.032960; calibration gap 0.005140 (16,993 2024 plays, 806 TDs).
lightgbm: log loss 0.113635; Brier 0.032829; calibration gap 0.003603 (16,993 2024 plays, 806 TDs).
Selected: **logistic** — LightGBM did not meet the predeclared material-improvement rule.
Sigmoid calibration applied: **False** (selected only if prior-season late-week log loss and Brier both improve).

| Bin | Plays | Mean xTD | TD rate |
|---|---:|---:|---:|
| 0-1% | 10779 | 0.0028 | 0.0038 |
| 1-2% | 1661 | 0.0142 | 0.0120 |
| 2-5% | 1523 | 0.0313 | 0.0223 |
| 5-10% | 884 | 0.0722 | 0.0588 |
| 10-20% | 747 | 0.1437 | 0.1593 |
| 20-35% | 742 | 0.2702 | 0.3046 |
| 35-50% | 451 | 0.4156 | 0.4346 |
| 50%+ | 206 | 0.5835 | 0.5728 |

Leading fitted features: relative_to_endzone, air_yards, log_yardline, yardline_100, game_seconds_remaining, inside_20, qtr, ydstogo.
Sanity: {"end_zone_one_vs_short_fifty_ratio": 159.62763010290195, "end_zone_target_greater": true}.

## Actual touchdowns versus xTD

These are regular-season player totals over modeled rushes and targets. They illustrate finishing above or below opportunity-based expectation; they are not estimates of future ability.

| Player | Season | Actual TD | xTD | Difference |
|---|---:|---:|---:|---:|
| Derrick Henry | 2019 | 18 | 9.49 | +8.51 |
| Deebo Samuel Sr. | 2021 | 14 | 5.59 | +8.41 |
| Alvin Kamara | 2020 | 21 | 12.93 | +8.07 |
| Austin Ekeler | 2021 | 20 | 12.09 | +7.91 |
| James Cook | 2024 | 18 | 10.32 | +7.68 |
| Leonard Fournette | 2019 | 3 | 10.03 | -7.03 |
| Tony Pollard | 2023 | 6 | 12.86 | -6.86 |
| Diontae Johnson | 2022 | 0 | 6.86 | -6.86 |
| Trey McBride | 2024 | 3 | 8.84 | -5.84 |
| Benny Snell | 2020 | 4 | 9.67 | -5.67 |

## Lagged feature availability

A window's `*_xtd` is mean xTD per game; `*_xtd_sum` is the total across eligible games. The season window resets each season. Every value uses earlier games only.

| Season | Candidate rows | Rows with prior xTD | Last-3 total xTD present | Season-to-date total xTD present |
|---|---:|---:|---:|---:|
| 2018 | 7,230 | 6,624 | 91.62% | 91.62% |
| 2019 | 7,121 | 6,602 | 92.71% | 90.42% |
| 2020 | 7,642 | 7,316 | 95.73% | 93.23% |
| 2021 | 8,487 | 8,068 | 95.06% | 91.95% |
| 2022 | 8,322 | 7,943 | 95.45% | 91.95% |
| 2023 | 8,006 | 7,537 | 94.14% | 91.04% |
| 2024 | 8,108 | 7,630 | 94.10% | 90.17% |

## Assumptions and limitations

- Only primary rushing and targeted receiving opportunities are modeled. Two-point plays, kneels, spikes, deleted plays and lateral/ambiguous scoring plays are excluded; the source coverage file counts excluded ambiguous TD plays.
- Rusher/receiver position uses a prior same-team weekly player-stat position. Prior-game publication is an explicit **assumed kickoff + 24 hours** proxy, never an observed release timestamp. Unknown position is retained rather than filled from the target game.
- No historical team implied total, market field, route participation, or player identity is used in the xTD model. Complete timestamped historical market coverage is unavailable.
- Run gap is retained with an explicit Unknown category; it is unavailable on roughly 27% of rushing plays. Prior-game position is Unknown on roughly 9% of opportunities. Per-season and per-field model-input coverage is in `reports/phase3_metrics.json`.
- Model inputs are context known at the play, suitable for opportunity-quality measurement. These current-play inputs are never used as pregame features for that same game; only earlier-game aggregates pass the strict lag filter.
- Sigmoid calibration is fitted on weeks 1–9 of the prior season only and kept only when it improves both log loss and Brier on weeks 10+. The 2024 validation season does not fit calibration or model parameters.
- The architecture comparison uses 2024 outcome labels, but historical xTD features are always produced by the preregistered logistic baseline. A later 2024 downstream model-selection exercise should still account for this reuse of the validation year. 2025 remains untouched for a final holdout.
- Historical lagged xTD is available only after an earlier game with a modeled opportunity. Rates and shares are null when there is no observed denominator. Missing features remain null.
- 2017 xTD is unavailable by design because no earlier training season is present. Phase 4 has not started.

Detailed calibration for both models, example probability tables, per-season feature missingness, and actual TD versus xTD player examples are in `reports/phase3_metrics.json`.
