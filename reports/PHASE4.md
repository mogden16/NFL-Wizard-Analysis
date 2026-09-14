# Phase 4 completion report

The target is a team's nflverse rushing plus receiving touchdowns credited to the possessing team. Defensive returns, special-teams returns, blocked-kick scores, and two-point tries are excluded. A touchdown on a lateral play remains an offensive touchdown when nflverse credits its `rush_touchdown` or `pass_touchdown` to the possessing team. Every team-game target agreed with the separate weekly team rushing-plus-passing TD totals.

Football-only history: 4,222 team-games across 2017–2024 (2017-09-08T00:30:00+00:00 to 2025-01-06T01:20:00+00:00). Strict T−60 market history: 2,174 team-games, 1,087 games across 2021–2024 (2021-09-10T00:20:00+00:00 to 2025-01-06T01:20:00+00:00); 223 cached daily snapshots. All 2024 regular-season games are represented. No 2025-season data was loaded.

Sportsbook values come from one book's spread and total market updates in a historical snapshot at or before kickoff minus 60 minutes. The home-spread convention is the handicap added to home points, so a home favorite has a negative spread, `home_margin = -home_spread`, and home implied points are `(total + home_margin)/2`. nflverse closing lines never enter the strict table.
Quote age relative to each T−60 cutoff: median 10 minutes, 90th percentile 419, maximum 660.

Training: 2021–2023 strict market team-games (1,630); 2024 chronological validation (544); 2025 untouched. The football-only diagnostic trains on 2017–2023 and validates on 2024. No random split or betting-return selection was used.
The points-per-TD reference divides implied team points by the empirical 2021–2023 league ratio of 9.512 points per offensive touchdown; it does not divide by seven.

## 2024 count-model comparison

| Model | MAE | RMSE | Poisson deviance | Bias | Calibration gap |
|---|---:|---:|---:|---:|---:|
| points-per-TD ratio | 1.0117 | 1.2981 | 0.8127 | -0.0970 | 0.1743 |
| market-only | 1.0070 | 1.2914 | 0.8069 | -0.1063 | 0.1342 |
| football+market | 1.0044 | 1.2860 | 0.8022 | -0.0910 | 0.1260 |
| football-only diagnostic | 1.0486 | 1.3407 | 0.8621 | -0.0083 | 0.0795 |

Market-only benchmark: **A_market_only_poisson**. Best controlled football challenger: **D_plus_rest_poisson**. Final selected model: **A_market_only_poisson**.
Football minus market: MAE -0.26%, RMSE -0.42%, count deviance -0.58%, calibration gap -6.09% (negative means improvement). Material improvement requires at least 1% lower RMSE and deviance with no worse MAE.
**Did football improve on the sportsbook in 2024? No material improvement; the market-only model remains selected.**

Leading market-model features: implied_team_points, team_spread, game_total, home.
Leading football-challenger features: implied_team_points, team_spread, game_total, last5_off_rush_epa, home, last5_off_pass_epa, missingindicator_last5_off_red_zone_td_conversion, last5_off_red_zone_trips.
The C+xTD diagnostic did not improve both RMSE and deviance over C without xTD on 2024; xTD was not added to the selected market model.

## Controlled feature-family ablation

A uses market only; B adds prior offense/opponent efficiency; C adds pace and red-zone usage; D adds rest. A separate C+xTD test adds prior team and opponent xTD. Weather and starter fields are not added without timestamped pregame evidence. Every experiment below was fixed in advance and uses the same 2024 validation team-games.

| Experiment | MAE | RMSE | Deviance | Calibration gap |
|---|---:|---:|---:|---:|
| A_market_only_poisson | 1.0070 | 1.2914 | 0.8069 | 0.1342 |
| A_market_only_lightgbm | 1.0107 | 1.3065 | 0.8213 | 0.1853 |
| B_market_efficiency_poisson | 1.0057 | 1.2864 | 0.8034 | 0.1409 |
| B_market_efficiency_lightgbm | 1.0149 | 1.3082 | 0.8264 | 0.1186 |
| C_market_efficiency_pace_redzone_poisson | 1.0047 | 1.2868 | 0.8031 | 0.1286 |
| C_market_efficiency_pace_redzone_lightgbm | 1.0164 | 1.3076 | 0.8261 | 0.0904 |
| D_plus_rest_poisson | 1.0044 | 1.2860 | 0.8022 | 0.1260 |
| D_plus_rest_lightgbm | 1.0155 | 1.3095 | 0.8272 | 0.1108 |
| C_plus_xtd_poisson | 1.0041 | 1.2876 | 0.8035 | 0.1224 |
| C_plus_xtd_lightgbm | 1.0178 | 1.3100 | 0.8286 | 0.0933 |

## Team TD count distribution

Training offensive TD mean 2.335, variance 1.912, variance/mean 0.819; conditional Poisson Pearson dispersion 0.713. Material overdispersion required both ratios to exceed 1.2 and was not present; Negative Binomial was not justified. The published P(0), P(1), P(2), P(3), P(4+) use a Poisson reference distribution and sum to one; its shape is a working assumption, not a claim of perfect count calibration.

Observed 2024 offensive TD counts: 0: 40, 1: 95, 2: 177, 3: 115, 4: 73, 5: 31, 6: 12, 7: 1.

| Count | Mean predicted probability | Observed frequency |
|---|---:|---:|
| 0 | 0.108 | 0.074 |
| 1 | 0.231 | 0.175 |
| 2 | 0.256 | 0.325 |
| 3 | 0.196 | 0.211 |
| 4+ | 0.210 | 0.215 |

The Poisson reference overpredicts 0- and 1-TD team-games and underpredicts 2-TD team-games on 2024 validation, consistent with underdispersion. Negative Binomial would add overdispersion and was not fitted.

## Coverage and missingness

The strict market table has complete spread, total, implied-points and home/away fields for every included team-game. The football-only table retains null lagged fields until the required earlier games exist; missing forecasts and starter signals remain null.

| Season | Football team-games | Strict market team-games | Prior offense EPA present | Prior opponent EPA present | Prior xTD present |
|---|---:|---:|---:|---:|---:|
| 2017 | 512 | 0 | 93.8% | 93.8% | 0.0% |
| 2018 | 512 | 0 | 100.0% | 100.0% | 68.8% |
| 2019 | 512 | 0 | 100.0% | 100.0% | 100.0% |
| 2020 | 512 | 0 | 100.0% | 100.0% | 100.0% |
| 2021 | 544 | 544 | 100.0% | 100.0% | 100.0% |
| 2022 | 542 | 542 | 100.0% | 100.0% | 100.0% |
| 2023 | 544 | 544 | 100.0% | 100.0% | 100.0% |
| 2024 | 544 | 544 | 100.0% | 100.0% | 100.0% |

## Calibration by predicted team TD

### Market Only

| Predicted range | Team-games | Mean prediction | Mean actual |
|---|---:|---:|---:|
| <1.5 | 3 | 1.378 | 1.667 |
| 1.5–2.0 | 148 | 1.808 | 1.757 |
| 2.0–2.5 | 215 | 2.268 | 2.442 |
| 2.5–3.0 | 139 | 2.710 | 2.827 |
| 3.0–3.5 | 34 | 3.180 | 3.441 |
| 3.5–4.0 | 5 | 3.613 | 4.000 |
| 4.0+ | 0 | — | — |

### Football Market

| Predicted range | Team-games | Mean prediction | Mean actual |
|---|---:|---:|---:|
| <1.5 | 4 | 1.387 | 1.500 |
| 1.5–2.0 | 145 | 1.783 | 1.724 |
| 2.0–2.5 | 206 | 2.249 | 2.374 |
| 2.5–3.0 | 138 | 2.718 | 2.862 |
| 3.0–3.5 | 40 | 3.177 | 3.500 |
| 3.5–4.0 | 11 | 3.730 | 3.636 |
| 4.0+ | 0 | — | — |

### Selected

| Predicted range | Team-games | Mean prediction | Mean actual |
|---|---:|---:|---:|
| <1.5 | 3 | 1.378 | 1.667 |
| 1.5–2.0 | 148 | 1.808 | 1.757 |
| 2.0–2.5 | 215 | 2.268 | 2.442 |
| 2.5–3.0 | 139 | 2.710 | 2.827 |
| 3.0–3.5 | 34 | 3.180 | 3.441 |
| 3.5–4.0 | 5 | 3.613 | 4.000 |
| 4.0+ | 0 | — | — |

## Representative 2024 games

| Case | Team | Market implied pts | Market-only TD | Football+market TD | Actual TD |
|---|---|---:|---:|---:|---:|
| heavy favorite | BAL | 30.75 | 3.63 | 3.92 | 4 |
| underdog | CLE | 10.75 | 1.22 | 1.17 | 1 |
| high total | DET | 29.75 | 3.53 | 3.62 | 4 |
| low total | CLE | 15.25 | 1.60 | 1.55 | 0 |

## Assumptions and limitations

- Prior-game football and xTD publication uses the explicit assumed kickoff + 24 hours proxy from the earlier phases, never an observed release timestamp. Every source game's proxy precedes the target prediction time.
- This is a team offensive TD count model only; it does not calculate player anytime-TD probabilities.
- The market series starts in 2021. Earlier football seasons remain in the separate football-only table and never receive fabricated closing or historical market lines.
- A pregame starting-QB signal and timestamped weather forecasts were unavailable for the full series, so their model inputs remain null. PROE was not safely reconstructed; lagged pass rate is the simpler tendency measure.
- The T−60 snapshot is taken at the earliest kickoff cutoff on each Eastern game day. Later games that day can use an older eligible quote. Both market update times and the snapshot wrapper time are checked against each game's own T−60 cutoff.
- Only the controlled ablations in this report were evaluated; no broad feature search or betting ROI optimization occurred.
- Full feature missingness by season, model importance, count probabilities, and every experiment are in `reports/phase4_metrics.json`.

Phase 5 has not begun.
