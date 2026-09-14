# Phase 7 historical ATD market-edge backtest

**Decision: `PHASE7_RESULT = NO_ROBUST_EDGE`.** No rule from the fixed 60-rule
grid passed the 2023 development gates, so the 2024 primary strategy is **NO
BET**. No 2025 holdout result or September 13, 2026 exhibition result was
opened. The Phase 5 hierarchical football probabilities were held fixed.

## Data and chronology

The sample is all 256 regular-season games in each of 2023 and 2024 with
Phase 5 player-universe and Phase 4 event IDs (weeks 1–18). The Odds API
historical event endpoint supplied ATD Yes prices at the nearest snapshot at
or before kickoff minus 60 minutes. The collector retained 52,435 book-level
2023 quotes and 45,781 in 2024, across all 512 games. The Stage A file
contains every quoted player: 14,746 in 2023 and 9,793 in 2024. Of these,
5,564 and 5,516 respectively match a frozen football prediction. The Stage A
CSV and every underlying quote are frozen with SHA-256 hashes; it has no
outcome, current-game usage, closing price, or settlement column. Stage B is
a separate command, checks those hashes, and reads season-specific outcomes.
The 2023-only rule was frozen and hashed before 2024 settlement or closing
prices were opened.

The Stage A prediction SHA-256 is
`b6eba31fb02936643b2e1650d0983f7240cdbafa29d54e7bebcf2ea08fe5f8f8`.
The 2023 rule SHA-256 is
`fc0fe94c9d88167b939b01d2f98186db07b41f8419dfed3c9bdc1080ba5d6d0b`.
The frozen rule file has an empty shortlist and a `NO_BET` status.

The analysis could not establish authoritative historical T-60 active status
or sportsbook-specific historical DNP settlement terms. Therefore no player
is marked strictly eligible at prediction time. Historical prior appearance
is only a role proxy. Stage B grades players with affirmative PFR offensive
snap, player-stat, or play-level participation evidence: 5,240 in 2023 and
5,142 in 2024. It leaves 324 and 374 matched players respectively as
`policy_unknown` with null outcomes and P/L; their frozen predictions remain
intact. This complete-case grading can create participation selection bias.
**All portfolio returns below are research proxies, not executable strategy
returns.** No claim of a deployable rule can follow from this sample.

## Development and validation

The predeclared grid combined five minimum probability edges (0, 2.5, 5,
7.5, 10 percentage points), four minimum football-estimated EVs (0, 5, 10,
15%), and three book-count minimums (1, 2, 3). A 2023 candidate needed at
least 100 graded bets, positive best-price ROI, nonnegative median-price ROI,
nonnegative same-book mean CLV with at least 50% CLV coverage, and no single
winner exceeding 50% of gross winnings. None passed. The top raw 2023
best-price return in the grid was +25.0% over only 40 bets; its median-price
return was −4.2%. It failed the predeclared gates. The full grid and each
failure remain in `reports/phase7/development/development_report.json`.

As a descriptive fixed comparison, players with both positive best-price
football edge and positive estimated EV had:

| Season | Graded proxy bets | Best-price ROI | Median-price ROI | Mean best CLV | Beat close |
|---|---:|---:|---:|---:|---:|
| 2023 development | 1,605 | −21.3% | −42.5% | +0.30 pp | 32.1% |
| 2024 validation | 1,756 | +1.2% | −20.3% | +0.23 pp | 30.0% |

This is **not** a promoted betting rule: it failed badly in 2023, loses at
median prices in both seasons, and lacks active-status certainty. Same-book
CLV means entry implied probability subtracted from closing implied
probability, so positive is favorable. Many prices are unchanged at close;
positive mean CLV alongside only 30% beating close should not be overstated.
Closing quotes were retrieved separately and never entered Stage A.

The fixed best-price disagreement buckets are not monotonic:

| Football minus raw market | 2023 observations | 2023 ROI | 2024 observations | 2024 ROI |
|---|---:|---:|---:|---:|
| ≤−10 pp | 524 | −1.1% | 684 | −6.2% |
| −10 to −5 pp | 948 | −5.8% | 944 | −5.0% |
| −5 to −2.5 pp | 882 | −8.9% | 737 | −18.0% |
| −2.5 to 0 pp | 1,281 | −9.9% | 1,021 | −23.4% |
| 0 to +2.5 pp | 934 | −21.7% | 908 | −15.9% |
| +2.5 to +5 pp | 391 | −26.9% | 525 | +29.1% |
| +5 to +10 pp | 239 | −17.8% | 281 | −5.7% |
| >+10 pp | 41 | +22.0% | 42 | +67.4% |

The detailed best **and median** bucket tables also report mean football and
market probability, actual TD rate, market calibration residual, average
odds, and CLV in `reports/phase7/supplemental_diagnostics.json`. The 2024
>+10 pp pocket had average +978 odds and eight winners across 42 bets. Its
best-price game-bootstrap 95% ROI interval was **−47% to +198%**, and its
top two winners produced 51% of gross winnings. The corresponding 2023
pocket lost 6.6% at median prices. This is high-variance longshot evidence,
not an ex post strategy candidate.

Among 2024 positive-edge/EV rows, median pricing was negative in every
predeclared best-price dispersion bin (≤5%, 5–15%, 15–30%, >30%). The
>30% best-versus-median bin had 719 proxy bets, −6.5% best-price ROI and
−37.6% median ROI. Fresh 2024 quotes (≤5 minutes) produced +1.3% best-price
ROI and −20.4% median ROI; 5–15-minute quotes produced +1.0% and −19.4%.
Only one positive-edge 2024 quote was 15–30 minutes old and none exceeded
30 minutes. Thus stale age is not the main observed concentration; price
dispersion and line-shopping dependence are. The raw market's one-sided Yes
prices are **not** de-vigged; `P_market_raw` and the separately fitted 2023
calibrated market probability remain distinct.

The 2024 unfiltered graded proxy sample lost at best prices in every position:
RB −5.5% (1,286), WR −9.6% (2,150), TE −7.2% (1,208), QB −15.2% (498).
No position-specific rule was predeclared from 2023. Odds-range tables in
the machine-readable report show that +500-and-longer bets form 2,661 of
5,142 2024 graded rows and lose 14.0% at best prices. The small +100–199
range was +1.9%; this is descriptive only. Book-level and position-level
mean/median CLV and percentage beating close are in the supplemental report.

## Conditional information beyond the market

A 2023-only logistic diagnostic fit `TD ~ market_logit + 10 ×
(P_football − P_market_raw_best)`. The coefficient on each 10-pp football
edge was **0.0041** (standard error 0.0724, two-sided p≈0.955). Adding
the edge changed 2023 log loss from 0.4062650 to 0.4062648, an immaterial
in-sample difference. On independently evaluated 2024 graded rows:

| Probability | Log loss | Brier |
|---|---:|---:|
| Raw T-60 market | 0.42521 | 0.13529 |
| 2023-calibrated market | 0.42540 | 0.13548 |
| 2023 market plus football edge | 0.42538 | 0.13548 |
| Frozen football hierarchy | 0.43438 | 0.13850 |

The residual diagnostic provides no evidence that the football probability
adds useful conditional information beyond the raw market. This matched
sample differs from the fixed Phase 6 benchmark sample, so the metrics are
not a replacement for Phase 6 model selection.

## Decision and limitations

The answers are: football-minus-market disagreement was not reliably
predictive; its relationship with outcomes and ROI was not monotonic; no
2023-developed rule qualified for independent 2024 validation; apparent
positive 2024 best-price pockets failed median-price robustness; average
CLV was slightly positive but did not rescue the strategy; extreme price
dispersion and longshot winners matter more than quote age; no position
provides predeclared evidence for a distinct betting rule; and the 2023
market-residual coefficient was effectively zero. **There is insufficient
evidence to open the 2025 holdout with a frozen betting rule.** The Phase 7
no-bet decision is frozen. It is also reinforced by the absence of verified
pregame active eligibility and historical DNP policies.

Model training, player-probability calibration, and football feature logic
were not changed. Market calibration and residual regression were research
diagnostics fitted on 2023 only. No 2024 outcome changed the 2023 rule.
No betting stake optimization or Phase 8 work was performed.
