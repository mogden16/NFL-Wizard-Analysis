You are building a production-quality research system for NFL anytime-touchdown scorer probabilities.

The objective is NOT to maximize historical betting profit or manufacture a profitable backtest.

The objective is to determine, as rigorously as possible, whether publicly available NFL data can produce well-calibrated player anytime-touchdown probabilities that contain information beyond sportsbook prices.

The project must prioritize:
1. point-in-time data correctness
2. prevention of future-data leakage
3. probability calibration
4. reproducibility
5. interpretability
6. honest out-of-sample evaluation
7. only then, betting-market performance

Call the project:

nfl_td_model

==================================================
1. RESEARCH REFERENCES
==================================================

Before implementing modeling logic, inspect the following public GitHub repositories for architectural and methodological ideas.

Do not blindly copy code. Check licenses before reusing code directly. Prefer reimplementation of concepts.

Repositories:

1. ffverse/ffopportunity
   Purpose:
   - play-level expected touchdown methodology
   - rushing TD probability
   - receiving TD probability
   - expected opportunity modeling
   - contextual play features

   Concepts worth studying:
   - xTD at the individual play level
   - goal line and red-zone context
   - yardline_100
   - goal_to_go
   - air_yards
   - relative_to_endzone
   - down
   - distance
   - QB scramble
   - run direction
   - implied team total
   - weather/context

   Important:
   - independently validate all implied-team-total calculations
   - do not trust inherited formulas without tests

2. SRock44/sports-prediction-model
   Purpose:
   - binary anytime-TD probability modeling
   - LightGBM
   - chronological validation
   - log loss
   - Brier score
   - recency weighting

   Concepts worth studying:
   - anytime TD is a probability problem, not a regression problem
   - optimize probabilistic loss
   - separate-season holdout
   - recency weighting as a candidate, not an assumption

3. aaronlaporte/sports-analytics
   Purpose:
   - anytime TD models
   - global + position-specific models
   - calibrated logistic regression
   - Poisson modeling for 2+ TD
   - live odds comparison

   Concepts worth studying:
   - global player model
   - RB / WR / TE specific models
   - model stacking

   IMPORTANT:
   Do NOT reproduce same-game leakage observed in that project's historical feature generation.

   All player usage-share features must be calculated only from games occurring BEFORE the prediction timestamp.

4. gesmith0606/nfl_data_engineering
   Purpose:
   - robust NFL data pipeline
   - Bronze / Silver / Gold architecture
   - lagged player and team features
   - opponent-adjusted metrics
   - snap share
   - target share
   - carry share
   - red-zone opportunity
   - vacated opportunity
   - automated testing

   Concepts worth studying:
   - immutable raw data
   - derived analytical layers
   - opponent-adjusted EPA
   - lagged strength-of-schedule features
   - early-season priors
   - vacated opportunity

5. gmalbert/nfl-predictions
   Purpose:
   - player props
   - XGBoost / LightGBM
   - EWMA
   - market baselines
   - CLV
   - immutable prediction journal
   - documented lessons from leakage

   Concepts worth adopting:
   - immutable predictions
   - separate prediction, quote, decision, and settlement records
   - market benchmarks
   - flat shadow staking
   - calibration gates
   - closing-line-value tracking

==================================================
2. PROJECT PHILOSOPHY
==================================================

The system must be capable of concluding:

"There is no evidence of a profitable edge."

That is an acceptable and important outcome.

Never tune the system simply to maximize historical ROI.

Every feature must represent information that would actually have been available at prediction time.

Core invariant:

feature_available_at <= prediction_time

If this cannot be demonstrated for a feature, exclude the feature from strict backtesting.

==================================================
3. PRIMARY MODELING ARCHITECTURE
==================================================

Use a hierarchical touchdown model.

The architecture should eventually contain four major components.

A. GAME / TEAM SCORING ENVIRONMENT

Estimate:

E[team offensive touchdowns]

for each team.

Primary market inputs:

- spread
- game total
- implied team total

Football inputs may include:

- offensive EPA/play
- defensive EPA/play
- pass EPA
- rush EPA
- success rate
- explosive-play rate
- early-down EPA
- PROE or pass rate over expectation if available
- pace
- plays per game
- drives per game
- red-zone efficiency
- home field
- rest
- opponent strength
- weather
- starting QB information when point-in-time reliable

Market-implied team score must be calculated with one canonical and fully tested sign convention.

Do not approximate team TDs as team_points / 7.

Fit an empirical model:

E[team_offensive_TD] = f(
    implied_team_points,
    spread,
    total,
    offensive_strength,
    defensive_strength,
    pace,
    environment
)

Build a market-only baseline first.

B. PLAYER OPPORTUNITY MODEL

Predict expected player opportunity volume.

Candidate outputs:

E[carries]
E[targets]
E[red_zone_carries]
E[carries_inside_10]
E[carries_inside_5]
E[red_zone_targets]
E[end_zone_targets]

Core historical features:

- snap share
- route participation
- target share
- carry share
- touch share
- goal-line opportunity share
- red-zone target share
- red-zone rush share
- end-zone target share

Prefer player SHARE metrics over raw volume when both are available.

Every rolling feature MUST exclude the game being predicted.

C. EXPECTED TOUCHDOWN OPPORTUNITY QUALITY

Build a play-level expected-touchdown model inspired by ffopportunity.

For each rushing or receiving opportunity estimate:

P(TD | play context)

Then aggregate:

player_xTD = sum(P(TD_j))

Useful rushing variables may include:

- yardline_100
- goal_to_go
- down
- ydstogo
- score differential
- quarter
- time remaining
- shotgun
- no huddle
- QB scramble
- run direction
- run gap
- position
- team implied total

Useful receiving variables may include:

- yardline_100
- air_yards
- relative_to_endzone
- end-zone target indicator
- down
- ydstogo
- score differential
- receiver position
- QB context
- team implied total

Create:

rush_xTD
receiving_xTD
total_xTD
xTD_per_opportunity
xTD_share_of_team

Create rolling and exponentially weighted versions using strictly historical games.

D. PLAYER ANYTIME-TD PROBABILITY

Build at least two conceptually distinct models.

Model 1:
Hierarchical model

Estimate:

lambda_player

using:

E[team TD]
player opportunity share
player xTD profile
expected usage

Baseline probability:

P(player >= 1 TD) = 1 - exp(-lambda_player)

Do not assume Poisson is correct. Treat it as a baseline.

Model 2:
Direct binary classifier

Target:

anytime_td = 1 if player scores at least one rushing or receiving TD
otherwise 0

Exclude passing touchdowns from QB anytime-TD scoring.

First model:

calibrated logistic regression

Challenger:

LightGBM

Optional later challenger:

XGBoost

Do not introduce more complex models unless they demonstrate real chronological out-of-sample improvement.

==================================================
4. POSITION-SPECIFIC MODELING
==================================================

Build:

global model
RB model
WR model
TE model
QB-rushing model if sample size supports it

RB emphasis:

- carries
- carry share
- carries inside 10
- carries inside 5
- goal-line share
- receiving role
- team TD expectation

WR emphasis:

- route participation
- target share
- red-zone target share
- end-zone targets
- air yards
- receiving xTD
- expected passing TD environment

TE emphasis:

- routes
- target share
- red-zone targets
- end-zone targets
- receiving xTD

QB anytime-TD emphasis:

- designed rush share
- scramble rate
- red-zone carries
- carries inside 10
- carries inside 5
- rushing xTD

Do not count QB passing TDs toward anytime-TD scorer labels.

==================================================
5. MODEL ENSEMBLING
==================================================

Eventually compare:

P_hierarchical
P_global
P_position
P_lightgbm

Do NOT stack using predictions generated on the same samples used to fit the component model.

Generate meta-model training inputs using chronological out-of-fold predictions.

Example:

Train through season N
predict season N+1

Repeat to create true out-of-fold predictions.

Only build an ensemble if it improves:

- log loss
- Brier score
- calibration
- stability

on chronological validation data.

==================================================
6. DATA SOURCES
==================================================

Primary NFL source:

nflverse / nflreadpy

Use public play-by-play data.

Historical football data can extend well before sportsbook prop history.

Recommended minimum football-history period:

2017-present

Use earlier history if it materially improves model stability and schema consistency.

Odds:

The Odds API

Historical player prop snapshots begin approximately May 2023.

Market:

player_anytime_td

Use sportsbook-specific quotes.

Store:

sportsbook
player
market
side
American odds
decimal odds
quote timestamp
event ID

Historical backtests must retrieve the closest quote available AT OR BEFORE the defined prediction timestamp.

Never use a later quote.

Weather:

Historical forecasts must represent forecasts available before the prediction timestamp.

Do not use observed future weather in historical predictions.

Use NOAA HRRR where practical for historical U.S. games.

For live / recent operation, an easier forecast API may be used if properly timestamped.

==================================================
7. PREDICTION TIME
==================================================

Default strict player-prop prediction time:

kickoff - 60 minutes

Reason:

NFL inactive lists are generally established around 90 minutes before kickoff, allowing a short period for the market to incorporate information.

This must be configurable.

Store:

prediction_time
kickoff_time
minutes_to_kickoff

All input information must satisfy:

information_timestamp <= prediction_time

==================================================
8. MODEL TIME SPLITS
==================================================

Football probability models do not require sportsbook odds and may use longer history.

Recommended initial framework:

2017-2023:
training / model development data

2024:
validation, calibration, model selection

2025:
untouched final historical holdout

2026:
prospective paper-trading experiment

The exact earlier training start may change based on data quality.

STRICT RULE:

2025 cannot be used for:

- hyperparameter selection
- feature selection
- probability threshold selection
- ensemble weighting
- calibration fitting

Once 2025 is evaluated, do not change the model and continue describing 2025 as untouched.

For historical betting ROI:

The strict odds-backed backtest begins only when trustworthy timestamped historical ATD prices exist, approximately 2023 onward.

Older football data may train football relationships but cannot be used to fabricate historical sportsbook returns.

==================================================
9. TEMPORAL VALIDATION
==================================================

Never use random train/test splitting as the primary validation methodology.

Use season-forward or expanding-window testing.

Example:

train <= 2021, test 2022
train <= 2022, test 2023
train <= 2023, test 2024
train <= 2024, final test 2025

All rolling features must be shifted.

The current game can never contribute to its own predictors.

==================================================
10. RECENCY
==================================================

Test rather than assume:

rolling last 3
rolling last 4
rolling last 5
rolling last 8
season-to-date
EWMA

Recency weighting is allowed.

Possible example-training half-lives:

30 days
45 days
60 days
90 days
no weighting

Select based on chronological validation log loss / Brier score.

Do not select based purely on historical ROI.

==================================================
11. EARLY-SEASON PRIORS
==================================================

Weeks 1-3 may have insufficient current-season data.

Implement conservative prior blending.

Possible prior inputs:

- prior-season role
- depth-chart status
- returning usage
- vacated targets
- vacated carries
- roster role

Shrink prior influence as current-season evidence accumulates.

Conceptually:

feature =
w_current * current_season
+
w_prior * prior

with:

w_prior decreasing rapidly as current-season opportunities accumulate.

By approximately Week 5-6, current-season usage should generally dominate.

Do not hard-code exact weights without validation.

==================================================
12. FEATURES THAT MUST BE TESTED CAREFULLY
==================================================

Candidate useful features:

Team:
- implied team total
- spread
- total
- expected team offensive TDs
- offense EPA
- defense EPA
- pace
- red-zone efficiency
- opponent-adjusted strength

Player:
- snap share
- route share
- target share
- rush share
- touch share
- red-zone opportunity share
- goal-line opportunity share
- end-zone target share
- rush xTD
- receiving xTD
- total xTD
- xTD share
- xTD per opportunity

Secondary candidate:
- actual historical TD rate

Actual TD rate should not dominate opportunity or xTD features.

==================================================
13. FEATURES TO EXCLUDE FROM V1
==================================================

Do not use as core V1 features:

- head-to-head player/team records
- winning streaks
- subjective "hot player" indicators
- social-media sentiment
- manually created weather multipliers
- manually created injury multipliers
- fantasy-site matchup rankings such as "31st vs TE"
- current-game target share
- current-game rush share
- current-game snaps
- future injury knowledge
- closing sportsbook information when predicting earlier

Any current-game feature is leakage.

==================================================
14. OPPONENT FEATURES
==================================================

Prefer continuous lagged measures over ordinal rankings.

Use candidate opponent features such as:

- defensive EPA/play
- defensive rush EPA
- defensive pass EPA
- success rate allowed
- explosive-play rate allowed
- red-zone TD rate allowed
- opponent-adjusted versions where practical

Do not simply use:

"Defense ranked 28th against RB touchdowns."

Touchdowns allowed by position may be tested as a weak secondary feature but should not be assumed predictive.

==================================================
15. WEATHER
==================================================

Weather belongs primarily in:

team scoring environment
passing/rushing mix
opportunity environment

Candidate features:

wind
gust
temperature
precipitation
humidity/dew point
roof

Do not manually implement rules such as:

wind > 15 mph => multiply passing TD probability by 0.85

Let historical data learn the relationship.

==================================================
16. INJURIES
==================================================

V1 historical modeling must not depend on unreliable injury history.

Official inactive status may eventually be used for live predictions.

For strict historical testing, include injuries only when:

1. the source is trustworthy
2. historical timestamp is available
3. designation was actually known before prediction_time

Otherwise omit the injury feature.

Do not allow today's roster state to leak into old games.

==================================================
17. MARKET BENCHMARK
==================================================

Keep the player's own sportsbook price separate from the football model initially.

Build:

P_football

and independently:

P_market

Do NOT feed the player's ATD price directly into P_football in the base model.

Spread and total ARE allowed in the football model because they represent the team's overall scoring environment.

For each player collect:

best available price
median price across books
sportsbook
quote timestamp

Convert American to decimal odds.

Raw implied probability:

positive odds:
100 / (odds + 100)

negative odds:
abs(odds) / (abs(odds) + 100)

ATD markets are often one-sided, so do not claim a fully de-vigged fair probability when only the YES side exists.

Build an empirical market calibration baseline from historical outcomes.

Compare:

P_football
P_market
P_combined

Only build P_combined after evaluating football-model independence.

==================================================
18. EXPECTED VALUE
==================================================

For decimal odds D and model probability p:

EV_per_unit = p * D - 1

Store:

model_probability
market_probability
best_price
decimal_price
EV
probability_edge

Do not label something a "value bet" simply because:

model probability > raw implied probability.

Require configurable minimum EV / probability edge thresholds.

Those thresholds must be selected using validation data only.

==================================================
19. NO BET
==================================================

The model must support:

BET
NO BET

Most players may appropriately be NO BET.

The system should not try to generate a prescribed number of wagers.

==================================================
20. STAKING
==================================================

Historical and prospective evaluation default:

flat 1-unit stake

Do not optimize bankroll strategy in V1.

Do not use Kelly sizing until:

- probability calibration is strong
- sufficient sample size exists
- positive CLV exists
- holdout performance passes predefined gates

If Kelly is later added, use fractional Kelly and report it only secondarily.

==================================================
21. CLOSING-LINE VALUE
==================================================

Track CLV where historical closing player prices are available.

For each prediction store:

prediction quote
closing quote
prediction implied probability
closing implied probability
price movement

Develop a clear CLV convention for plus and minus odds.

Evaluate whether recommended positions systematically beat the closing market.

CLV matters even when individual bets lose.

==================================================
22. EVALUATION METRICS
==================================================

Primary probability metrics:

log loss
Brier score
calibration error
reliability diagram
calibration slope
calibration intercept

Secondary discrimination:

ROC-AUC
PR-AUC

Do not optimize classification accuracy.

A model predicting "no TD" for everyone may have high accuracy and zero usefulness.

Evaluate calibration buckets, for example:

0-10%
10-20%
20-30%
30-40%
40-50%
50-60%
60%+

For every bucket show:

predicted average probability
actual TD rate
sample size

==================================================
23. BETTING EVALUATION
==================================================

For historical odds-backed testing report:

number of qualifying bets
wins
losses
pushes if applicable
average odds
hit rate
average model probability
ROI
units won
95% confidence interval
maximum drawdown
CLV
results by sportsbook
results by odds bucket
results by model-edge bucket
results by position
results by team
results by season

Never present a profitable total without showing the yearly breakdown.

==================================================
24. MODEL COMPARISON
==================================================

Every advanced model must beat simple baselines.

Required baselines:

A. market-only probability model
B. historical TD-rate model
C. logistic regression using core opportunity features
D. hierarchical Poisson-style TD model

Challengers:

LightGBM
XGBoost
ensemble

Do not accept increased model complexity unless it materially improves chronological out-of-sample:

log loss
Brier score
calibration

Betting ROI is a downstream test, not the primary model-selection metric.

==================================================
25. FEATURE ABLATION
==================================================

Perform controlled feature-family ablations.

Candidate feature groups:

market/team scoring
basic usage
share metrics
red-zone usage
xTD
opponent strength
weather
early-season priors
recency

For each experiment record:

feature groups
training period
validation period
hyperparameters
metrics

Do not perform uncontrolled feature-subset searching.

Maintain an experiment registry.

Never delete losing experiments.

==================================================
26. EXPERIMENT REGISTRY
==================================================

Create a persistent experiment log containing:

experiment_id
created_at
git_commit
model_type
feature_set
training_period
validation_period
test_period
hyperparameters
calibration_method
probability metrics
betting metrics if applicable
notes

Use MLflow if appropriate, or a simple DuckDB/Parquet experiment registry.

==================================================
27. IMMUTABLE JOURNAL
==================================================

Predictions must be append-only.

Create distinct entities:

Prediction
MarketQuote
BetDecision
Settlement

Never overwrite a historical prediction with a later version.

Prediction must include:

prediction_id
model_version
game_id
player_id
prediction_time
kickoff_time
P_TD
expected_TD
team_expected_TD
input_snapshot_hash

MarketQuote:

quote_id
prediction_id or game/player linkage
sportsbook
quote_time
odds

BetDecision:

decision_id
prediction_id
quote_id
decision
edge
EV
stake_units

Settlement:

result
player_TD_count
profit_units
closing_quote
CLV

==================================================
28. DATA ARCHITECTURE
==================================================

Use:

Python 3.12+
uv
Polars preferred
DuckDB
Parquet
nflreadpy
scikit-learn
LightGBM
XGBoost only if needed
Pydantic Settings
Typer
pytest
Ruff
mypy

Preferred data layers:

data/raw
data/normalized
data/features
data/predictions
data/market
data/results

Raw source data must be immutable.

==================================================
29. SUGGESTED REPOSITORY STRUCTURE
==================================================

nfl_td_model/
    pyproject.toml
    README.md
    .env.example

    src/
        nfl_td_model/
            config.py
            cli.py

            ingest/
                nflverse.py
                odds.py
                weather.py

            normalize/
                games.py
                players.py
                pbp.py
                odds.py

            features/
                team.py
                player_usage.py
                opponent.py
                early_season.py
                weather.py
                xtd.py

            models/
                team_td.py
                opportunity.py
                hierarchical_td.py
                logistic_td.py
                lightgbm_td.py
                calibration.py
                ensemble.py

            backtest/
                temporal_split.py
                simulate.py
                betting.py
                clv.py
                settlement.py

            evaluation/
                probability.py
                calibration.py
                betting.py
                ablation.py

            journal/
                predictions.py
                quotes.py
                decisions.py
                settlement.py

    tests/
        test_market_math.py
        test_time_integrity.py
        test_rolling_features.py
        test_xtd.py
        test_no_current_game_usage.py
        test_odds_timestamp.py
        test_weather_timestamp.py
        test_holdout_integrity.py

==================================================
30. ANTI-LEAKAGE TESTS
==================================================

These tests are mandatory.

1. No future games

Every historical statistic used must come from a game completed before prediction_time.

2. Current-game exclusion

The game being predicted must never contribute to:

target share
rush share
snap share
route share
xTD
TD rate
team metrics

3. Odds timestamp

quote_time <= prediction_time

4. Weather timestamp

forecast_issuance_time <= prediction_time

5. Current roster leakage

Do not use a current roster status as if it existed historically.

6. Training/prediction parity

Feature definitions used in historical training must be mathematically identical to feature definitions used live.

7. Holdout protection

2025 observations may never participate in fitting, calibration, hyperparameter optimization, feature selection, or threshold tuning.

If any invariant fails, terminate the backtest with an explicit error.

==================================================
31. MARKET MATH TESTS
==================================================

Create unit tests for:

American-to-decimal conversion
decimal-to-implied probability
team implied score
spread sign convention
EV
CLV
Poisson >=1 TD probability
Poisson >=2 TD probability

Test favorites and underdogs.

Use explicit examples with known answers.

==================================================
32. REPORTING
==================================================

Weekly player output should eventually resemble:

Player: Example RB
Team: ABC
Opponent: XYZ

Model P(TD): 48.2%
Market baseline P(TD): 43.1%

Expected TD: 0.658
Expected team offensive TD: 3.05

Rush xTD share: 31%
Receiving xTD share: 8%
Goal-line opportunity share: 54%

Best price: +140
Best sportsbook: ExampleBook
Raw implied probability: 41.67%

Model EV: +15.7%

Decision: BET or NO BET

Reason summary should be derived from model inputs, not generated from arbitrary narrative rules.

Example:

"High projected team scoring environment and strong goal-line share; recent TD conversion is below expected TD opportunity."

==================================================
33. PROSPECTIVE 2026 MODE
==================================================

Once the historical model is frozen:

Run paper predictions in 2026.

At approximately kickoff -60 minutes:

1. ingest latest available data
2. freeze feature snapshot
3. ingest available sportsbook prices
4. generate predictions
5. create decision
6. commit or persist immutable record
7. never revise that prediction after kickoff

After games:

settle predictions
record outcomes
record closing prices
calculate CLV
update dashboard/report

Do not retrain the model in response to individual weekly outcomes unless following a predefined retraining schedule.

==================================================
34. IMPLEMENTATION PHASES
==================================================

Build this project incrementally.

PHASE 0
Repository and environment setup.

Deliver:
- pyproject
- package structure
- settings
- DuckDB connection
- test framework
- logging
- CLI skeleton

PHASE 1
Point-in-time data proof of concept.

Select approximately 10 NFL games from 2024.

For each game reconstruct:

- kickoff timestamp
- prediction timestamp = kickoff -60 min
- teams
- players
- strictly historical player usage
- spread
- total
- implied team totals
- ATD quote snapshots if available
- final outcome

Demonstrate that no information after prediction_time enters any feature.

Do NOT proceed to machine learning until Phase 1 tests pass.

PHASE 2
Historical player feature pipeline.

Build:

- usage
- shares
- team context
- opponent context
- red-zone usage
- goal-line usage
- end-zone usage

PHASE 3
Play-level xTD engine.

Build rushing and receiving touchdown probability models.

Validate:

- calibration
- intuitive monotonic relationships
- no leakage

PHASE 4
Team touchdown environment model.

Build market-only baseline and football challenger.

PHASE 5
Player probability baselines.

Build:

historical TD-rate baseline
hierarchical model
calibrated logistic model

PHASE 6
LightGBM challenger and position models.

Only retain them if probability performance improves.

PHASE 7
Odds-backed betting simulation.

Use timestamped 2023-2025 historical ATD prices.

PHASE 8
Freeze model and run 2025 untouched evaluation.

PHASE 9
Prospective 2026 paper predictions.

==================================================
35. PHASE 1 ACCEPTANCE CRITERIA
==================================================

Before any predictive modeling is allowed:

For the 10-game proof-of-concept sample, produce an audit table containing:

game
player
kickoff_time
prediction_time
latest_player_game_used
latest_team_game_used
odds_timestamp
feature_available_at_max
result

Require:

latest_player_game_used < prediction_time
latest_team_game_used < prediction_time
odds_timestamp <= prediction_time
feature_available_at_max <= prediction_time

Create automated pytest tests enforcing these rules.

Also manually print several representative player feature histories so the calculations can be inspected.

==================================================
36. SOFTWARE QUALITY
==================================================

Require:

type hints
docstrings for important transformations
small testable functions
deterministic random seeds
structured logging
no hard-coded API keys
.env support
caching of external API calls
rate-limit handling
retry logic
data provenance fields

Do not silently fill critical missing data.

Log missingness.

Distinguish:

0
missing
not applicable

when those meanings differ.

==================================================
37. IMPORTANT CODING BEHAVIOR
==================================================

Do not rush ahead and build every phase.

Start with Phase 0 and Phase 1 only.

After Phase 1:

1. run the full test suite
2. inspect the 10-game audit
3. identify any leakage or data-quality concerns
4. summarize exactly what was built
5. document assumptions
6. recommend any required schema changes
7. STOP

Do not begin Phase 2 until explicitly instructed.

==================================================
38. FIRST TASK
==================================================

Begin now with:

PHASE 0
and
PHASE 1.

Before writing significant implementation code:

1. inspect the relevant GitHub repositories above
2. confirm nflverse/nflreadpy schemas required for the proof of concept
3. confirm how the historical odds source represents player_anytime_td
4. define one canonical spread sign convention
5. write the market-math unit tests first

Then build the 10-game point-in-time reconstruction.

The purpose of Phase 1 is not to make predictions.

The purpose is to prove that the future modeling dataset can be trusted.

==================================================
39. PHASE 4.5 DIAGNOSTIC EXHIBITION (ADDED SEPTEMBER 13, 2026)
==================================================

The September 13, 2026 full slate is a diagnostic, end-to-end, point-in-time
replay between Phases 4 and 5. It is not model training, validation, feature
selection, calibration, or threshold optimization. The complete instruction
for this one-off milestone is the user's September 13, 2026 Phase 4.5 request.

Freeze and SHA-256 hash the T-60 player predictions and all available T-60
ATD quotes before loading any current-game result. Settle in a separate stage
only after every slate game is final. Preserve both the best-price and median
price diagnostics, the predeclared EV thresholds, calibration snapshot, and
closing-line comparison. Keep the Phase 1-4 checkpoints frozen.

Flag every September 13, 2026 prediction and result as
`diagnostic_exhibition_slate = TRUE`. Outcomes from this date must never be
used to choose Phase 5 features or model class, tune hyperparameters, select
ensemble weights or betting thresholds, fit calibration, or alter the
Phase 4.5 player allocation. A one-day result provides essentially no
statistical evidence of a sustainable betting edge. Stop after Phase 4.5;
Phase 5 requires a new explicit instruction.

==================================================
40. PHASE 4.6 REPLAY INTEGRITY HARDENING (ADDED SEPTEMBER 14, 2026)
==================================================

Phase 4.6 changes replay infrastructure only. It does not refit or alter a
Phase 1-4 model, change the Phase 4.5 player allocation or betting rule, or
use September 13 results to choose prediction logic. The original September
13 prediction, quote, settlement, audit, and sensitivity artifacts remain
unchanged and permanently carry `diagnostic_exhibition_slate = TRUE`.

Future prospective Stage A events use a closed-schema pregame snapshot with
only historical player opportunity summaries, a frozen team expectation,
timestamped T-60 quotes, and contemporaneous official NFL availability
evidence. The offline prediction worker has no network, 2026 game-stat, or
settlement reader. It rejects unknown fields at all nested levels, and its
frozen prediction rows contain no settlement or result placeholders. Stage B
is a separate command that verifies Stage A hashes before requesting final
game data, refuses to run until the scoreboard marks the game final, and
writes its output outside the Stage A artifact tree.

At T-60, a known official inactive/out status excludes a player from betting
eligibility. Both a complete official availability snapshot for the player's
team and positive official active-roster evidence for that player must be
documented; otherwise the prediction remains visible but no wager is eligible.
Any pre-cutoff inactive/out status overrides active evidence. Later DNP
information never rewrites Stage A. Stage B records
`eligible_at_prediction_time`, `inactive_known_at_prediction_time`,
`ultimately_played`, and `settlement_status` separately. A DNP bet is graded
only when both participation evidence and the relevant sportsbook's documented
DNP rule are available; otherwise settlement remains pending.

All book quotes retain price, book, quote timestamp, and age at T-60. Best and
median prices, book count, and an outcome-independent extreme-price flag are
reported. The flag is descriptive and never removes a price or changes a bet.

The prospective task scheduler discovers future events daily, registers an
event task at each kickoff minus 60 minutes, and invokes Stage A only then.
The event-odds request cannot pass market updates after that cutoff to the
worker. Missing official availability coverage results in no eligible bets.
No Stage B task is scheduled automatically; the separate settlement command
must be invoked after the game is final. Stop before Phase 5.

==================================================
41. PHASE 5 PLAYER ANYTIME-TOUCHDOWN BASELINES (AUTHORIZED SEPTEMBER 14, 2026)
==================================================

Phase 5 builds pregame player-game binary probabilities for at least one
eligible rushing or receiving touchdown. Quarterback passing touchdowns do
not count. Eligible player positions are RB, WR, TE and QB; include FB only
where point-in-time feature quality permits. The historical player universe
must come from pregame knowledge, never current-game participation.

Use 2017-2023 for development where strict input coverage permits, 2024 for
chronological validation and model comparison, and leave the 2025 historical
holdout entirely untouched. September 13, 2026 is permanently a diagnostic
exhibition. Its outcomes, bets, CLV and thresholds may never inform Phase 5
features, weights, model selection or calibration. Preserve all Phase 0-4.6
predictive models and artifacts as frozen inputs.

Compare strictly lagged historical TD-rate summaries, a Phase 4 team-TD
hierarchical player allocation, and calibrated logistic regression. Conduct
controlled actual-TD, xTD, usage, and team-environment feature-family tests.
Fit base logistic models on training observations and any probability
calibrator on a separate chronological period. Compare raw and sigmoid
calibration; use isotonic only if supported by sample size and validation.
Do not fit tree models, neural networks, or complex ensembles in Phase 5.

Evaluate primarily with 2024 log loss, Brier score, calibration intercept,
slope and expected calibration error. Report ROC-AUC and PR-AUC secondarily,
probability buckets, position diagnostics, standardized coefficients,
multicollinearity, missingness, and team-TD coherence. Where strict T-60
player ATD prices exist, benchmark raw market implied probability on matched
rows only; never include player price in the football model. Do not choose
betting thresholds or optimize return.

Save chronological 2024 player-game validation predictions and a report that
answers whether xTD beats historical TD rate, hierarchy beats historical TD,
logistic beats hierarchy, football beats the limited market benchmark, and
whether logistic probabilities materially violate team coherence. Include
successful and missed representative examples. Run all prior tests and
verifiers plus a Phase 5 verifier, then commit/tag `phase5-complete` and stop
before Phase 6. The user's Phase 5 instruction is the authoritative complete
requirement for this milestone.

==================================================
42. PHASE 6 POSITION AND NONLINEAR PLAYER MODELS (AUTHORIZED SEPTEMBER 14, 2026)
==================================================

Phase 6 preserves all Phase 0-5 predictive artifacts, the Phase 4.6 replay
firewall, the untouched 2025 historical holdout, and the permanent September
13, 2026 diagnostic exhibition exclusion. It compares position-specific RB,
WR, TE, and QB rushing/receiving anytime-TD logistic models, one disciplined
LightGBM model family, and a simple blend based only on chronological OOF
component predictions. QB passing touchdowns never count. Player ATD odds
remain a separate T-60 benchmark, never a football-model feature. Do not
optimize betting returns or build XGBoost, neural networks, or a broad model
search.

Use lagged Phase 2 and 3 role, xTD, team environment and opponent fields,
plus the frozen Phase 4 team TD expectation. Test core usage with and without
xTD separately for RB, WR and TE. Do not fabricate historical designed-run,
scramble, air-yard, or route participation features. Build 2023 component
OOF predictions from 2022-only fits, select fixed simple-blend weights on
2023, and validate in 2024. Calibrate only on chronologically separate data.

The predeclared model configs, five blend weights, chronological split, and
materiality/uncertainty promotion gates are in `docs/PHASE6_PROTOCOL.md`.
Evaluate log loss first, then Brier score and calibration, overall and by
position, with early/middle/late season stability and game-cluster uncertainty.
Record every LightGBM configuration and feature contribution diagnostic.
Promote a challenger only if it beats the frozen Phase 5 hierarchical 0.3804
log-loss champion by the predeclared material margin without calibration,
position, stability or integrity problems. It is acceptable to retain the
Phase 5 hierarchy as `PHASE6_CHAMPION`.

Save validation predictions, true OOF component predictions, experiment
metrics, calibration tables, position diagnostics, market comparison, and
an explicit Phase 6 report. Run the full prior tests/verifiers and a new
Phase 6 verifier. Commit/tag `phase6-complete`, then stop before Phase 7.
The user's Phase 6 instruction is the authoritative complete requirement.
