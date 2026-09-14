# Phase 7 predeclared historical ATD backtest protocol

Written before collecting new 2023/2024 player odds or reading Phase 7
settlement outcomes. The Phase 6 champion is the frozen Phase 5 hierarchical
probability. There is no football-model refit, player probability recalibration,
staking optimization, or use of September 13, 2026. The 2025 holdout is sealed.

## Acquisition and chronology

Use every 2023 and 2024 regular-season game with a frozen Phase 5 player
universe row and a Phase 4 strict market event ID, subject to actual ATD
availability. Request one US-region `player_anytime_td` market snapshot per
game at kickoff minus 60 minutes, selecting only quotes whose market-update
timestamp is at or before that cutoff. The provider returns the nearest prior
snapshot. Record missing markets and quotes, never substitute later prices.
Stage A receives a closed-schema projection without current-game outcomes,
participation, scores, closing odds or settlement fields and freezes its CSV
and SHA-256 manifest before Stage B can read any result. Stage B for 2023 runs
first; a 2023-only development rule specification is frozen and hashed before
Stage B for 2024 loads outcomes or closing quotes.

Closing prices, where available, use a separate historical snapshot at or
before kickoff. Best-price CLV compares with the same book's close; median
CLV compares with the median closing market. A missing same-book close stays
null. Positive CLV means the entry decimal price exceeded its closing price.

The historical player universe is the frozen Phase 5 prior-appearance rule.
It is not proof of an active roster. Stage A never removes a player based on
current-game participation. Stage B treats a player as `played` only with
affirmative current-game PFR offensive-snap, player-stat, or play-by-play
opportunity evidence. For all others, historical DNP status and sportsbook
DNP policy are unknown; record `settlement_status=policy_unknown`, do not
grade a wager, and report counts separately. No DNP is silently counted as a
loss or removed from the original Stage A prediction file.

## Diagnostics and rule grid

Edge buckets for both seasons: <= -10pp; -10 to -5; -5 to -2.5; -2.5 to 0;
0 to +2.5; +2.5 to +5; +5 to +10; >+10. Analyze both best and median prices,
with one-unit graded wager returns. Quote-age buckets: <=5, (5,15], (15,30],
and >30 minutes. Odds buckets: negative, +100-199, +200-299, +300-499,
and +500+. Report book count, price dispersion and CLV. These buckets are
diagnostics, never retroactively optimized filters.

The complete fixed candidate grid is probability-edge minimum
{0, 0.025, 0.05, 0.075, 0.10}, EV minimum {0, 0.05, 0.10, 0.15}, and minimum
book count {1, 2, 3}. A candidate must satisfy both edge and EV at the best
legitimate T-60 price. For every candidate, report the same picks at best and
median prices. Flat stake is one unit; no Kelly staking.

2023 narrows the 60 candidates to at most five **before** opening 2024
settlement. A shortlisted rule needs at least 100 graded 2023 bets, positive
best and nonnegative median ROI, nonnegative average same-book CLV with at
least 50% CLV coverage, and no single winner contributing over 50% of gross
winnings. Rank qualifying rules by median ROI, then CLV, then graded sample
size. This is a development heuristic, not a validated optimal threshold.
If none qualifies, freeze an empty shortlist and use NO BET as the primary
2024 validation benchmark; still report all predeclared diagnostic buckets.

At most one rule advances from 2024 only if it has at least 100 graded bets,
positive best and median ROI, nonnegative average CLV with at least 50%
same-book coverage, positive or zero units in at least two of the fixed
chronological thirds (weeks 2-6, 7-12, 13+), no greater than 50% of gross
winnings from its top two winners, and no more than half of positive net units
from >30-minute quotes or +500-and-longer winners. These conditions use no
new post-hoc thresholds. If no shortlisted rule passes, set
`PHASE7_RESULT=NO_ROBUST_EDGE`. Never select a rule solely by highest ROI.

Bootstrap one-unit ROI by resampling games (2,000 replicates, fixed seed 1729)
for the shortlist and 2024 validation. Report 95% intervals, drawdown and
losing streak. Fit 2023-only market-logit and market-logit-plus-football-edge
logistic diagnostics, then evaluate both on 2024. No market-residual model
influences the rule grid or football champion.
