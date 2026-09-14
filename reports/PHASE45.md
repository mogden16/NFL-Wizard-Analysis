# Phase 4.5 — September 13, 2026 diagnostic exhibition

This is a one-day, end-to-end point-in-time *reconstruction*, not a live betting
record or a Phase 5 player model. September 13 outcomes are marked
`diagnostic_exhibition_slate = TRUE` and cannot enter later feature, model,
calibration, ensemble, or threshold decisions. The Phase 1–4 checkpoints remain
unchanged.

## Frozen prediction stage

All 13 Sunday games used The Odds API's historical event snapshot at kickoff
minus 60 minutes. Every selected ATD, spread, and total market update was at
or before its game's prediction time. The files were created at
`2026-09-14T01:24:36.529603+00:00`, after the day games had finished, as a
historical T−60 reconstruction. The prediction stage's code path loads no
September 13 scores, play-by-play, player statistics, touchdown labels,
post-cutoff quotes, or closing odds. It was committed separately in `b414798`.
The provider [documents](https://the-odds-api.com/historical-odds-data/) that
historical event requests return the nearest snapshot at or before the
requested time; the implementation also checks each market's own update time.

| Frozen file | SHA-256 |
| --- | --- |
| `phase45_stage_a_predictions.csv` | `41d9e8f6af1a7550248c04abfa90dc4b2d20ca69e12e408d4b44c23dc79ffc62` |
| `phase45_stage_a_all_quotes.csv` | `097feaffc0efbde6dde6c0f1e4a6f89e69fa9095c563febf5b9c99f588ec5f1d` |

The table contains 426 unique quoted players and 2,763 sportsbook quotes from
eight books. The complete quotes file includes the book and quote timestamp
for each quote. Of the quoted players, 201 have an unambiguous match to at
least three 2025 regular-season player games on the same team and three 2025
games with scored xTD opportunities. The remaining 225 retain null model
probabilities and a missing-history reason; no probability is fabricated for
them. The 201 modeled players comprise 51 RBs, 79 WRs, 51 TEs, and 20 QBs.

The Phase 4 market-only Poisson model was refit only on its original
2021–2023 development data, with no 2024 validation or 2025 outcome in its
fit. Both Phase 3 play-level logistic models retained their original
2017–2022 fitting and 2023 calibration; they *inferred* xTD for 2025 plays
without fitting to 2025. All 2025 source games were complete long before the
September 2026 cutoffs. Their publication time remains the Phase 2
**assumption** of kickoff plus 24 hours, never an observed timestamp.

The fixed player allocation uses a weighted sum of prior opportunity shares:
45% total xTD, 8% rushing xTD, 8% receiving xTD, 8% goal-line opportunity,
6% carries inside the 5, 4% carries inside the 10, 5% red-zone targets,
5% end-zone targets, 5% carry share, 5% target share, and 1% snap share.
Missing families are renormalized over available weights; snap share is null
for all modeled players because its 2026 roster continuity was not established.
Each team's scores are then normalized across its 2025 historical candidate
pool and multiplied by the Phase 4 expected offensive touchdowns. The ATD
probability is `1 - exp(-expected_player_TD)`. This formula was specified and
frozen before settlement; none of its weights were fitted to this slate.

The predeclared diagnostic decision was best-price EV at least 5% **and**
model probability minus raw best-price implied probability at least 2.5
percentage points, with a one-unit flat stake. The frozen file marks 33
best-price bets and 26 independently selected median-price bets. The median
price is the arithmetic median of the available book payoffs, so it can be
interpolated between two books and is a hypothetical reference, not necessarily
an executable quote.

## Integrity and important limitations

The original CSV bytes are immutable through an overwrite guard, SHA-256
manifest, and git commit. A clean-process reproduction matched all 426 rows
and decisions numerically within `4.5e-16`; its hash differed because Polars
floating-point aggregation order changed some least-significant digits. This
does not change any reported bet, but exact bitwise regeneration is not yet
guaranteed. The original prediction hash above remains the settlement anchor.

During preliminary setup, a 2026 nflverse schedule table was inspected before
the frozen Stage A run. Its schema can include results. The Stage A builder did
not use that table or any of its outcome fields, and an isolated rerun matched
all predictions and decisions numerically, but the broader work session did
not meet a literal zero-access-to-result-bearing-files standard. This is a
procedural limitation of the exhibition.

The eligibility screen is **weaker than the requested standard**. A T−60
bookmaker quote plus 2025 same-team history does not prove a player was on the
active 2026 game-day roster. A retrospective injury-status check already
identified at least one frozen bet on a player reported inactive before T−60.
The prediction artifact is preserved as-is; settlement reports both its raw
decision and a separate inactive/void audit. Any reported ROI from the raw
33-bet set must therefore be read as a flawed diagnostic, not a verified
portfolio of legitimate wagers. The 225 unmatched players include newcomers,
transfers, and players without enough same-team history; a complete 2026
pregame roster and inactive archive was not established.

The allocation also computes each player's xTD share over that player's last
eight *appearances*, not over the team's last eight games. Fringe players with
few appearances can therefore receive inflated shares and strikingly large
longshot EV values. Those are baseline failure signals, not evidence of an
edge. Route participation is unavailable and unused. 2025 snaps were not
carried into a 2026 snap-share feature without reliable continuity.

## Settlement and predeclared portfolios

Stage B began only after all 13 games were final. It verified the frozen Stage
A SHA-256 hashes first, then saved the final [ESPN scoreboard and game-summary
source](https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20260913)
under `data/raw/`. Across the 13 games, rushing plus receiving TD counts in
player boxscores agreed with the scoring-play list for every game. Three
defensive/return TD scorers were identified separately; none had an ATD quote
in the frozen table. The settled 426-player [CSV](phase45_settled_predictions.csv),
[sortable HTML](phase45_full_slate.html), and machine-readable
[metrics](phase45_report.json) preserve every quoted player, including those
without a model probability. The frozen predictions were not modified.

The predeclared 5% EV / 2.5-point edge rule at best T−60 prices selected 33
one-unit wagers. Five won and 28 lost: 15.2% hit rate, 33 units risked, 40.2
units of *winning profit* before losing stakes, 45.2 units returned including
winning stakes, **+12.2 net units and +37.0% ROI**. Mean best price was about
`+1290` American (13.90 decimal), mean model probability 21.5%, mean raw
probability edge 9.2 points, and mean claimed EV +136.9%. Those extraordinary
figures themselves signal how aggressively the unvalidated allocation priced
longshots. They are not evidence that the market was mispriced by that amount.

| Predeclared EV threshold; edge ≥2.5pp | Best bets | Wins | Avg American odds | Best net units | Best ROI | Median bets | Median wins | Median net units | Median ROI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EV > 0% | 33 | 5 | +1290 | +12.20 | +37.0% | 26 | 4 | +6.65 | +25.6% |
| EV ≥ 5% | 33 | 5 | +1290 | +12.20 | +37.0% | 26 | 4 | +6.65 | +25.6% |
| EV ≥ 10% | 33 | 5 | +1290 | +12.20 | +37.0% | 26 | 4 | +6.65 | +25.6% |
| EV ≥ 15% | 32 | 4 | +1325 | +10.20 | +31.9% | 26 | 4 | +6.65 | +25.6% |

The median portfolio applies the same model and eligibility rules but
reselects bets using median-market EV. To isolate *price alone*, settling the
identical 33 best-selected wagers at median prices yields only **+2.275 units,
+6.9% ROI**. Thus line shopping changes this one-day result materially;
roughly 9.925 of the best portfolio's 12.2 net units disappear at median
prices. Three longshot wins explain most of the best-price return:

| Player | T−60 best | Model P | Result | Best P/L |
| --- | ---: | ---: | ---: | ---: |
| Jahdae Walker | +1500 | 19.1% | 1 TD | +15.0 |
| Elic Ayomanor | +1200 | 18.5% | 1 TD | +12.0 |
| Devin Singletary | +900 | 28.3% | 1 TD | +9.0 |
| Kyle Monangai | +220 | 36.7% | 1 TD | +2.2 |
| Aaron Jones | +200 | 38.1% | 1 TD | +2.0 |

## Eligibility and executable-price audit

The retrospective ESPN injury records identify **nine quoted players** as
inactive with a record timestamp at or before T−60, and **17** as inactive by
kickoff. One predeclared best-price wager, **DJ Giddens**, was recorded
inactive at `15:50 UTC`, ten minutes before his `16:00 UTC` cutoff, yet the
frozen quote updated at `15:54:57 UTC` and the frozen decision marked a bet.
**Ja'Tavion Sanders** was recorded inactive at `16:02 UTC`, after his T−60
cutoff but before kickoff; his frozen bet would ordinarily be void if he did
not participate. This is direct evidence that available quotes were an
insufficient eligibility screen. The postgame injury feed's timestamps are
retrospective metadata, so this audit does not retroactively repair Stage A.
Book acceptance and each book's void rule were not independently verified.

If those two selected inactives are treated as void, the best-price portfolio
has 31 settled bets, five wins, **+14.2 units and +45.8% ROI**; the independently
selected median portfolio has 24 settled bets, four wins, **+8.65 units and
+36.0% ROI**. These are post-settlement eligibility sensitivities, not a
replacement prediction or a newly selected strategy. No strict, fully
verified T−60 eligible portfolio can be claimed from this artifact.

## Descriptive probability and closing-line checks

Among 201 modeled players, the probabilities summed to 34.21 expected
scorers; 47 scored. This one-slate snapshot does **not** establish
calibration.

| Model P bin | Players | Expected scorers | Actual scorers |
| --- | ---: | ---: | ---: |
| <10% | 71 | 3.60 | 6 |
| 10–20% | 57 | 8.14 | 10 |
| 20–30% | 48 | 12.16 | 17 |
| 30–40% | 14 | 4.75 | 6 |
| 40–50% | 4 | 1.79 | 1 |
| 50–60% | 6 | 3.16 | 6 |
| 60%+ | 1 | 0.62 | 1 |

Model edge did not rise monotonically with realized TD rate. Among players
with 2.5–10 point claimed best-price edges, 2 of 23 scored (8.7%); among
those with 10–20 point edges, 3 of 7 scored (42.9%); none of three with
20-point-plus edges scored. The 151 modeled players whose probabilities were
*below* raw market implication scored at a 26.5% rate. These groups have
different baseline probabilities and tiny counts, so they cannot validate
or refute an edge. They do flag allocation error as a serious possibility.

Closing quotes were retrieved **after settlement**, separately from Stage A.
For 32 of the 33 best-price bets, the same sportsbook still had a recorded
pre-kickoff quote. Eleven had positive probability-based CLV, three negative,
and 18 unchanged; one had no comparable close. Mean raw implied-probability
CLV was **+0.253 percentage points**, median zero. The narrow positive mean
does not show consistent positive CLV or a sustainable advantage.

## Largest misses and attribution

The largest *underpredictions* were players who scored despite low baseline
probabilities: Jack Bech (7.0%), Kendre Miller (7.1%), Baker Mayfield (7.5%),
Juwan Johnson (7.5%), Cole Kmet (8.4%), and Caleb Williams (9.0%; two TDs).
The Bears scored **eight** offensive TDs against 2.60 expected, and the
Panthers scored five against 2.32 expected. That extraordinary game explains
several misses through team-environment variance as well as possible player
allocation error. New Orleans scored four against 2.18 expected, likewise
contributing to Miller and Johnson misses. Baker Mayfield's Buccaneers scored
two against 2.38 expected, so his individual score is more naturally ordinary
outcome variance or an allocation miss than a team-volume miss.

Large *overpredictions* that did not score included Saquon Barkley (49.6%;
Philadelphia scored three offensive TDs versus 2.70 expected), Jaylen Warren
(44.9%; Pittsburgh scored one versus 2.48), and Kimani Vidal (40.6%; the
Chargers scored two versus 3.21). Barkley's miss is consistent with ordinary
player-level variance; Warren and Vidal also had team-volume shortfalls.
The largest *claimed-edge* misses were Vidal, Raheim Sanders, Chimere Dike,
Bam Knight, and British Brooks. Their very long market prices and inflated
baseline shares point particularly to the last-eight-*appearances* allocation
problem, though a single zero-TD outcome cannot prove the cause. No weights,
thresholds, or feature choices were changed in response.

**Interpretation:** The provisional best-price ledger was profitable on this
Sunday: 33 marked bets, +12.2 units, +37.0% flat-stake ROI. It was **not** a
fully valid strict replay because the frozen eligibility rule let an already
inactive player through; actual sportsbook acceptance is unverified. Median
pricing materially reduced the same-bet result to +2.275 units. Claimed edge
did not track realized scoring consistently, and CLV was only slightly
positive on average with most lines unchanged. The largest misses include a
huge team-TD outlier and likely player-allocation weaknesses, alongside normal
touchdown variance. One NFL Sunday supplies essentially **no statistical
evidence of a sustainable betting edge**. The entire slate remains excluded
from Phase 5 development.
