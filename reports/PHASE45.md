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

## Settlement and interpretation

Pending the final Sunday game. The separate settlement stage refuses to load
game summaries unless all 13 scoreboard records are final. It verifies the
frozen Stage A hashes before any result access. The final section will report
both portfolios, all four predeclared EV thresholds, descriptive calibration,
closing-line value, the largest misses, and the eligibility audit.
