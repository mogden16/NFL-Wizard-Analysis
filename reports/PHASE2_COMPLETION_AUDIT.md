# Phase 2 completion audit

The original master project specification was recovered in full from the prior
pasted attachment and copied **byte-for-byte** to [docs/PROJECT_SPEC.md](../docs/PROJECT_SPEC.md).
Both files have SHA-256
`8e9ed754d12570003615f846132f78d093f1ac693b0cc303f47b4f16f9188cb7`.
No missing requirements were invented. Its recommended football-history start is
2017; 2025 remains the final historical holdout for later modeling.

## What the earlier table actually covered

Before this audit, `phase2_2024_player_features.parquet` covered **2024 only**:
8,108 player rows in 256 regular-season games, Weeks **2–18**. It contained no
Week 1 game. The exact distribution is:

| 2024 week | Games | Player rows |
| ---: | ---: | ---: |
| 2 | 16 | 339 |
| 3 | 16 | 398 |
| 4 | 16 | 433 |
| 5 | 14 | 399 |
| 6 | 14 | 404 |
| 7 | 15 | 455 |
| 8 | 16 | 504 |
| 9 | 15 | 478 |
| 10 | 14 | 459 |
| 11 | 14 | 469 |
| 12 | 13 | 444 |
| 13 | 16 | 548 |
| 14 | 13 | 456 |
| 15 | 16 | 569 |
| 16 | 16 | 575 |
| 17 | 16 | 583 |
| 18 | 16 | 595 |
| **Total** | **256** | **8,108** |

Thus the earlier implementation had validated the method on one season; it did
**not** yet cover the specification's intended multi-season football history.

## Expanded historical feature set

The same source-game filter, window summaries, denominator pairing, source hashing,
and one-day assumed availability boundary now run separately for every regular
season from **2017 through 2025**. The combined table is
`data/derived/phase2_2017_2025_player_features.parquet`: **70,319 rows in 2,240
games**. Per-season Parquet files remain alongside it, with content hashes in
[the artifact manifest](phase2_artifact_hashes.json). The 2024 run reuses the
frozen Phase 1 nflverse sources and retains its original 8,108 rows. The 2025
output contains only pregame feature candidates; no 2025 outcome labels, model
fits, calibration, tuning, or betting evaluation were created. No 2026 rows are
included in this completed historical range.

Windows are still last 3, 5, and 8 eligible player games, season-to-date, and
three-game-half-life EWMA. They use **same-season** history only. Week 1 therefore
has no player rows in any season: 15 such games in 2017 and 16 in each later
season. This is a deliberate coverage limit, not a zero-filled early-season
prior. The 2022 schedule contains 271 completed regular-season games; the
cancelled game is not fabricated. Candidate rows identify players with prior
same-team weekly history, not necessarily a verified pregame active roster.

## Feature-availability matrix

The percentages below are the proportion of **produced player rows** with a
non-null last-three-game feature. They do not count omitted Week 1 games as
covered. The [full season matrix](phase2_feature_availability_by_season.csv)
also separates target, touch, snap, red-zone target, and goal-line shares;
the [season-by-window matrix](phase2_feature_availability_by_season_window.csv)
reports present, missing, and coverage for every important family in all five
windows. Zero usage is an observed zero; a zero share denominator or unknown
source value is null.

| Season | Rows | Games | Carries | Targets | Carry share | Snaps/share | RZ carries | RZ targets | Goal-line share | End-zone targets | End-zone share | Routes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 7,008 | 241/256 | 100% | 100% | 100% | 99.93% | 100% | 100% | 94.46% | 99.96% | 97.30% | 0% |
| 2018 | 7,230 | 240/256 | 100% | 100% | 100% | 100% | 100% | 100% | 93.62% | 100% | 98.20% | 0% |
| 2019 | 7,121 | 240/256 | 100% | 100% | 100% | 99.83% | 100% | 100% | 93.27% | 99.87% | 96.24% | 0% |
| 2020 | 7,642 | 240/256 | 100% | 100% | 100% | 100% | 100% | 100% | 95.88% | 99.96% | 95.83% | 0% |
| 2021 | 8,487 | 256/272 | 100% | 100% | 100% | 99.84% | 100% | 100% | 93.13% | 99.96% | 96.76% | 0% |
| 2022 | 8,322 | 255/271 | 100% | 100% | 100% | 99.93% | 100% | 100% | 93.61% | 100% | 95.59% | 0% |
| 2023 | 8,006 | 256/272 | 100% | 100% | 100% | 99.88% | 100% | 100% | 92.48% | 100% | 95.78% | 0% |
| 2024 | 8,108 | 256/272 | 100% | 100% | 100% | 99.58% | 100% | 100% | 96.02% | 100% | 97.26% | 0% |
| 2025 | 8,395 | 256/272 | 100% | 100% | 100% | 99.42% | 100% | 100% | 93.11% | 100% | 96.30% | 0% |

Target and touch shares are also 100% in all seasons. Red-zone carries, carries
inside the 10 and 5, goal-line opportunity counts, and red-zone target counts
have 100% non-null coverage in produced rows because all 2,383 completed
regular-season games have PBP with final scores matching the schedule. Some
shares are null when the paired team denominator is zero. PFR snap counts are
joined through the GSIS/PFR identity crosswalk; a missing game-level snap value
now makes the entire affected candidate window null.

End-zone targets use `air_yards >= yardline_100` on prior target plays. Four
eligible PBP targets in 2017, 2019, 2020, and 2021 lack air yards. Their player
and team end-zone counts are marked unknown, and every window containing such a
game is null for the affected measure. The route field in participation data
describes only a primary receiver and the 2024+ participation releases were not
available in-season; route participation remains null in **all** years rather
than filling an unreliable proxy.

## Point-in-time limits and verification

No feature row uses its own game in any player, team, opponent, numerator, or
denominator history. `phase2-verify` checks all 70,319 target rows against the
frozen source schedules, same-team player-stat keys, source-game IDs, season,
kickoff, and source availability. It also checks source hashes, all season and
window coverage counts, share bounds, null route columns, and exact membership
of the combined table. Tests specifically inject same-game usage into all
important shares and verify unchanged pregame features. All 40 Phase 1 audit
player-games remain in the 2024 table, and their carries and targets reconcile.

The 24-hour availability timestamp for prior player/team stats, PBP, and PFR
snaps remains **assumed**, not observed. Today's historical source files can
include retrospective corrections. The frozen hashes establish reproducibility
of this reconstruction, but archived in-season source versions and actual
publication times remain necessary for a fully evidenced point-in-time backtest.
This audit does not authorize model fitting or Phase 3.

Manual cross-season spot checks against the frozen weekly player-stat files
also reconciled last-three carries and targets for Benny Cunningham in 2017
Week 4 (0 carries, 1.5 targets per game), Samaje Perine in 2021 Week 4
(2 carries, 1 target), and James Conner in 2025 Week 4 (10.667 carries,
3 targets). Their latest assumed source availabilities all precede their
respective prediction timestamps; no 2025 outcome was inspected for this check.

Source notes: [nflverse availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html),
[player stats](https://nflreadr.nflverse.com/reference/load_player_stats),
[PFR snap counts](https://nflreadr.nflverse.com/reference/load_snap_counts.html),
and [participation data dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html).
