# Phase 2 historical player feature pipeline — original 2024 validation

The pipeline now covers 2017–2025. See the
[completion audit](PHASE2_COMPLETION_AUDIT.md) for the expanded source and
feature-availability matrix. This report retains the detailed 2024 validation.

**Completed through feature generation and verification only.** No xTD engine,
predictive model, window selection, betting optimization, or 2025 holdout outcome
evaluation was started. Phase 1 remains preserved at Git commit `309163e` and tag
`phase1-complete`. The original specification has since been recovered verbatim in
`docs/PROJECT_SPEC.md`.

Run `python -m uv run nfl-td phase2-features` to rebuild the 2024 table at
`data/derived/phase2_2024_player_features.parquet` as part of the full history,
then run `python -m uv run nfl-td phase2-verify`. The latter checks frozen source hashes,
all 8,108 row keys and source-game lists, same-team player history, kickoff-minus-60-
minute timestamps, the one-day availability boundary, maximum availability,
share ranges, and missingness counts. The new tests mutate a game's own carries,
targets, touches, snaps, red-zone usage, and team denominators and verify that none
changes its pregame features.

## Feature contract

Each row represents a 2024 regular-season team/player/game candidate whose player
ID and same-team weekly history were already observed before prediction. A game
in which the player first appears for that team cannot have a row yet. This is a
historically seen candidate universe, **not a pregame active-roster guarantee**;
injured, benched, or traded players may remain candidates. Player name and position
come from the latest eligible prior weekly record, never the current game.

Windows are the last 3, 5, or 8 eligible *player games*, season-to-date, and EWMA
with a three-player-game half-life. Team and opponent windows use their own last
3/5/8 games. Counts are per-game averages; each share is the sum of player counts
divided by the sum of matching team counts over the **same historical player games**.
EWMA uses matching weighted numerator and denominator. No window is selected or
optimized in this phase. A zero denominator yields null, not zero. A missing
source-game value makes the corresponding whole candidate window null.

| Family | Definition and source |
| --- | --- |
| Carries, targets, receptions, touches | Weekly nflverse player stats; touches = carries + receptions. Carry, target, and touch shares use same-game team totals from weekly team stats, but only for games prior to the target game. |
| Snaps and snap share | PFR game-level offensive snaps joined to GSIS player IDs through the nflverse player crosswalk. Team snaps are the rounded median of `offense_snaps / offense_pct` among players with positive values in that game. Missing or ambiguous joins stay null. |
| Red-zone carries, inside-10 carries, inside-5 carries | Prior-game PBP rushing attempts at `yardline_100 <= 20`, `<= 10`, and `<= 5`; kneels and two-point attempts excluded. |
| Red-zone and end-zone targets | Prior-game PBP pass attempts with a receiver ID; red zone means `yardline_100 <= 20`; end zone means `air_yards >= yardline_100`. Two-point attempts excluded. This end-zone flag is a geometric proxy, not tracking of the catch point. |
| Goal-line opportunity share | Player carries plus targets from inside the 5 divided by the team's corresponding prior-game opportunities. |
| Red-zone/end-zone target share | Player target count divided by the team's count for matching prior player games. |
| Team offense | Prior team carries, targets, red-zone opportunities, offensive plays (`carries + pass attempts + sacks suffered`), and rushing-plus-passing TDs per game. |
| Opponent context | What a defense allowed in its prior games: opponent offensive plays, carries, targets, red-zone carries/targets, and rushing-plus-passing TDs. No target-game result is used. |
| Routes | Null. The available participation file gives a primary receiver's route, not every receiver's route; 2024 participation was released only after the season. It is not a point-in-time source for 2024 games. |

The previous Phase 1 assumption remains explicit: all prior-game football summaries,
PBP, and snap counts are treated as available **24 hours after scheduled kickoff**.
This is an *assumed* availability timestamp, not an observed publication timestamp.
The current nflverse and PFR-derived files can include later corrections. Frozen
content hashes make this run reproducible, but archived versions with publication
metadata are needed for a fully evidenced historical backtest. The schedule is also
a current historical file, not a record of every old published kickoff revision.
The player ID crosswalk is used only to identify past snap counts, not as a
performance predictor.

## Coverage and manual review

The table has **8,108 rows across 256 games**: 2,025 RB, 3,354 WR, 1,734 TE,
and 995 QB rows. The remaining 16 games are Week 1, with no prior 2024 same-team
history. All 272 schedule games have PBP and matching PBP final scores. The
[coverage file](phase2_feature_coverage.json) gives present/missing counts for
every candidate column. For last-3 features, carries, targets, touches, all
red-zone counts, and team/opponent context have 8,108 values; snap share has
8,074 (34 missing), goal-line opportunity share has 7,785 (323 null because
the paired denominator is zero), red-zone target share has 8,065 (43 null),
and end-zone target share has 7,886 (222 null). Route participation has no
reliable values. PFR has 25,398 offensive player rows for these games and
25,359 mapped uniquely to GSIS IDs; 39 lack a unique usable match.

The [12 representative histories](phase2_representative_histories.json) include
three RBs, WRs, TEs, and QBs from three different Week 4 games, with individual
prior player, team, and opponent records and assumed availability timestamps.
The [Week 4 review CSV](phase2_week4_examples.csv) shows 433 candidate rows.
All 40 Phase 1 audited player-games are present in the Phase 2 table; their
last-three carries and targets reconcile exactly to Phase 1 historical totals.
One schema meaning is deliberately clearer here: Phase 1's
`historical_touch_share` was a carries-plus-targets *opportunity* proxy, whereas
Phase 2 `touches` and `touch_share` mean carries plus receptions. These columns
must not be treated as interchangeable in later modeling.
Manual examples checked against the source histories:

- Ezekiel Elliott at 2024 Week 4 DAL–NYG uses carries 10, 6, 3 from Weeks
  1–3: `last3_carries_per_game = 19/3 = 6.3333`.
- Brandin Cooks in that game uses targets 7, 2, 6: `last3_targets_per_game = 5`.
- Jake Ferguson uses Week 1 and Week 3 targets 5 and 11, with no Week 2 player
  record: `last3_targets_per_game = 8` over two eligible player games.
- Bo Nix at 2024 Week 4 DEN–NYJ uses Week 1–3 carries 5, 4, 9:
  `last3_carries_per_game = 6`.

The cited source games in each example occur before its prediction. For the
Thursday Dallas game, the prediction is 2024-09-26 23:15 UTC; its latest assumed
source availability is 2024-09-23 20:25 UTC. For the Sunday Carolina example,
the prediction is 2024-09-29 16:00 UTC and latest assumed source availability
is 2024-09-25 00:15 UTC. The full verifier checks this boundary for every row.

Source documentation: [nflverse update and availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html),
[PFR snap-count dictionary](https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html),
and [participation dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html).
