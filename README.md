# nfl_td_model

Point-in-time NFL anytime-touchdown research through Phase 4, plus a one-day
Phase 4.5 diagnostic exhibition. Phase 5 player probability modeling has not begun.
The ten-game Phase 1 audit uses paid historical quotes and fails closed when a
required market timestamp is missing.

```powershell
python -m pip install uv
python -m uv sync --python 3.13 --extra dev
python -m uv run pytest
python -m uv run nfl-td phase1-audit
python -m uv run nfl-td phase1-verify
```

Set `ODDS_API_KEY` in `.env` only if the account can access paid historical event odds.
The historical API charges credits. The audit requests 2024 week-four games with snapshot
time equal to kickoff minus 60 minutes. Output and evidence are under `data/` and a human
readable report under `reports/`.

`home_spread` is the sportsbook handicap applied to the home score. A home favorite of
three has `home_spread=-3`; home and away implied points are `(total-home_spread)/2`
and `(total+home_spread)/2`. nflverse schedule spread/total columns have no quote
timestamp, so Phase 1 never treats them as strict market features.

Historical player/team game stats are considered available one full day after the
scheduled kickoff. This deliberately conservative proxy excludes same-day games and
is documented in the audit. It is not evidence of an actual data-publication timestamp.
Current nflverse files can be revised, so raw downloads are content-addressed and
source hashes retained. Results come from final play-by-play, cross-checked against
weekly stats when present; results are never feature inputs.

The 2025 season is reserved as a final holdout. No model code is part of this phase.
`phase1-verify` checks source hashes, history times, outcomes, and each reported market
quote against a saved Odds API response. The ten-game mechanical gate passes, but the
football-history publication times remain a documented proxy, not archived proof of
the actual 2024 source versions. See [the Phase 1 report](reports/PHASE1.md) before
using this as a historical backtest.

Phase 2 constructs strictly lagged 2017–2025 player features without selecting windows or
fitting models. It reuses the frozen local Phase 1 nflverse source files, so run the
Phase 1 audit first when setting up a new checkout:

```powershell
python -m uv run nfl-td phase2-features
python -m uv run nfl-td phase2-verify
```

The generated per-season and combined Parquet tables are under `data/derived/`;
coverage, representative source histories, and the exact feature contract are in
[the Phase 2 report](reports/PHASE2.md) and
[completion audit](reports/PHASE2_COMPLETION_AUDIT.md).
The same 24-hour *assumed* availability boundary applies to prior-game football data.
Routes remain null where a point-in-time receiver route source is unavailable.

Phase 4.5 reconstructs all September 13, 2026 games at kickoff minus 60 minutes.
The frozen prediction and quote CSVs, their SHA-256 manifest, and the separate
settlement report live under `reports/`. The exact replay commands are:

```powershell
python -m uv run nfl-td phase45-predict  # refuses to overwrite the frozen artifact
python -m uv run nfl-td phase45-verify
python -m uv run nfl-td phase45-settle   # requires all 13 games to be final
python -m uv run nfl-td phase45-verify
```

The September 13 outcomes are flagged `diagnostic_exhibition_slate` and are
forbidden as inputs to later model, feature, calibration, or betting-rule choices.
The one-day betting figures are descriptive and do not validate an edge.

Phase 4.6 adds a separate prospective replay path for games after September 13,
2026. The September 13 files are never regenerated. The Windows scheduler can
be installed with:

```powershell
./scripts/phase46_tasks.ps1
```

It discovers upcoming games each morning, registers one event task for each
T-60 cutoff, captures current odds, and invokes the offline Stage A worker in
a separate Python process. The Odds API documents that the live `/events`
endpoint does not consume quota; each event-odds request may consume credits.
The task needs the project `.venv` and `ODDS_API_KEY` in `.env`.

Official availability is a fail-closed input. For an event to have eligible
bets, put `data/pregame_availability/<event-id>.json` in place before T-60:

```json
{
  "event_id": "event-id",
  "availability": [
    {"player": "Example Player", "team": "ATL", "status": "inactive",
     "source_url": "https://www.nfl.com/news/example-official-game-day-inactives", "source_kind": "official_nfl",
     "published_at": "2026-09-20T15:00:00Z", "retrieved_at": "2026-09-20T16:00:00Z"},
    {"player": "Example Active Player", "team": "ATL", "status": "active",
     "source_url": "https://www.nfl.com/news/example-official-game-day-roster", "source_kind": "official_nfl",
     "published_at": "2026-09-20T15:00:00Z", "retrieved_at": "2026-09-20T16:00:00Z"}
  ],
  "availability_coverage": [
    {"team": "ATL", "source_url": "https://www.nfl.com/news/example-official-game-day-inactives",
     "source_kind": "official_nfl", "published_at": "2026-09-20T15:00:00Z",
     "retrieved_at": "2026-09-20T16:00:00Z", "complete": true}
  ]
}
```

Coverage must be complete for each team separately, based on a full official
game-day status source captured by T-60. An injury-report page listing only
some players must not be marked complete. Without documented complete coverage,
the worker still freezes quotes and predictions but marks wagers ineligible.
Each eligible player also needs affirmative official active-roster evidence;
absence from an inactive list alone does not establish eligibility. An
inactive/out record overrides an active record at T-60.
The availability file's source and timestamps are recorded; its authenticity
is an operator responsibility until an official machine-readable feed exists.

After the game is final, run Stage B as a separate process:

```powershell
./.venv/Scripts/python.exe -m nfl_td_model.phase46_settle reports/phase46/YYYY-MM-DD/EVENT_ID `
  --participation-file data/participation/EVENT_ID.json `
  --book-rules-file data/book_dnp_rules.json
```

The optional participation file contains `{"players":[{"player":"...",
"team":"...","played":false,"source_url":"https://www.nfl.com/...",
"published_at":"..."}]}`. The optional book-rules file maps a sportsbook key
to `{"rule":"void_if_dnp","source_url":"https://...",
"retrieved_at":"..."}` (or `action_if_dnp`). A player missing from a boxscore
is **not** assumed to have been a DNP; without affirmative participation
evidence or a book rule, its wager remains pending rather than being graded
as a loss or silently voided. Quote age and best-versus-median deviation are
retained for future quality research, never used to revise the September 13
portfolio or to filter wagers in this phase.

Phase 5 builds the first formal pregame player anytime-TD baselines from
2017-2024 historical inputs. The strict-market player comparison fits on
2022, calibrates on 2023, and validates on 2024. The 2021 market season only
warms up the Phase 4 team model, so every team expectation is scored with a
fit from earlier seasons. No 2025 holdout result or September 13, 2026
exhibition result enters Phase 5.

```powershell
./.venv/Scripts/nfl-td.exe phase5-build
./.venv/Scripts/nfl-td.exe phase5-verify
```

The validation predictions are in `reports/phase5_validation_predictions.csv`,
the analysis is in `reports/PHASE5.md`, and the local historical feature table
is `data/derived/phase5_2017_2024_player_features.parquet`. The feature table
retains nulls and audited source IDs. The player ATD market comparison uses
only the ten previously captured Phase 1 T-60 event snapshots.

Phase 6 evaluates predeclared position-specific logistic models, three small
LightGBM configurations, and fixed blends selected on 2023 out-of-fold
component predictions. The 2024 validation benchmark is the frozen Phase 5
hierarchical model; the 2025 holdout and September 13, 2026 exhibition are
excluded from development.

```powershell
./.venv/Scripts/nfl-td.exe phase6-build
./.venv/Scripts/nfl-td.exe phase6-verify
```

See `docs/PHASE6_PROTOCOL.md` for the predeclared promotion criteria,
`reports/PHASE6.md` for results, and `reports/phase6_metrics.json` for full
calibration and position tables. The 2023 OOF component and 2024 validation
prediction CSVs are retained under `reports/`.

## Phase 7 historical ATD market-edge backtest

Phase 7 freezes the Phase 5 hierarchical football model, captures 2023/2024
historical ATD Yes quotes at kickoff minus 60 minutes, and keeps pregame
predictions separate from result settlement. The complete predeclared
protocol is in [docs/PHASE7_PROTOCOL.md](docs/PHASE7_PROTOCOL.md); the
findings and limitations are in [docs/PHASE7_REPORT.md](docs/PHASE7_REPORT.md).
The 2023-only frozen rule has no qualifying candidate, so the primary 2024
benchmark is NO BET. The 2025 holdout remains sealed.

```powershell
./.venv/Scripts/nfl-td.exe phase7-verify
```

## Live cross-book ATD price scanner

Run `./.venv/Scripts/nfl-td.exe atd-price-scan` to compare today's NFL
`player_anytime_td` Yes prices across books in a single live event snapshot.
The command writes a ranked Markdown report, sortable player CSV, and
underlying per-book quote CSV under `reports/`. Consensus differences require
three distinct books; stale and unusually dispersed quotes remain visible
with flags. These are raw price differences, not bets or expected-value claims.

## Usage props MVP

The `usage_props` extension reuses `phase2_*_player_features.parquet` for
lagged targets/carries, EWMA and rolling shares; the frozen season-specific
player-stat snapshots for receptions and rushing-attempt labels; `market_math`
for American conversion and no-vig calculations; `odds.parse_time` and the
existing live Odds API event endpoint for timestamps and quotes; and the
existing Typer CLI/test setup. It does not read 2025 or alter any ATD model.

Run `./.venv/Scripts/nfl-td.exe usage-prop-validate` for the 2017–2023 fit and
2024 validation report, or `./.venv/Scripts/nfl-td.exe usage-prop-scan` for the
current receptions and rushing-attempt market report. Outputs are written as
`reports/usage_prop_opportunities_*.md`, sortable CSV, and quote CSV.
