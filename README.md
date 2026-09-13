# nfl_td_model

Phase 0 and Phase 1 point-in-time proof of concept for NFL anytime-touchdown research. No
probability models or betting recommendations are implemented. The ten-game audit uses
paid historical quotes and fails closed when a required market timestamp is missing.

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
