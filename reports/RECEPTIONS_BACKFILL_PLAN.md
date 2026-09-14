# Receptions Backfill Step 1: Plan and Cost Audit

**Scope:** planning only. No new Odds API requests were made, no models were
modified, and the 2025 holdout was not inspected.

## Request contract

- Sport: `americanfootball_nfl`
- Target market: `player_receptions`
- Historical event-ID endpoint:
  `GET /v4/historical/sports/americanfootball_nfl/events?date=...`
- Historical player-prop endpoint:
  `GET /v4/historical/sports/americanfootball_nfl/events/{event_id}/odds`
- Parameters for each prop request: `date=kickoff-60m`, `regions=us`,
  `markets=player_receptions`, `oddsFormat=american`.
- The API returns the closest snapshot at or before the requested date; the
  response and each market `last_update` must still pass the existing T-60
  cutoff check.

Player props are supported historically from May 3, 2023, so both target
seasons are within the documented availability period. The current client
already uses the same base URL, timestamp validation, and immutable cache; a
backfill call must request only `player_receptions` rather than the current
multi-market ATD bundle.

## Games and calls

| Season | Regular-season games | Event IDs cached at/before T-60 | IDs needing lookup | Player-receptions odds calls |
|---|---:|---:|---:|---:|
| 2023 | 272 | 259 | 13 | 272 |
| 2024 | 272 | 267 | 5 | 272 |
| **Total** | **544** | **526** | **18** | **544** |

The event-ID endpoint can be called only for the 18 missing T-60 lookups; the
526 locally recoverable IDs should be reused. Historical event-ID lookups are
metadata requests; the documented historical odds charge is applied to the
event-odds requests.

## Credits and cache state

The Odds API documents historical event-odds cost as 10 credits per region per
market per event. With one US region and one market, the worst-case target
backfill is **544 calls × 10 = 5,440 credits**. Empty historical responses do
not count, but the plan budgets for the full amount. Existing raw odds files
contain ATD, spread, and total markets only; **0** `player_receptions` target
responses and **0** target calls are already cached. Therefore remaining
target calls are **544**.

Each response will continue to be written content-addressably under
`data/raw/the_odds_api/{sha256}.json`, with the request hash indexed under
`data/raw/odds_request_index/{request_sha256}.txt` and cataloged in DuckDB.
The index makes an identical endpoint/date/parameter request a cache hit, and
the content hash prevents mutation of an existing raw artifact.

## Concerns

1. The existing `event_odds` helper requests additional markets; the backfill
   must use a target-specific request to avoid paying for unnecessary markets.
2. Event IDs are Odds API identifiers and cannot be derived from nflverse
   `game_id`; the 18 missing IDs require timestamped historical event lookups
   before odds calls.
3. Player-prop availability can be sparse by bookmaker and game. Missing or
   empty target responses must remain explicit and must not be fabricated.
4. The current repository has no target-market historical responses, so no
   market-matched validation can be performed until the backfill is authorized.

The credit formula and historical endpoint behavior are documented by [The
Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/) and its
[historical odds data reference](https://the-odds-api.com/historical-odds-data/).
