# Phase 1 audit — 2024 Week 4

**Mechanical acceptance gate: passed.** The first ten regular-season Week 4 games,
ordered by kickoff, contain 40 audited players (two per team), and all 40 rows pass
`nfl-td phase1-verify`. No model training, betting simulation, or 2025 holdout analysis
has begun.

The [audit table](phase1_2024_week4_audit.csv) records kickoff, kickoff-minus-60-minute
prediction time, latest player and team games used, assumed history availability,
historical event and quote timestamps, usage features, the ATD price, and the settled
result. [Full feature histories](phase1_full_histories.json) identify every prior game
in each player numerator and team denominator; [representative histories](phase1_representative_histories.json)
provide six compact examples. Frozen local source artifacts are identified by the
[nflverse hashes](phase1_source_hashes.json) and [Odds API response hashes](phase1_odds_hashes.json).
The verifier checks these hashes, recomputes usage and settlement labels from the
frozen source files, and reconciles each recorded ATD, spread, total, event ID, and
kickoff against the saved API response. All ten Odds API event kickoffs equal the
corresponding audit kickoff exactly.

Players were ranked by carries plus targets in earlier 2024 games with their current
team, then the top two with an ATD Yes quote available by prediction time were selected
per team. This uses quote *availability*, not its price or the game result. The
[exclusion log](phase1_market_exclusions.csv) records one higher-usage player, Cooper
Kupp, who had no qualifying ATD quote; Demarcus Robinson was selected next. Name
matching was exact for 37 rows and omitted only a terminal name suffix for three:
Travis Etienne/Jr., Michael Pittman/Jr., and Deebo Samuel Sr./no suffix. The matched
API spelling and matching method are recorded per row. The ATD market updates precede
prediction by 3.83–4.35 minutes across these rows.

Historical Odds API event responses were requested at the prediction time and retained
by content hash. The API wrapper snapshot and each relevant market `last_update` are
separately required to be no later than prediction. In these responses, market update
times can exceed the wrapper timestamp by seconds, so the wrapper is not treated as
an upper bound on each market update. The maximum of all relevant timestamps is the
row's `feature_available_at_max`. The selected home spread and game total come from
one sportsbook; the home team's implied points use `(total-home_spread)/2`. The
untimestamped nflverse schedule lines are never market features.

Football history uses a declared availability proxy: one full day after each prior
game's scheduled kickoff. This excludes current and same-day games, but **it is not a
verified source-publication timestamp**. Current nflverse files may contain later
corrections; their hashes make this reconstruction reproducible, not a proof that the
same file version existed in September 2024. Likewise, today's schedule file cannot
prove that an old kickoff time was published unchanged then. Archived source versions
and actual publication metadata are needed before treating this as a fully evidenced
historical backtest. The historical quote may also have belonged to a player who was
inactive, because lineup status was not used as a Phase 1 feature or selection filter.

PBP is used only for settlement labels. Ten PBP final scores match the schedule, and
weekly stats corroborate 39 selected-player labels. One selected player, Joe Mixon,
has no Week 4 weekly-stat row; his zero label is derived from complete game PBP and
marked `pbp_only`. The status above means the implemented Phase 1 gate passes under
the explicit football-availability assumption; it does not erase that remaining
point-in-time source-version limitation. Phase 2 remains stopped.

## Source contract and references

The inspected nflreadpy 0.1.5 tables supply schedule game IDs and Eastern local
`gameday`/`gametime`, weekly player and team carries/targets, and PBP touchdown player
IDs and final scores. The Odds API historical single-event endpoint supplies event
`commence_time`, bookmaker markets, outcome descriptions, and market update times.
The 2025 season remains reserved as a final holdout.

- [nflreadpy load functions](https://nflreadpy.nflverse.com/api/load_functions/)
- [The Odds API historical-event documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [The Odds API betting markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)

Public example repositories inspected for research direction, with no code reused:
[ffverse/ffopportunity](https://github.com/ffverse/ffopportunity),
[SRock44/sports-prediction-model](https://github.com/SRock44/sports-prediction-model),
[aaronlaporte/sports-analytics](https://github.com/aaronlaporte/sports-analytics),
[gesmith0606/nfl_data_engineering](https://github.com/gesmith0606/nfl_data_engineering),
and [gmalbert/nfl-predictions](https://github.com/gmalbert/nfl-predictions).
