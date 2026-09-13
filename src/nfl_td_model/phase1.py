"""Ten-game historical reconstruction with explicit unavailable fields."""

import csv
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import nflreadpy  # type: ignore[import-untyped]
import polars as pl

from nfl_td_model.audit import audit_row, prediction_timestamp, select_quote
from nfl_td_model.config import Settings
from nfl_td_model.market_math import implied_team_points
from nfl_td_model.odds import HistoricalOddsClient, extract_market_rows, parse_time
from nfl_td_model.storage import connect_catalog

LOGGER = logging.getLogger(__name__)
EASTERN = ZoneInfo("America/New_York")
FIELDS = [
    "game",
    "player",
    "player_id",
    "team",
    "opponent",
    "selection_policy",
    "kickoff_time",
    "prediction_time",
    "minutes_to_kickoff",
    "latest_player_game_used",
    "latest_team_game_used",
    "player_history_available_at",
    "team_history_available_at",
    "odds_timestamp",
    "odds_snapshot_time",
    "atd_quote_time",
    "spread_quote_time",
    "total_quote_time",
    "odds_event_id",
    "feature_available_at_max",
    "historical_carries",
    "historical_targets",
    "historical_touch_share",
    "home_spread",
    "game_total",
    "implied_team_points",
    "sportsbook",
    "market_sportsbook",
    "atd_american_odds",
    "matched_odds_player_name",
    "name_match_method",
    "result",
    "result_source",
    "status",
]


def kickoff_utc(game: dict[str, Any]) -> datetime:
    """Interpret nflverse gametime as local U.S. Eastern clock time."""
    local = datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}").replace(tzinfo=EASTERN)
    return local.astimezone(UTC)


def conservative_available_at(game: dict[str, Any]) -> datetime:
    """Use next-day availability, avoiding assumptions about exact final whistle."""
    return kickoff_utc(game) + timedelta(days=1)


def normalize_name(name: str) -> str:
    """Normalize punctuation and an optional terminal name suffix for odds matching."""
    tokens = re.findall(r"[a-z0-9]+", name.casefold())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def player_quote_candidates(
    market_rows: list[dict[str, Any]], player_name: str
) -> list[dict[str, Any]]:
    """Return unique-per-book Yes quotes for a player, rejecting ambiguous aliases."""
    candidates = [
        row
        for row in market_rows
        if row["market"] == "player_anytime_td"
        and row["name"] == "Yes"
        and normalize_name(str(row["description"] or "")) == normalize_name(player_name)
    ]
    by_book: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        by_book[row["sportsbook"]].add(str(row["description"]))
    if any(len(names) != 1 for names in by_book.values()):
        raise ValueError(f"Ambiguous ATD player alias: {player_name}")
    return candidates


def _snapshot_nfl_data(data_dir: Path, label: str, frame: pl.DataFrame) -> str:
    """Persist a content-addressed dataframe and catalog its provenance."""
    target_dir = data_dir / "raw" / "nflverse"
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = target_dir / f"{label}.temporary.parquet"
    frame.write_parquet(temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    path = target_dir / f"{label}-{digest}.parquet"
    if path.exists():
        temporary.unlink()
    else:
        temporary.replace(path)
    with connect_catalog(data_dir) as catalog:
        catalog.execute(
            "INSERT OR IGNORE INTO raw_artifacts VALUES (?, ?, ?, now(), ?)",
            [digest, "nflreadpy", str(path), f"nflreadpy:{label}"],
        )
    return digest


def _find_event(
    events: dict[str, Any], game: dict[str, Any], team_names: dict[str, str]
) -> dict[str, Any] | None:
    matches = [
        event
        for event in events.get("data", [])
        if event.get("home_team") == team_names[game["home_team"]]
        and event.get("away_team") == team_names[game["away_team"]]
        and abs((parse_time(event["commence_time"]) - kickoff_utc(game)).total_seconds()) <= 3600
    ]
    if len(matches) > 1:
        raise ValueError("Ambiguous historical event match")
    return matches[0] if matches else None


def _team_market(rows: list[dict[str, Any]], home_name: str) -> dict[str, Any] | None:
    """Find a single book with a timestamped home handicap and game total."""
    for book in sorted({row["sportsbook"] for row in rows}):
        spread = select_quote(
            [
                r
                for r in rows
                if r["sportsbook"] == book and r["market"] == "spreads" and r["name"] == home_name
            ],
            max(r["snapshot_time"] for r in rows),
        )
        total = select_quote(
            [
                r
                for r in rows
                if r["sportsbook"] == book and r["market"] == "totals" and r["name"] == "Over"
            ],
            max(r["snapshot_time"] for r in rows),
        )
        if spread and total and spread["point"] is not None and total["point"] is not None:
            return {"sportsbook": book, "spread": spread, "total": total}
    return None


def build_phase1_audit(settings: Settings, games_count: int = 10) -> tuple[Path, Path]:
    """Reconstruct features for historical-role-selected players in 2024 week four."""
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    schedules = nflreadpy.load_schedules(2024)
    player_stats = nflreadpy.load_player_stats(2024, summary_level="week")
    team_stats = nflreadpy.load_team_stats(2024, summary_level="week")
    pbp = nflreadpy.load_pbp(2024)
    teams = nflreadpy.load_teams()
    source_hashes = {
        "schedules": _snapshot_nfl_data(data_dir, "schedules-2024", schedules),
        "player_stats": _snapshot_nfl_data(data_dir, "player-stats-2024", player_stats),
        "team_stats": _snapshot_nfl_data(data_dir, "team-stats-2024", team_stats),
        "pbp": _snapshot_nfl_data(data_dir, "pbp-2024", pbp),
        "teams": _snapshot_nfl_data(data_dir, "teams", teams),
    }
    schedule_rows = schedules.to_dicts()
    game_by_id = {g["game_id"]: g for g in schedule_rows}
    players = player_stats.to_dicts()
    team_rows = team_stats.to_dicts()
    team_names = {r["team_abbr"]: r["team_name"] for r in teams.to_dicts()}
    sample = sorted(
        [
            g
            for g in schedule_rows
            if g["season"] == 2024 and g["game_type"] == "REG" and g["week"] == 4
        ],
        key=lambda g: (kickoff_utc(g), g["game_id"]),
    )[:games_count]
    if len(sample) != games_count:
        raise ValueError(f"Expected {games_count} sample games; found {len(sample)}")
    pbp_sample = pbp.filter(pl.col("game_id").is_in([game["game_id"] for game in sample]))
    scoring = pbp_sample.filter((pl.col("rush_touchdown") == 1) | (pl.col("pass_touchdown") == 1))
    if scoring.filter(pl.col("td_player_id").is_null()).height:
        raise ValueError("Scoring play without player ID in proof-of-concept sample")
    scoring_players = {
        (row["game_id"], row["td_player_id"])
        for row in scoring.select("game_id", "td_player_id").to_dicts()
    }
    for game in sample:
        plays = pbp_sample.filter(pl.col("game_id") == game["game_id"])
        if plays.is_empty() or tuple(plays.select("home_score", "away_score").tail(1).row(0)) != (
            game["home_score"],
            game["away_score"],
        ):
            raise ValueError(f"Incomplete PBP final score for {game['game_id']}")
    client = (
        HistoricalOddsClient(settings.odds_api_key, data_dir, settings.odds_regions)
        if settings.odds_api_key
        else None
    )
    audit: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    odds_hashes: dict[str, str] = {}
    for game in sample:
        kickoff = kickoff_utc(game)
        prediction = prediction_timestamp(kickoff, settings.prediction_lead_minutes)
        market_rows: list[dict[str, Any]] = []
        event_id: str | None = None
        if client:
            event = _find_event(client.events(prediction), game, team_names)
            if event:
                event_id = event["id"]
                odds_payload = client.event_odds(event_id, prediction)
                odds_hashes[event_id] = hashlib.sha256(
                    json.dumps(odds_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                market_rows = extract_market_rows(odds_payload, prediction)
            else:
                LOGGER.warning("No historical odds event for %s", game["game_id"])
        team_market = (
            _team_market(market_rows, team_names[game["home_team"]]) if market_rows else None
        )
        for team in (game["home_team"], game["away_team"]):
            opponent = game["away_team"] if team == game["home_team"] else game["home_team"]
            prior_team = sorted(
                [
                    r
                    for r in team_rows
                    if r["team"] == team
                    and r["game_id"] in game_by_id
                    and conservative_available_at(game_by_id[r["game_id"]]) < prediction
                ],
                key=lambda r: kickoff_utc(game_by_id[r["game_id"]]),
            )
            if not prior_team:
                raise ValueError(f"No prior team history: {game['game_id']} {team}")
            prior_players: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in players:
                source_game = game_by_id.get(row["game_id"])
                if (
                    row["team"] == team
                    and row["season"] == 2024
                    and row["position"] in {"QB", "RB", "WR", "TE"}
                    and source_game
                    and conservative_available_at(source_game) < prediction
                ):
                    prior_players[row["player_id"]].append(row)
            ranked_all = sorted(
                prior_players.items(),
                key=lambda item: (
                    -sum((r["carries"] or 0) + (r["targets"] or 0) for r in item[1]),
                    item[0],
                ),
            )
            if market_rows:
                ranked = []
                for candidate_id, candidate_history in ranked_all:
                    candidate_name = max(
                        candidate_history,
                        key=lambda r: kickoff_utc(game_by_id[r["game_id"]]),
                    )["player_display_name"]
                    if player_quote_candidates(market_rows, candidate_name):
                        ranked.append((candidate_id, candidate_history))
                    else:
                        exclusions.append(
                            {
                                "game": game["game_id"],
                                "player": candidate_name,
                                "player_id": candidate_id,
                                "team": team,
                                "reason": "NO_ATD_QUOTE_AT_PREDICTION_TIME",
                                "prior_carries_plus_targets": sum(
                                    (r["carries"] or 0) + (r["targets"] or 0)
                                    for r in candidate_history
                                ),
                            }
                        )
                    if len(ranked) == 2:
                        break
            else:
                ranked = ranked_all[:2]
            if len(ranked) != 2:
                raise ValueError(
                    f"Insufficient quote-eligible historical players: {game['game_id']} {team}"
                )
            for player_id, all_history in ranked:
                recent = sorted(
                    all_history,
                    key=lambda r: kickoff_utc(game_by_id[r["game_id"]]),
                )[-3:]
                latest_player = conservative_available_at(game_by_id[recent[-1]["game_id"]])
                latest_team = conservative_available_at(game_by_id[prior_team[-1]["game_id"]])
                latest_player_kickoff = kickoff_utc(game_by_id[recent[-1]["game_id"]])
                latest_team_kickoff = kickoff_utc(game_by_id[prior_team[-1]["game_id"]])
                carries = sum(r["carries"] or 0 for r in recent)
                targets = sum(r["targets"] or 0 for r in recent)
                recent_game_ids = {r["game_id"] for r in recent}
                team_opportunities = sum(
                    (r["carries"] or 0) + (r["targets"] or 0)
                    for r in prior_team
                    if r["game_id"] in recent_game_ids
                )
                touch_share = (
                    (carries + targets) / team_opportunities if team_opportunities else None
                )
                name = recent[-1]["player_display_name"]
                quote = select_quote(
                    player_quote_candidates(market_rows, name),
                    prediction,
                )
                result_row = next(
                    (
                        r
                        for r in players
                        if r["game_id"] == game["game_id"] and r["player_id"] == player_id
                    ),
                    None,
                )
                result = int((game["game_id"], player_id) in scoring_players)
                if result_row is not None:
                    weekly_result = int(
                        (result_row["rushing_tds"] or 0) + (result_row["receiving_tds"] or 0) > 0
                    )
                    if weekly_result != result:
                        raise ValueError(
                            f"Weekly stats and PBP scoring disagree for {game['game_id']} {player_id}"
                        )
                result_source = "pbp+weekly_stats" if result_row is not None else "pbp_only"
                market_ok = quote is not None and team_market is not None
                odds_timestamp = (
                    max(
                        quote["quote_time"],
                        team_market["spread"]["quote_time"],
                        team_market["total"]["quote_time"],
                    )
                    if quote is not None and team_market is not None
                    else None
                )
                if odds_timestamp is not None:
                    max_available = audit_row(
                        kickoff_time=kickoff,
                        prediction_time=prediction,
                        latest_player_game_used=latest_player,
                        latest_team_game_used=latest_team,
                        odds_timestamp=odds_timestamp,
                        odds_snapshot_time=market_rows[0]["snapshot_time"],
                    )
                else:
                    max_available = max(latest_player, latest_team)
                    if max_available >= prediction:
                        raise ValueError("Historical feature leakage")
                home_points, away_points = (
                    implied_team_points(
                        team_market["total"]["point"], team_market["spread"]["point"]
                    )
                    if team_market
                    else (None, None)
                )
                audit.append(
                    {
                        "game": game["game_id"],
                        "player": name,
                        "player_id": player_id,
                        "team": team,
                        "opponent": opponent,
                        "selection_policy": (
                            "top_prior_usage_with_ATD_quote"
                            if market_rows
                            else "diagnostic_top_prior_usage_no_market"
                        ),
                        "kickoff_time": kickoff.isoformat(),
                        "prediction_time": prediction.isoformat(),
                        "minutes_to_kickoff": settings.prediction_lead_minutes,
                        "latest_player_game_used": latest_player_kickoff.isoformat(),
                        "latest_team_game_used": latest_team_kickoff.isoformat(),
                        "player_history_available_at": latest_player.isoformat(),
                        "team_history_available_at": latest_team.isoformat(),
                        "odds_timestamp": odds_timestamp.isoformat() if odds_timestamp else None,
                        "odds_snapshot_time": (
                            market_rows[0]["snapshot_time"].isoformat() if market_rows else None
                        ),
                        "atd_quote_time": quote["quote_time"].isoformat() if quote else None,
                        "spread_quote_time": (
                            team_market["spread"]["quote_time"].isoformat() if team_market else None
                        ),
                        "total_quote_time": (
                            team_market["total"]["quote_time"].isoformat() if team_market else None
                        ),
                        "odds_event_id": event_id,
                        "feature_available_at_max": max_available.isoformat(),
                        "historical_carries": carries,
                        "historical_targets": targets,
                        "historical_touch_share": touch_share,
                        "home_spread": team_market["spread"]["point"] if team_market else None,
                        "game_total": team_market["total"]["point"] if team_market else None,
                        "implied_team_points": (
                            home_points if team == game["home_team"] else away_points
                        ),
                        "sportsbook": quote["sportsbook"] if quote else None,
                        "market_sportsbook": (team_market["sportsbook"] if team_market else None),
                        "atd_american_odds": quote["price"] if quote else None,
                        "matched_odds_player_name": quote["description"] if quote else None,
                        "name_match_method": (
                            "exact"
                            if quote and str(quote["description"]).casefold() == name.casefold()
                            else "terminal_suffix_alias"
                            if quote
                            else None
                        ),
                        "result": result,
                        "result_source": result_source,
                        "status": ("PASS" if market_ok else "BLOCKED_MISSING_ODDS"),
                    }
                )
                histories.append(
                    {
                        "game": game["game_id"],
                        "player": name,
                        "player_id": player_id,
                        "prediction_time": prediction.isoformat(),
                        "source_games": [
                            {
                                "game_id": r["game_id"],
                                "kickoff": kickoff_utc(game_by_id[r["game_id"]]).isoformat(),
                                "available_at_proxy": conservative_available_at(
                                    game_by_id[r["game_id"]]
                                ).isoformat(),
                                "carries": r["carries"],
                                "targets": r["targets"],
                            }
                            for r in recent
                        ],
                        "team_source_games": [
                            {
                                "game_id": r["game_id"],
                                "kickoff": kickoff_utc(game_by_id[r["game_id"]]).isoformat(),
                                "available_at_proxy": conservative_available_at(
                                    game_by_id[r["game_id"]]
                                ).isoformat(),
                                "carries": r["carries"],
                                "targets": r["targets"],
                            }
                            for r in prior_team
                            if r["game_id"] in recent_game_ids
                        ],
                    }
                )
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    audit_path = reports / "phase1_2024_week4_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    with (reports / "phase1_market_exclusions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game",
                "player",
                "player_id",
                "team",
                "reason",
                "prior_carries_plus_targets",
            ],
        )
        writer.writeheader()
        writer.writerows(exclusions)
    history_path = reports / "phase1_representative_histories.json"
    history_path.write_text(json.dumps(histories[:6], indent=2), encoding="utf-8")
    (reports / "phase1_full_histories.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    (reports / "phase1_source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2), encoding="utf-8"
    )
    (reports / "phase1_odds_hashes.json").write_text(
        json.dumps(odds_hashes, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Audit: %s games, %s rows, %s passing",
        games_count,
        len(audit),
        sum(row["status"] == "PASS" for row in audit),
    )
    return audit_path, history_path
