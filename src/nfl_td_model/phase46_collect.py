"""Pregame-only collector for prospective 2026 event snapshots.

This process sees only live event metadata, T-60 market quotes, frozen prior
football sources, and an optional contemporaneous official availability file.
It never opens a results, current-game PBP, or settlement source.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.market_math import implied_team_points
from nfl_td_model.odds import parse_time
from nfl_td_model.phase1 import _team_market
from nfl_td_model.phase4_models import MARKET, fit_count_model
from nfl_td_model.phase45 import _historical_players, allocation_score

LIVE_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MODEL_VERSION = (
    "phase4-complete:A_market_only_poisson + "
    "phase3-complete:2024_logistic_inference_2025 + phase45_fixed_allocation_v1"
)


def _client(settings: Settings) -> httpx.Client:
    if not settings.odds_api_key:
        raise ValueError("The Odds API key is required for prospective capture")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return httpx.Client(timeout=30)


def discover_events(settings: Settings) -> list[dict[str, str]]:
    """The live /events endpoint supplies schedule metadata without odds or results."""
    with _client(settings) as client:
        response = client.get(f"{LIVE_BASE}/events", params={"apiKey": settings.odds_api_key})
        response.raise_for_status()
        events = response.json()
    return [
        {"event_id": e["id"], "kickoff": e["commence_time"],
         "home": e["home_team"], "away": e["away_team"]}
        for e in events if parse_time(e["commence_time"]).year == 2026
        and parse_time(e["commence_time"]).astimezone(ZoneInfo("America/New_York")).date().isoformat() > "2026-09-13"
    ]


def _live_quote_rows(payload: dict[str, Any], cutoff: datetime) -> list[dict[str, Any]]:
    """Project the live response to the same pre-cutoff market schema as Phase 4.5."""
    rows = []
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            updated = parse_time(market["last_update"])
            if updated > cutoff:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "sportsbook": book["key"], "market": market["key"],
                    "quote_time": updated, "snapshot_time": cutoff, "name": outcome.get("name"),
                    "description": outcome.get("description"), "price": outcome.get("price"),
                    "point": outcome.get("point"),
                })
    return rows


def _availability_file(path: Path | None, cutoff: datetime, event_id: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"availability": [], "availability_coverage": []}
    source = json.loads(path.read_text(encoding="utf-8"))
    if set(source) != {"event_id", "availability", "availability_coverage"}:
        raise ValueError("Availability input has unknown fields")
    if source["event_id"] != event_id:
        raise ValueError("Availability event ID mismatch")
    for group in ("availability", "availability_coverage"):
        for record in source[group]:
            if parse_time(record["retrieved_at"]) > cutoff:
                raise ValueError("Availability was retrieved after T-60")
    return {"availability": source["availability"],
            "availability_coverage": source["availability_coverage"]}


def capture_event(
    settings: Settings, event: dict[str, str], root: Path,
    availability_file: Path | None = None, now: datetime | None = None,
) -> Path:
    """Fetch live quotes once, filter by market update time, then seal pregame JSON."""
    kickoff = parse_time(event["kickoff"])
    cutoff = kickoff - timedelta(minutes=60)
    captured = now or datetime.now(UTC)
    if kickoff.astimezone(ZoneInfo("America/New_York")).date().isoformat() <= "2026-09-13":
        raise ValueError("September 13 exhibition is immutable")
    if captured < cutoff - timedelta(minutes=2) or captured > cutoff + timedelta(seconds=45):
        raise ValueError("Prospective capture must run at T-60 (45-second tolerance)")
    output_dir = root / cutoff.date().isoformat() / event["event_id"]
    input_file = output_dir / "pregame.json"
    if output_dir.exists():
        raise FileExistsError("Prospective event already captured")
    with _client(settings) as client:
        response = client.get(
            f"{LIVE_BASE}/events/{event['event_id']}/odds",
            params={"apiKey": settings.odds_api_key, "regions": settings.odds_regions,
                    "markets": "player_anytime_td,spreads,totals", "oddsFormat": "american"},
        )
        response.raise_for_status()
        payload = response.json()
    captured = now or datetime.now(UTC)
    if payload["id"] != event["event_id"] or parse_time(payload["commence_time"]) != kickoff:
        raise ValueError("Live market event or kickoff changed")
    market_rows = _live_quote_rows(payload, cutoff)
    market = _team_market(market_rows, event["home"])
    if market is None:
        raise ValueError("No T-60 spread/total pair")
    quotes = [
        {"player": str(r["description"]), "sportsbook": r["sportsbook"],
         "price": int(r["price"]), "quote_time": r["quote_time"].isoformat(),
         "quote_age_seconds": (cutoff - r["quote_time"]).total_seconds(),
         "market": r["market"], "name": r["name"]}
        for r in market_rows if r["market"] == "player_anytime_td" and r["name"] == "Yes"
    ]
    if not quotes:
        raise ValueError("No ATD quotes available at T-60")
    teams_file = next((settings.data_dir / "raw/nflverse").glob("teams-*.parquet"))
    names = {r["team_name"]: r["team_abbr"] for r in pl.read_parquet(teams_file).to_dicts()}
    home, away = names[event["home"]], names[event["away"]]
    total = float(market["total"]["point"])
    spread = float(market["spread"]["point"])
    home_points, away_points = implied_team_points(total, spread)
    context = pl.DataFrame([
        {"implied_team_points": home_points, "team_spread": spread, "game_total": total, "home": 1},
        {"implied_team_points": away_points, "team_spread": -spread, "game_total": total, "home": 0},
    ])
    model_data = pl.read_parquet("data/derived/phase4_strict_market_2021_2024.parquet")
    model = fit_count_model(model_data.filter(pl.col("season") <= 2023), MARKET, "poisson")
    team_td = {home: float(model.predict(context)[0]), away: float(model.predict(context)[1])}
    history = _historical_players(settings)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in history.values():
        if player["team"] in {home, away}:
            grouped[player["team"]].append(player)
    players = []
    for team_players in grouped.values():
        for player in team_players:
            players.append({
                "player": player["player"], "player_id": player["player_id"],
                "position": player["position"], "team": player["team"],
                "allocation_score": allocation_score(player, team_players),
                "history_games": player["history_games"],
                "latest_prior_available_at": player["latest_prior_available_at"].isoformat(),
                **{key: player.get(key) for key in (
                    "total_xtd_share", "rushing_xtd_share", "receiving_xtd_share",
                    "goal_line_opportunities_share", "inside_5_carries_per_game",
                    "inside_10_carries_per_game", "red_zone_targets_share", "end_zone_targets_share",
                    "carry_share", "target_share", "snap_share",
                )},
            })
    availability = _availability_file(availability_file, cutoff, event["event_id"])
    snapshot = {
        "schema": "phase46-pregame-v1", "event_id": event["event_id"],
        "kickoff": kickoff.isoformat(), "prediction_time": cutoff.isoformat(),
        "captured_at": captured.isoformat(), "home": home, "away": away,
        "team_expected_td": team_td, "players": players, "quotes": quotes,
        **availability, "model_version": MODEL_VERSION,
        "diagnostic_exhibition_slate": False,
        "quote_snapshot_sha256": hashlib.sha256(json.dumps(quotes, sort_keys=True).encode()).hexdigest(),
    }
    # The worker's strict schema check runs before writing anything.
    from nfl_td_model.phase46_predict import validate_snapshot

    validate_snapshot(snapshot)
    output_dir.mkdir(parents=True)
    input_file.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
    return input_file
