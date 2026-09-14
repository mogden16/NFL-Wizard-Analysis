"""Strict T-minus-60 historical spread/total reconstruction."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.market_math import implied_team_points
from nfl_td_model.odds import HistoricalOddsClient, parse_time
from nfl_td_model.phase1 import kickoff_utc

LOGGER = logging.getLogger(__name__)
EASTERN = ZoneInfo("America/New_York")
MARKET_SEASONS = range(2021, 2025)
BOOK_PRIORITY = ("fanduel", "draftkings", "betmgm", "caesars", "betrivers", "bovada")


def market_cutoff(kickoff: datetime) -> datetime:
    return kickoff - timedelta(minutes=60)


def quote_is_eligible(
    snapshot_time: datetime, spread_time: datetime, total_time: datetime, cutoff: datetime
) -> bool:
    """Require the wrapper and both actual market updates to precede T-60."""
    return max(snapshot_time, spread_time, total_time) <= cutoff


def normalize_home_spread(outcome_point: float, *, convention: str) -> float:
    """Convert either home handicap or expected home margin to sportsbook handicap."""
    if convention == "home_handicap":
        return outcome_point
    if convention == "home_margin":
        return -outcome_point
    raise ValueError("Unknown sportsbook spread convention")


def _schedule_games(data_dir: Path) -> list[dict[str, Any]]:
    hashes = json.loads((Path("reports") / "phase2_historical_source_hashes.json").read_text())
    games: list[dict[str, Any]] = []
    for season in MARKET_SEASONS:
        path = (
            data_dir
            / "raw"
            / "nflverse"
            / f"schedules-{season}-{hashes[str(season)]['schedules']}.parquet"
        )
        frame = pl.read_parquet(path)
        games.extend(
            row for row in frame.to_dicts() if row["season"] == season and row["game_type"] == "REG"
        )
    return sorted(games, key=kickoff_utc)


def snapshot_dates(games: list[dict[str, Any]]) -> list[datetime]:
    """One at the earliest T-60 cutoff per Eastern calendar day with games."""
    by_day: dict[str, list[datetime]] = defaultdict(list)
    for game in games:
        kickoff = kickoff_utc(game)
        by_day[kickoff.astimezone(EASTERN).date().isoformat()].append(market_cutoff(kickoff))
    return sorted(min(cutoffs) for cutoffs in by_day.values())


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def collect_market_snapshots(settings: Settings) -> Path:
    """Cache historical featured-market snapshots without persisting the credential."""
    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required for strict historical market reconstruction")
    games = _schedule_games(settings.data_dir)
    dates = snapshot_dates(games)
    if any(game["season"] >= 2025 for game in games):
        raise ValueError("Protected 2025 holdout entered market collection")
    LOGGER.info("Collecting %s daily pregame snapshots for 2021–2024", len(dates))
    client = HistoricalOddsClient(settings.odds_api_key, settings.data_dir, settings.odds_regions)
    records = []
    for index, cutoff in enumerate(dates, 1):
        payload = client.featured_odds(cutoff)
        records.append(
            {
                "requested_at": cutoff.isoformat(),
                "snapshot_time": payload["timestamp"],
                "payload_sha256": _payload_hash(payload),
                "events": len(payload.get("data") or []),
            }
        )
        if index % 10 == 0 or index == len(dates):
            LOGGER.info("Cached %s/%s market snapshots", index, len(dates))
    output = Path("reports") / "phase4_market_snapshot_manifest.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output


def _team_names(data_dir: Path) -> dict[tuple[int, str], str]:
    hashes = json.loads((Path("reports") / "phase1_source_hashes.json").read_text())
    frame = pl.read_parquet(data_dir / "raw" / "nflverse" / f"teams-{hashes['teams']}.parquet")
    names = {
        (season, row["team_abbr"]): row["team_name"]
        for season in MARKET_SEASONS
        for row in frame.to_dicts()
    }
    names[(2021, "WAS")] = "Washington Football Team"
    return names


def _event_quotes(
    event: dict[str, Any], snapshot_time: datetime, home_name: str, cutoff: datetime
) -> list[dict[str, Any]]:
    quotes = []
    for book in event.get("bookmakers") or []:
        spread = next((m for m in book.get("markets") or [] if m.get("key") == "spreads"), None)
        total = next((m for m in book.get("markets") or [] if m.get("key") == "totals"), None)
        if not spread or not total:
            continue
        spread_time = parse_time(spread["last_update"])
        total_time = parse_time(total["last_update"])
        if not quote_is_eligible(snapshot_time, spread_time, total_time, cutoff):
            continue
        home = next((o for o in spread["outcomes"] if o.get("name") == home_name), None)
        over = next((o for o in total["outcomes"] if o.get("name") == "Over"), None)
        if not home or not over or home.get("point") is None or over.get("point") is None:
            continue
        quotes.append(
            {
                "sportsbook": book["key"],
                "home_spread": normalize_home_spread(
                    float(home["point"]), convention="home_handicap"
                ),
                "game_total": float(over["point"]),
                "spread_quote_time": spread_time,
                "total_quote_time": total_time,
                "quote_time": max(spread_time, total_time),
                "older_market_time": min(spread_time, total_time),
                "snapshot_time": snapshot_time,
                "event_id": event["id"],
            }
        )
    return quotes


def build_strict_market_table(settings: Settings) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Select the freshest eligible same-book spread/total package per game."""
    games = _schedule_games(settings.data_dir)
    names = _team_names(settings.data_dir)
    manifest = json.loads((Path("reports") / "phase4_market_snapshot_manifest.json").read_text())
    client = HistoricalOddsClient(
        settings.odds_api_key or "", settings.data_dir, settings.odds_regions
    )
    snapshots: list[tuple[datetime, dict[tuple[str, str], list[dict[str, Any]]]]] = []
    for record in manifest:
        cutoff = parse_time(record["requested_at"])
        payload = client.featured_odds(cutoff)
        if _payload_hash(payload) != record["payload_sha256"]:
            raise ValueError("Historical market cache changed since collection")
        snapshot_time = parse_time(payload["timestamp"])
        events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in payload.get("data") or []:
            events[(event["home_team"], event["away_team"])].append(event)
        snapshots.append((snapshot_time, dict(events)))
    rows = []
    missing: dict[str, int] = defaultdict(int)
    for game in games:
        kickoff = kickoff_utc(game)
        cutoff = market_cutoff(kickoff)
        home_name = names.get((game["season"], game["home_team"]))
        away_name = names.get((game["season"], game["away_team"]))
        if not home_name or not away_name:
            missing["team_name"] += 1
            continue
        candidates = []
        for snapshot_time, events in snapshots:
            if snapshot_time > cutoff:
                continue
            for event in events.get((home_name, away_name), []):
                if abs((parse_time(event["commence_time"]) - kickoff).total_seconds()) > 3 * 3600:
                    continue
                candidates.extend(_event_quotes(event, snapshot_time, home_name, cutoff))
        if not candidates:
            missing["no_eligible_quote"] += 1
            continue
        quote = max(
            candidates,
            key=lambda item: (
                item["older_market_time"],
                item["quote_time"],
                -BOOK_PRIORITY.index(item["sportsbook"])
                if item["sportsbook"] in BOOK_PRIORITY
                else -len(BOOK_PRIORITY),
            ),
        )
        home_points, away_points = implied_team_points(quote["game_total"], quote["home_spread"])
        for home in (True, False):
            rows.append(
                {
                    "season": game["season"],
                    "week": game["week"],
                    "game_id": game["game_id"],
                    "team": game["home_team"] if home else game["away_team"],
                    "opponent": game["away_team"] if home else game["home_team"],
                    "home": int(home),
                    "kickoff": kickoff,
                    "prediction_time": cutoff,
                    "sportsbook": quote["sportsbook"],
                    "event_id": quote["event_id"],
                    "odds_timestamp": quote["quote_time"],
                    "spread_quote_time": quote["spread_quote_time"],
                    "total_quote_time": quote["total_quote_time"],
                    "odds_snapshot_time": quote["snapshot_time"],
                    "home_spread": quote["home_spread"],
                    "home_margin": -quote["home_spread"],
                    "team_spread": quote["home_spread"] if home else -quote["home_spread"],
                    "game_total": quote["game_total"],
                    "implied_team_points": home_points if home else away_points,
                }
            )
    table = pl.DataFrame(rows).sort("season", "week", "game_id", "team")
    if table.filter(pl.col("odds_timestamp") > pl.col("prediction_time")).height:
        raise ValueError("Future odds quote entered strict market table")
    coverage = {
        "snapshot_count": len(snapshots),
        "team_games": table.height,
        "games": table["game_id"].n_unique(),
        "by_season": {
            str(season): {
                "team_games": table.filter(pl.col("season") == season).height,
                "games": table.filter(pl.col("season") == season)["game_id"].n_unique(),
            }
            for season in MARKET_SEASONS
        },
        "missing_games": dict(missing),
    }
    return table, coverage
