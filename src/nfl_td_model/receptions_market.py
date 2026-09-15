"""Pure normalization helpers for immutable historical receptions quotes.

The downloader owns retrieval and caching; this module deliberately accepts a
payload and game metadata so normalization can be tested without network or
model inputs.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from nfl_td_model.market_math import american_to_decimal, raw_implied_probability
from nfl_td_model.odds import extract_market_rows
from nfl_td_model.phase1 import normalize_name
from nfl_td_model.usage_props import two_sided_de_vig


def _match(name: str, team: str | None, players: dict[str, list[dict[str, Any]]],
           allowed_teams: set[str] | None = None) -> tuple[str | None, str]:
    candidates = players.get(normalize_name(name), [])
    if team:
        candidates = [p for p in candidates if p.get("team") in {team, None}]
    elif allowed_teams:
        candidates = [p for p in candidates if p.get("team") in allowed_teams]
    ids = {str(p["player_id"]) for p in candidates if p.get("player_id") is not None}
    if len(ids) == 1:
        return next(iter(ids)), "MATCHED"
    return None, "AMBIGUOUS" if len(ids) > 1 else "UNMATCHED"


def normalize_quotes(
    payload: dict[str, Any],
    game: dict[str, Any],
    players: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten and conservatively match one cached player_receptions payload."""
    players = players or {}
    prediction_time = datetime.fromisoformat(str(game["prediction_time"]))
    result: list[dict[str, Any]] = []
    for row in extract_market_rows(payload, prediction_time):
        if row["market"] != "player_receptions" or row["name"] not in {"Over", "Under"}:
            continue
        name = str(row["description"] or "")
        allowed_teams = {str(game.get("home_team")), str(game.get("away_team"))} - {"None"}
        player_id, status = _match(name, game.get("team"), players, allowed_teams)
        price = int(row["price"])
        result.append({
            "event_id": game.get("event_id"), "game_id": game.get("game_id"),
            "season": game.get("season"), "week": game.get("week"),
            "kickoff_time": game.get("kickoff_time"), "prediction_time": game["prediction_time"],
            "snapshot_time": row["snapshot_time"].isoformat(), "quote_time": row["quote_time"].isoformat(),
            "sportsbook": row["sportsbook"], "player": name, "player_id": player_id,
            "match_status": status, "line": float(row["point"]), "side": row["name"],
            "american_odds": price, "decimal_odds": american_to_decimal(price),
            "raw_implied_probability": raw_implied_probability(price),
        })
    return result


def pair_quotes(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair Over/Under only within the same book, player and exact line."""
    grouped: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["sportsbook"]), normalize_name(str(row["player"])), float(row["line"]))][str(row["side"])] = row
    paired: list[dict[str, Any]] = []
    for sides in grouped.values():
        if "Over" not in sides or "Under" not in sides:
            continue
        over, under = sides["Over"], sides["Under"]
        raw_over, raw_under, fair_over = two_sided_de_vig(int(over["american_odds"]), int(under["american_odds"]))
        paired.append({**over, "under_american_odds": under["american_odds"],
                       "under_decimal_odds": under["decimal_odds"], "raw_implied_under_probability": raw_under,
                       "raw_implied_over_probability": raw_over, "hold": raw_over + raw_under - 1,
                       "no_vig_over_probability": fair_over,
                       "no_vig_under_probability": 1 - fair_over})
    return paired


def consensus(paired: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize identical player/line pairs across sportsbooks."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(normalize_name(str(row["player"])), float(row["line"]))].append(row)
    out: list[dict[str, Any]] = []
    for (player, line), rows in groups.items():
        over = [int(r["american_odds"]) for r in rows]
        under = [int(r["under_american_odds"]) for r in rows]
        out.append({"player": rows[0]["player"], "player_id": rows[0].get("player_id"), "line": line,
                    "books_quoting": len(rows), "best_over_american_odds": max(over, key=american_to_decimal),
                    "best_under_american_odds": max(under, key=american_to_decimal),
                    "median_over_american_odds": statistics.median(over),
                    "median_under_american_odds": statistics.median(under),
                    "median_no_vig_over_probability": statistics.median(r["no_vig_over_probability"] for r in rows)})
    return out
