"""Play-level rushing and receiving opportunity extraction without player-ID features."""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.phase1 import conservative_available_at, kickoff_utc

RUSH_NUMERIC = (
    "yardline_100",
    "log_yardline",
    "goal_to_go",
    "down",
    "ydstogo",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "shotgun",
    "no_huddle",
    "qb_scramble",
    "inside_5",
    "inside_10",
    "inside_20",
)
RUSH_CATEGORICAL = ("run_location", "run_gap", "position")
RECEIVING_NUMERIC = (
    "yardline_100",
    "log_yardline",
    "air_yards",
    "relative_to_endzone",
    "end_zone_target",
    "down",
    "ydstogo",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "shotgun",
    "no_huddle",
    "inside_5",
    "inside_10",
    "inside_20",
)
RECEIVING_CATEGORICAL = ("pass_location", "position")
PBP_COLUMNS = (
    "game_id",
    "play_id",
    "posteam",
    "rush_attempt",
    "pass_attempt",
    "rush_touchdown",
    "pass_touchdown",
    "td_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "lateral_rusher_player_id",
    "lateral_receiver_player_id",
    "two_point_attempt",
    "play_deleted",
    "qb_kneel",
    "qb_spike",
    "play_type",
    "yardline_100",
    "goal_to_go",
    "down",
    "ydstogo",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "shotgun",
    "no_huddle",
    "qb_scramble",
    "run_location",
    "run_gap",
    "pass_location",
    "air_yards",
)


def classify_opportunity(play: dict[str, Any]) -> tuple[str, str, int] | None:
    """Return type, primary player, TD label; drop kneels, two-pointers and laterals."""
    if play.get("two_point_attempt") == 1 or play.get("play_deleted") == 1:
        return None
    if play.get("lateral_rusher_player_id") or play.get("lateral_receiver_player_id"):
        return None
    if play.get("rush_attempt") == 1 and play.get("rusher_player_id"):
        if play.get("qb_kneel") == 1 or play.get("play_type") != "run":
            return None
        player_id = str(play["rusher_player_id"])
        if play.get("rush_touchdown") == 1 and play.get("td_player_id") != player_id:
            return None
        return "rushing", player_id, int(play.get("rush_touchdown") == 1)
    if play.get("pass_attempt") == 1 and play.get("receiver_player_id"):
        if play.get("qb_spike") == 1 or play.get("play_type") != "pass":
            return None
        player_id = str(play["receiver_player_id"])
        if play.get("pass_touchdown") == 1 and play.get("td_player_id") != player_id:
            return None
        return "receiving", player_id, int(play.get("pass_touchdown") == 1)
    return None


def relative_to_endzone(yardline_100: float | None, air_yards: float | None) -> float | None:
    """Air distance beyond the goal line; zero denotes a target at the goal line."""
    return air_yards - yardline_100 if air_yards is not None and yardline_100 is not None else None


def end_zone_target(yardline_100: float | None, air_yards: float | None) -> int | None:
    relative = relative_to_endzone(yardline_100, air_yards)
    return int(relative >= 0) if relative is not None else None


def _position_history(
    stats: pl.DataFrame, games: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], list[tuple[datetime, str]]]:
    history: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
    for row in stats.select("game_id", "player_id", "team", "position").iter_rows(named=True):
        game = games.get(row["game_id"])
        if game and row["position"] in {"RB", "WR", "TE", "QB"}:
            history[(row["player_id"], row["team"])].append(
                (conservative_available_at(game), row["position"])
            )
    for records in history.values():
        records.sort()
    return dict(history)


def _asof_position(
    history: dict[tuple[str, str], list[tuple[datetime, str]]],
    player_id: str,
    team: str,
    cutoff: datetime,
) -> str:
    records = history.get((player_id, team), [])
    index = bisect.bisect_left(records, (cutoff, "")) - 1
    return records[index][1] if index >= 0 else "Unknown"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def play_context(play: dict[str, Any], opportunity_type: str, position: str) -> dict[str, Any]:
    """Construct only context known on the play, with explicit missing values."""
    yardline = _number(play.get("yardline_100"))
    air = _number(play.get("air_yards")) if opportunity_type == "receiving" else None
    context: dict[str, Any] = {
        "yardline_100": yardline,
        "log_yardline": math.log1p(yardline) if yardline is not None and yardline >= 0 else None,
        "goal_to_go": _number(play.get("goal_to_go")),
        "down": _number(play.get("down")),
        "ydstogo": _number(play.get("ydstogo")),
        "qtr": _number(play.get("qtr")),
        "game_seconds_remaining": _number(play.get("game_seconds_remaining")),
        "half_seconds_remaining": _number(play.get("half_seconds_remaining")),
        "score_differential": _number(play.get("score_differential")),
        "shotgun": _number(play.get("shotgun")),
        "no_huddle": _number(play.get("no_huddle")),
        "inside_5": int(yardline <= 5) if yardline is not None else None,
        "inside_10": int(yardline <= 10) if yardline is not None else None,
        "inside_20": int(yardline <= 20) if yardline is not None else None,
        "position": position,
    }
    if opportunity_type == "rushing":
        context.update(
            {
                "qb_scramble": _number(play.get("qb_scramble")),
                "run_location": play.get("run_location") or "Unknown",
                "run_gap": play.get("run_gap") or "Unknown",
            }
        )
    else:
        context.update(
            {
                "air_yards": air,
                "relative_to_endzone": relative_to_endzone(yardline, air),
                "end_zone_target": end_zone_target(yardline, air),
                "pass_location": play.get("pass_location") or "Unknown",
            }
        )
    return context


def extract_opportunities(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Extract 2017–2024 regular-season opportunities; never load 2025 PBP."""
    manifests: dict[str, dict[str, str]] = json.loads(
        (Path("reports") / "phase2_historical_source_hashes.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    root = data_dir / "raw" / "nflverse"
    for season in range(2017, 2025):
        hashes = manifests[str(season)]
        schedules = pl.read_parquet(root / f"schedules-{season}-{hashes['schedules']}.parquet")
        player_stats = pl.read_parquet(
            root / f"player-stats-{season}-{hashes['player_stats']}.parquet"
        )
        pbp = pl.read_parquet(
            root / f"pbp-{season}-{hashes['pbp']}.parquet", columns=list(PBP_COLUMNS)
        )
        games = {
            game["game_id"]: game
            for game in schedules.to_dicts()
            if game["season"] == season and game["game_type"] == "REG"
        }
        positions = _position_history(player_stats, games)
        counts = {"rushing": 0, "receiving": 0, "lateral_or_ambiguous_dropped": 0}
        for play in pbp.iter_rows(named=True):
            game = games.get(play["game_id"])
            if game is None:
                continue
            classified = classify_opportunity(play)
            if classified is None:
                if (
                    (play.get("rush_touchdown") == 1 or play.get("pass_touchdown") == 1)
                    and (play.get("rusher_player_id") or play.get("receiver_player_id"))
                    and (
                        play.get("lateral_rusher_player_id")
                        or play.get("lateral_receiver_player_id")
                        or play.get("td_player_id")
                        not in {play.get("rusher_player_id"), play.get("receiver_player_id")}
                    )
                ):
                    counts["lateral_or_ambiguous_dropped"] += 1
                continue
            opportunity_type, player_id, label = classified
            team = play["posteam"]
            if not team:
                raise ValueError(f"Opportunity without possessing team: {play['game_id']}")
            kickoff = kickoff_utc(game)
            context = play_context(
                play, opportunity_type, _asof_position(positions, player_id, team, kickoff)
            )
            rows.append(
                {
                    "season": season,
                    "week": game["week"],
                    "game": play["game_id"],
                    "play_id": play["play_id"],
                    "kickoff_time": kickoff,
                    "player_id": player_id,
                    "team": team,
                    "opportunity_type": opportunity_type,
                    "actual_td": label,
                    **context,
                }
            )
            counts[opportunity_type] += 1
        coverage[str(season)] = counts
    if any(row["season"] >= 2025 for row in rows):
        raise ValueError("Protected 2025 holdout entered play-level opportunity data")
    return pl.DataFrame(rows), coverage
