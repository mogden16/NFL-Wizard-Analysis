"""Point-in-time team offensive context and defensive context from nflverse."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.phase1 import kickoff_utc
from nfl_td_model.phase4_market import market_cutoff

PBP_FIELDS = (
    "game_id",
    "posteam",
    "defteam",
    "td_team",
    "rush_touchdown",
    "pass_touchdown",
    "rush_attempt",
    "pass_attempt",
    "two_point_attempt",
    "play_deleted",
    "qb_kneel",
    "qb_spike",
    "epa",
    "success",
    "yards_gained",
    "down",
    "yardline_100",
    "fixed_drive",
)
OFFENSE_METRICS = (
    "off_epa_per_play",
    "off_pass_epa",
    "off_rush_epa",
    "off_success_rate",
    "off_explosive_rate",
    "off_early_down_epa",
    "off_plays",
    "off_pass_rate",
    "off_drives",
    "off_red_zone_trips",
    "off_red_zone_td_conversion",
    "off_xtd",
    "off_red_zone_xtd",
)
DEFENSE_METRICS = (
    "def_epa_allowed",
    "def_pass_epa_allowed",
    "def_rush_epa_allowed",
    "def_success_allowed",
    "def_explosive_allowed",
    "def_red_zone_td_allowed",
    "def_xtd_allowed",
)


def normalize_team_abbr(team: str) -> str:
    """Use nflverse's modern Raiders abbreviation in pre-2020 schedules."""
    return "LV" if team == "OAK" else team


def offensive_td_play(play: dict[str, Any]) -> bool:
    """Include only offensive rush/pass TDs actually credited to the possessing team."""
    return bool(
        play.get("posteam")
        and play.get("td_team") == play.get("posteam")
        and (play.get("rush_touchdown") == 1 or play.get("pass_touchdown") == 1)
        and play.get("two_point_attempt") != 1
        and play.get("play_deleted") != 1
    )


def forecast_is_eligible(issued_at: datetime | None, prediction_time: datetime) -> bool:
    """A forecast without a verifiable pregame issue time cannot enter the model."""
    return issued_at is not None and issued_at <= prediction_time


def _game_metrics(
    pbp: pl.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], int]]:
    valid_team = pl.col("posteam").is_not_null()
    regular = (pl.col("two_point_attempt") != 1) & (pl.col("play_deleted") != 1)
    td = (
        pbp.filter(
            valid_team
            & regular
            & (pl.col("td_team") == pl.col("posteam"))
            & ((pl.col("rush_touchdown") == 1) | (pl.col("pass_touchdown") == 1))
        )
        .group_by("game_id", "posteam")
        .len(name="actual_offensive_td")
    )
    td_map = {(r["game_id"], r["posteam"]): r["actual_offensive_td"] for r in td.to_dicts()}
    plays = pbp.filter(
        valid_team
        & regular
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
        & ((pl.col("rush_attempt") == 1) | (pl.col("pass_attempt") == 1))
    )
    pass_play = pl.col("pass_attempt") == 1
    rush_play = pl.col("rush_attempt") == 1
    efficiency = plays.group_by("game_id", "posteam").agg(
        pl.col("epa").mean().alias("off_epa_per_play"),
        pl.when(pass_play).then(pl.col("epa")).otherwise(None).mean().alias("off_pass_epa"),
        pl.when(rush_play).then(pl.col("epa")).otherwise(None).mean().alias("off_rush_epa"),
        pl.col("success").mean().alias("off_success_rate"),
        pl.when(pass_play)
        .then(pl.col("yards_gained") >= 20)
        .otherwise(pl.col("yards_gained") >= 10)
        .mean()
        .alias("off_explosive_rate"),
        pl.when(pl.col("down") <= 2)
        .then(pl.col("epa"))
        .otherwise(None)
        .mean()
        .alias("off_early_down_epa"),
        pl.len().alias("off_plays"),
        pass_play.mean().alias("off_pass_rate"),
        pl.col("fixed_drive").n_unique().alias("off_drives"),
    )
    drives = (
        pbp.filter(valid_team & regular & pl.col("fixed_drive").is_not_null())
        .group_by("game_id", "posteam", "fixed_drive")
        .agg(
            (pl.col("yardline_100") <= 20).any().alias("red_zone_trip"),
            (
                (pl.col("td_team") == pl.col("posteam"))
                & ((pl.col("rush_touchdown") == 1) | (pl.col("pass_touchdown") == 1))
            )
            .any()
            .alias("off_td"),
        )
    )
    rz = (
        drives.group_by("game_id", "posteam")
        .agg(
            pl.col("red_zone_trip").sum().alias("off_red_zone_trips"),
            (pl.col("red_zone_trip") & pl.col("off_td")).sum().alias("red_zone_td_drives"),
        )
        .with_columns(
            pl.when(pl.col("off_red_zone_trips") > 0)
            .then(pl.col("red_zone_td_drives") / pl.col("off_red_zone_trips"))
            .otherwise(None)
            .alias("off_red_zone_td_conversion")
        )
    )
    joined = efficiency.join(rz, on=["game_id", "posteam"], how="left")
    return {(r["game_id"], r["posteam"]): r for r in joined.to_dicts()}, td_map


def build_team_history(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build 2017–2024 team-game outcomes and retrospective source metrics."""
    hashes = json.loads((Path("reports") / "phase2_historical_source_hashes.json").read_text())
    xplays = pl.read_parquet(
        data_dir / "derived" / "phase3_2018_2024_play_scores.parquet",
        columns=["game", "team", "yardline_100", "xtd"],
    )
    xmap = {
        (r["game"], r["team"]): r
        for r in xplays.group_by("game", "team")
        .agg(
            pl.col("xtd").sum().alias("off_xtd"),
            pl.when(pl.col("yardline_100") <= 20)
            .then(pl.col("xtd"))
            .otherwise(0.0)
            .sum()
            .alias("off_red_zone_xtd"),
        )
        .to_dicts()
    }
    rows = []
    target_mismatches = []
    for season in range(2017, 2025):
        manifest = hashes[str(season)]
        root = data_dir / "raw" / "nflverse"
        schedules = pl.read_parquet(root / f"schedules-{season}-{manifest['schedules']}.parquet")
        pbp = pl.read_parquet(
            root / f"pbp-{season}-{manifest['pbp']}.parquet", columns=list(PBP_FIELDS)
        )
        team_stats = pl.read_parquet(root / f"team-stats-{season}-{manifest['team_stats']}.parquet")
        regular_games = {
            game["game_id"]: game
            for game in schedules.to_dicts()
            if game["season"] == season and game["game_type"] == "REG"
        }
        pbp = pbp.filter(pl.col("game_id").is_in(regular_games))
        game_metrics, td_map = _game_metrics(pbp)
        stats_tds = {
            (r["game_id"], r["team"]): (r["rushing_tds"] or 0) + (r["passing_tds"] or 0)
            for r in team_stats.select("game_id", "team", "rushing_tds", "passing_tds").to_dicts()
        }
        for game in regular_games.values():
            kickoff = kickoff_utc(game)
            for is_home in (True, False):
                team = normalize_team_abbr(game["home_team"] if is_home else game["away_team"])
                opponent = normalize_team_abbr(game["away_team"] if is_home else game["home_team"])
                key = (game["game_id"], team)
                actual = td_map.get(key, 0)
                stat_td = stats_tds.get(key)
                if stat_td is None or actual != stat_td:
                    target_mismatches.append(
                        {"game_id": game["game_id"], "team": team, "pbp": actual, "stats": stat_td}
                    )
                metric = game_metrics.get(key, {})
                xmetric = xmap.get(key, {})
                rows.append(
                    {
                        "season": season,
                        "week": game["week"],
                        "game_id": game["game_id"],
                        "team": team,
                        "opponent": opponent,
                        "home": int(is_home),
                        "kickoff": kickoff,
                        "prediction_time": market_cutoff(kickoff),
                        "actual_offensive_td": actual,
                        "team_points": game["home_score"] if is_home else game["away_score"],
                        **{
                            field: metric.get(field)
                            for field in OFFENSE_METRICS
                            if field not in {"off_xtd", "off_red_zone_xtd"}
                        },
                        "off_xtd": xmetric.get("off_xtd") if season >= 2018 else None,
                        "off_red_zone_xtd": xmetric.get("off_red_zone_xtd")
                        if season >= 2018
                        else None,
                    }
                )
    if target_mismatches:
        raise ValueError(
            f"PBP offensive TD target disagrees with weekly team stats: {target_mismatches[:5]}"
        )
    frame = pl.DataFrame(rows, infer_schema_length=None).sort("season", "week", "game_id", "team")
    if frame.filter(pl.col("season") >= 2025).height:
        raise ValueError("Protected 2025 holdout entered football history")
    return frame, {
        "team_games": frame.height,
        "games": frame["game_id"].n_unique(),
        "target_mismatches": target_mismatches,
    }


def eligible_prior_games(
    history: list[dict[str, Any]], prediction_time: datetime, target_game_id: str
) -> list[dict[str, Any]]:
    """Apply the assumed +24h availability gate before any summary calculation."""
    return sorted(
        (
            item
            for item in history
            if item["game_id"] != target_game_id
            and item["kickoff"] + timedelta(days=1) < prediction_time
        ),
        key=lambda item: (item["kickoff"], item["game_id"]),
    )


def lag_team_context(history: pl.DataFrame) -> pl.DataFrame:
    """Create pregame offense, opponent-defense, and team xTD candidate features."""
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_defense: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history.to_dicts():
        by_team[row["team"]].append(row)
        by_defense[row["opponent"]].append(row)
    output = []
    for row in history.to_dicts():
        offense = eligible_prior_games(by_team[row["team"]], row["prediction_time"], row["game_id"])
        defense = eligible_prior_games(
            by_defense[row["opponent"]], row["prediction_time"], row["game_id"]
        )
        off5 = offense[-5:]
        def5 = defense[-5:]

        def avg(items: list[dict[str, Any]], field: str) -> float | None:
            if not items or any(item.get(field) is None for item in items):
                return None
            return sum(float(item[field]) for item in items) / len(items)

        features: dict[str, Any] = {
            "season": row["season"],
            "week": row["week"],
            "game_id": row["game_id"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home": row["home"],
            "kickoff": row["kickoff"],
            "prediction_time": row["prediction_time"],
            "actual_offensive_td": row["actual_offensive_td"],
            "team_points": row["team_points"],
            "off_source_game_ids": ";".join(item["game_id"] for item in offense),
            "def_source_game_ids": ";".join(item["game_id"] for item in defense),
            "off_history_games": len(offense),
            "def_history_games": len(defense),
            "feature_available_at_max": max(
                (item["kickoff"] + timedelta(days=1) for item in offense + defense), default=None
            ),
            "availability_type": "assumed_kickoff_plus_24h",
            "rest_days": (row["kickoff"] - offense[-1]["kickoff"]).total_seconds() / 86400
            if offense
            else None,
            "weather_forecast_issue_time": None,
            "weather_forecast_temperature": None,
            "starting_qb_signal": None,
        }
        for field in OFFENSE_METRICS:
            features[f"last5_{field}"] = avg(off5, field)
        for source, output_name in (
            ("off_epa_per_play", "def_epa_allowed"),
            ("off_pass_epa", "def_pass_epa_allowed"),
            ("off_rush_epa", "def_rush_epa_allowed"),
            ("off_success_rate", "def_success_allowed"),
            ("off_explosive_rate", "def_explosive_allowed"),
            ("off_red_zone_td_conversion", "def_red_zone_td_allowed"),
            ("off_xtd", "def_xtd_allowed"),
        ):
            features[f"last5_{output_name}"] = avg(def5, source)
        output.append(features)
    return pl.DataFrame(output, infer_schema_length=None).sort("season", "week", "game_id", "team")
