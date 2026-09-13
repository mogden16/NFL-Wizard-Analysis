"""Aggregate scored opportunities and build strictly lagged xTD candidates."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from nfl_td_model.feature_windows import EWMA_HALF_LIFE_GAMES, WINDOWS

XTD_METRICS = ("rushing_xtd", "receiving_xtd", "total_xtd")
SHARE_METRICS = (
    "rushing_xtd_share",
    "receiving_xtd_share",
    "total_xtd_share",
)
RATE_METRICS = (
    "rushing_xtd_per_attempt",
    "receiving_xtd_per_target",
    "total_xtd_per_opportunity",
)


def aggregate_games(scored: pl.DataFrame) -> pl.DataFrame:
    """Use a direct sum of play probabilities for each player and game."""
    if scored.is_empty():
        raise ValueError("No scored opportunities")
    base = scored.group_by("season", "game", "kickoff_time", "team", "player_id").agg(
        pl.when(pl.col("opportunity_type") == "rushing")
        .then(pl.col("xtd"))
        .otherwise(0.0)
        .sum()
        .alias("rushing_xtd"),
        pl.when(pl.col("opportunity_type") == "receiving")
        .then(pl.col("xtd"))
        .otherwise(0.0)
        .sum()
        .alias("receiving_xtd"),
        pl.when(pl.col("opportunity_type") == "rushing")
        .then(1)
        .otherwise(0)
        .sum()
        .alias("rush_attempts"),
        pl.when(pl.col("opportunity_type") == "receiving")
        .then(1)
        .otherwise(0)
        .sum()
        .alias("targets"),
        pl.col("actual_td").sum().alias("actual_td"),
    )
    team = base.group_by("game", "team").agg(
        pl.col("rushing_xtd").sum().alias("team_rushing_xtd"),
        pl.col("receiving_xtd").sum().alias("team_receiving_xtd"),
    )
    base = base.join(team, on=["game", "team"], how="left").with_columns(
        (pl.col("rushing_xtd") + pl.col("receiving_xtd")).alias("total_xtd"),
        (pl.col("team_rushing_xtd") + pl.col("team_receiving_xtd")).alias("team_total_xtd"),
    )
    return base.with_columns(
        pl.when(pl.col("rush_attempts") > 0)
        .then(pl.col("rushing_xtd") / pl.col("rush_attempts"))
        .otherwise(None)
        .alias("rushing_xtd_per_attempt"),
        pl.when(pl.col("targets") > 0)
        .then(pl.col("receiving_xtd") / pl.col("targets"))
        .otherwise(None)
        .alias("receiving_xtd_per_target"),
        (pl.col("total_xtd") / (pl.col("rush_attempts") + pl.col("targets"))).alias(
            "total_xtd_per_opportunity"
        ),
        pl.when(pl.col("team_rushing_xtd") > 0)
        .then(pl.col("rushing_xtd") / pl.col("team_rushing_xtd"))
        .otherwise(None)
        .alias("rushing_xtd_share"),
        pl.when(pl.col("team_receiving_xtd") > 0)
        .then(pl.col("receiving_xtd") / pl.col("team_receiving_xtd"))
        .otherwise(None)
        .alias("receiving_xtd_share"),
        pl.when(pl.col("team_total_xtd") > 0)
        .then(pl.col("total_xtd") / pl.col("team_total_xtd"))
        .otherwise(None)
        .alias("total_xtd_share"),
    ).sort("season", "game", "team", "player_id")


def summarize_xtd_history(row: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Exclude the target game and any source unavailable at prediction time."""
    prediction: datetime = row["prediction_time"]
    eligible = sorted(
        (
            item
            for item in history
            if item["game"] != row["game"]
            and item["kickoff_time"] + timedelta(days=1) < prediction
            and item["season"] <= row["season"]
        ),
        key=lambda item: (item["kickoff_time"], item["game"]),
    )
    result: dict[str, Any] = {
        "season": row["season"],
        "week": row["week"],
        "game": row["game"],
        "team": row["team"],
        "player_id": row["player_id"],
        "position": row["position"],
        "prediction_time": prediction,
        "xtd_source_game_ids": ";".join(item["game"] for item in eligible),
        "xtd_history_games": len(eligible),
        "xtd_feature_available_at_max": (
            eligible[-1]["kickoff_time"] + timedelta(days=1) if eligible else None
        ),
        "xtd_availability_type": "assumed_kickoff_plus_24h",
    }
    for window, size in WINDOWS.items():
        selected = (
            eligible[-size:]
            if size
            else [item for item in eligible if item["season"] == row["season"]]
            if window == "season"
            else eligible
        )
        result[f"{window}_xtd_games"] = len(selected)
        for metric in (*XTD_METRICS, *SHARE_METRICS, *RATE_METRICS):
            valid = [item for item in selected if item.get(metric) is not None]
            if not valid:
                result[f"{window}_{metric}"] = None
                continue
            weights = [
                0.5 ** ((len(selected) - 1 - selected.index(item)) / EWMA_HALF_LIFE_GAMES)
                if window == "ewma"
                else 1.0
                for item in valid
            ]
            # Rates and shares use denominators across the window, not averages of game ratios.
            if metric in SHARE_METRICS:
                numerator = metric.replace("_share", "")
                denominator = f"team_{numerator}"
                num = sum(
                    float(item[numerator]) * weight
                    for item, weight in zip(valid, weights, strict=True)
                )
                den = sum(
                    float(item[denominator]) * weight
                    for item, weight in zip(valid, weights, strict=True)
                )
                value = num / den if den > 0 else None
            elif metric in RATE_METRICS:
                numerator = metric.split("_per_")[0]
                rate_denominator: str | None = {
                    "rushing_xtd_per_attempt": "rush_attempts",
                    "receiving_xtd_per_target": "targets",
                    "total_xtd_per_opportunity": None,
                }[metric]
                num = sum(
                    float(item[numerator]) * weight
                    for item, weight in zip(valid, weights, strict=True)
                )
                den = sum(
                    float(
                        item[rate_denominator]
                        if rate_denominator
                        else item["rush_attempts"] + item["targets"]
                    )
                    * weight
                    for item, weight in zip(valid, weights, strict=True)
                )
                value = num / den if den > 0 else None
            else:
                value = sum(
                    float(item[metric]) * weight
                    for item, weight in zip(valid, weights, strict=True)
                ) / sum(weights)
            result[f"{window}_{metric}"] = value
            if metric in XTD_METRICS:
                result[f"{window}_{metric}_sum"] = sum(float(item[metric]) for item in selected)
        for metric in XTD_METRICS:
            if not selected:
                result[f"{window}_{metric}_sum"] = None
    return result


def lagged_player_games(phase2: pl.DataFrame, aggregates: pl.DataFrame) -> pl.DataFrame:
    """Derive candidate pregame features from scored games strictly before each row."""
    history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in aggregates.to_dicts():
        history[(record["team"], record["player_id"])].append(record)
    rows = [
        summarize_xtd_history(row, history.get((row["team"], row["player_id"]), []))
        for row in phase2.filter(pl.col("season").is_between(2018, 2024)).to_dicts()
    ]
    return pl.DataFrame(rows).sort("season", "game", "team", "player_id")
