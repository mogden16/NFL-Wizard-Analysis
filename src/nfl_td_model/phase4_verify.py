"""Independent Phase 4 source, odds cutoff, target, lag, and distribution checks."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.phase4_football import build_team_history, forecast_is_eligible, lag_team_context
from nfl_td_model.phase4_market import build_strict_market_table
from nfl_td_model.phase4_models import count_distribution

ARTIFACTS = {
    "football": "phase4_football_only_2017_2024.parquet",
    "strict": "phase4_strict_market_2021_2024.parquet",
    "predictions": "phase4_team_game_predictions_2024.parquet",
}


def verify_phase4(settings: Settings, reports: Path) -> dict[str, Any]:
    """Reject altered sources, current-game feature use, future odds, or 2025 tuning."""
    hashes: dict[str, str] = json.loads((reports / "phase4_artifact_hashes.json").read_text())
    if set(hashes) != set(ARTIFACTS):
        raise ValueError("Phase 4 artifact manifest is incomplete")
    paths = {key: settings.data_dir / "derived" / name for key, name in ARTIFACTS.items()}
    for key, path in paths.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
            raise ValueError(f"Missing or altered Phase 4 artifact: {key}")
    football = pl.read_parquet(paths["football"])
    strict = pl.read_parquet(paths["strict"])
    predictions = pl.read_parquet(paths["predictions"])
    if football.height != 4222 or strict.height != 2174 or predictions.height != 544:
        raise ValueError("Unexpected Phase 4 history or validation size")
    if any(cast(int, frame["season"].max()) >= 2025 for frame in (football, strict, predictions)):
        raise ValueError("Protected 2025 season entered Phase 4")
    if predictions["season"].unique().to_list() != [2024]:
        raise ValueError("Phase 4 prediction table is not 2024 validation")
    market_columns = (
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "home",
        "kickoff",
        "prediction_time",
        "sportsbook",
        "event_id",
        "odds_timestamp",
        "spread_quote_time",
        "total_quote_time",
        "odds_snapshot_time",
        "home_spread",
        "home_margin",
        "team_spread",
        "game_total",
        "implied_team_points",
    )
    market, _ = build_strict_market_table(settings)
    if not strict.select(market_columns).equals(market.select(market_columns)):
        raise ValueError("Strict table differs from immutable historical odds snapshots")
    for timestamp in (
        "odds_timestamp",
        "spread_quote_time",
        "total_quote_time",
        "odds_snapshot_time",
    ):
        if strict.filter(pl.col(timestamp) > pl.col("prediction_time")).height:
            raise ValueError(f"Future or closing {timestamp} entered T-60 table")
    if strict.filter(
        pl.col("prediction_time") != pl.col("kickoff") - pl.duration(minutes=60)
    ).height:
        raise ValueError("Market prediction timestamp is not kickoff minus 60 minutes")
    source_history, meta = build_team_history(settings.data_dir)
    rebuilt_football = lag_team_context(source_history)
    if meta["target_mismatches"] or rebuilt_football.columns != football.columns:
        raise ValueError("Football table differs from PBP scoring and lagged source reconstruction")
    for column in football.columns:
        if football[column].dtype in (pl.Float32, pl.Float64):
            left = np.asarray(football[column].to_numpy(), dtype=float)
            right = np.asarray(rebuilt_football[column].to_numpy(), dtype=float)
            if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"Football {column} differs from source reconstruction")
        elif not football[column].equals(rebuilt_football[column]):
            raise ValueError(f"Football {column} differs from source reconstruction")
    source_rows = source_history.to_dicts()
    by_team_game = {(r["game_id"], r["team"]): r for r in source_rows}
    by_def_game = {(r["game_id"], r["opponent"]): r for r in source_rows}
    checked = 0
    for row in football.to_dicts():
        for family, team in (("off", row["team"]), ("def", row["opponent"])):
            raw = row[f"{family}_source_game_ids"]
            ids = raw.split(";") if raw else []
            if row[f"{family}_history_games"] != len(ids) or row["game_id"] in ids:
                raise ValueError("Current game or invalid source count in team history")
            for game_id in ids:
                source = (by_team_game if family == "off" else by_def_game).get((game_id, team))
                if (
                    source is None
                    or source["kickoff"] + timedelta(days=1) >= row["prediction_time"]
                ):
                    raise ValueError("Unavailable team or opponent game entered rolling features")
        if row["weather_forecast_issue_time"] is not None and not forecast_is_eligible(
            row["weather_forecast_issue_time"], row["prediction_time"]
        ):
            raise ValueError("Weather forecast issued after prediction cutoff")
        checked += 1
    probabilities = predictions.select(
        "p_team_0_td", "p_team_1_td", "p_team_2_td", "p_team_3_td", "p_team_4plus_td"
    ).to_numpy()
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Team TD count probabilities do not sum to one")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Team TD count probability outside [0, 1]")
    expected = np.asarray(
        [count_distribution(float(value)) for value in predictions["predicted_offensive_td"]]
    )
    if not np.allclose(probabilities, expected, atol=1e-10):
        raise ValueError("Count distribution disagrees with predicted team TD mean")
    metrics = json.loads((reports / "phase4_metrics.json").read_text())
    if metrics["validation_rows"] != predictions.height or metrics["strict_train_rows"] != 1630:
        raise ValueError("Phase 4 reported split differs from artifacts")
    for experiment in metrics["experiments"].values():
        if experiment["fit_max_season"] > 2023 or experiment["train_seasons"][-1] > 2023:
            raise ValueError("2024 validation or 2025 entered fit or feature selection")
        forbidden = {"actual_offensive_td", "team_points", "home_score", "away_score"}
        if forbidden.intersection(experiment["fields"]):
            raise ValueError("Current-game outcome entered a Phase 4 model feature list")
    if metrics["headline_metrics"]["football-only diagnostic"]["fit_max_season"] > 2023:
        raise ValueError("Football-only fit entered validation or holdout")
    if metrics["selected_model"] != predictions["model_version"][0]:
        raise ValueError("Model version differs from reported selection")
    return {
        "football_team_games": football.height,
        "strict_market_team_games": strict.height,
        "validation_team_games": predictions.height,
        "lagged_rows_checked": checked,
    }
