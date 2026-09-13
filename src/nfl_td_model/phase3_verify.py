"""Independent Phase 3 artifact and temporal acceptance checks."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl

from nfl_td_model.xtd_features import aggregate_games

ARTIFACTS = {
    "opportunities": "phase3_2017_2024_opportunities.parquet",
    "scored": "phase3_2018_2024_play_scores.parquet",
    "aggregates": "phase3_2018_2024_player_game_xtd.parquet",
    "lagged": "phase3_2018_2024_lagged_xtd_features.parquet",
}


def verify_phase3(data_dir: Path, reports: Path) -> dict[str, Any]:
    """Reject altered artifacts, 2025 use, invalid sums, or current-game lag leakage."""
    manifest: dict[str, str] = json.loads(
        (reports / "phase3_artifact_hashes.json").read_text(encoding="utf-8")
    )
    if set(manifest) != set(ARTIFACTS):
        raise ValueError("Phase 3 artifact manifest is incomplete")
    paths = {key: data_dir / "derived" / name for key, name in ARTIFACTS.items()}
    for key, path in paths.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"Missing or altered Phase 3 artifact: {key}")
    opportunities = pl.read_parquet(paths["opportunities"])
    scored = pl.read_parquet(paths["scored"])
    aggregates = pl.read_parquet(paths["aggregates"])
    lagged = pl.read_parquet(paths["lagged"])
    if opportunities["season"].min() != 2017 or opportunities["season"].max() != 2024:
        raise ValueError("Phase 3 opportunity seasons are not 2017–2024")
    if scored["season"].min() != 2018 or scored["season"].max() != 2024:
        raise ValueError("Phase 3 scoring seasons are not 2018–2024")
    if any(
        cast(int, frame["season"].max()) >= 2025
        for frame in (opportunities, scored, aggregates, lagged)
    ):
        raise ValueError("2025 protected holdout entered Phase 3 artifacts")
    if scored.height != opportunities.filter(pl.col("season") >= 2018).height:
        raise ValueError("Scored opportunity count differs from extracted opportunity count")
    if scored.filter((pl.col("xtd") <= 0) | (pl.col("xtd") >= 1)).height:
        raise ValueError("Play xTD is not strictly between zero and one")
    if scored.filter(pl.col("model_fit_max_season") >= pl.col("season")).height:
        raise ValueError("Scoring model was fitted on target or future season")
    if scored.filter(pl.col("model_calibration_season") >= pl.col("season")).height:
        raise ValueError("Scoring calibrator used target or future season")
    if scored.select("season", "game", "play_id", "opportunity_type").is_duplicated().any():
        raise ValueError("Duplicate scored play opportunity")
    expected = aggregate_games(scored)
    columns = [
        "rushing_xtd",
        "receiving_xtd",
        "total_xtd",
        "actual_td",
        "rush_attempts",
        "targets",
        "team_rushing_xtd",
        "team_receiving_xtd",
        "team_total_xtd",
    ]
    comparison = expected.join(
        aggregates,
        on=["season", "game", "kickoff_time", "team", "player_id"],
        how="full",
        suffix="_saved",
    )
    if comparison.height != expected.height or comparison.height != aggregates.height:
        raise ValueError("Player-game aggregate keys differ from play sums")
    for column in columns:
        if (
            comparison.select((pl.col(column) - pl.col(f"{column}_saved")).abs().max()).item()
            > 1e-10
        ):
            raise ValueError(f"Player-game {column} differs from play sum")
    aggregates_by_key = {
        (row["game"], row["team"], row["player_id"]): row for row in aggregates.to_dicts()
    }
    checked = 0
    for row in lagged.to_dicts():
        ids = row["xtd_source_game_ids"].split(";") if row["xtd_source_game_ids"] else []
        if row["xtd_history_games"] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("Invalid lagged xTD source lineage")
        if row["game"] in ids or row["xtd_availability_type"] != "assumed_kickoff_plus_24h":
            raise ValueError("Current game or availability assumption invalid in lagged xTD")
        sources = []
        for game in ids:
            source = aggregates_by_key.get((game, row["team"], row["player_id"]))
            if (
                source is None
                or source["kickoff_time"] + timedelta(days=1) >= row["prediction_time"]
            ):
                raise ValueError("Unavailable or wrong-player source entered lagged xTD")
            sources.append(source)
        latest = max((item["kickoff_time"] + timedelta(days=1) for item in sources), default=None)
        if row["xtd_feature_available_at_max"] != latest:
            raise ValueError("Incorrect lagged xTD availability time")
        for window, size in (("last3", 3), ("last5", 5), ("last8", 8)):
            selected = sources[-size:]
            if row[f"{window}_xtd_games"] != len(selected):
                raise ValueError("Lagged xTD window count differs from source lineage")
            observed = row[f"{window}_total_xtd"]
            expected_mean = (
                sum(item["total_xtd"] for item in selected) / len(selected) if selected else None
            )
            if observed is None and expected_mean is None:
                continue
            if observed is None or expected_mean is None or abs(observed - expected_mean) > 1e-10:
                raise ValueError("Lagged xTD window includes another game's value")
        checked += 1
    fit_manifest = json.loads((reports / "phase3_fit_manifest.json").read_text(encoding="utf-8"))
    if len(fit_manifest) != 14 or any(
        entry["fit_max_season"] >= int(key[:4]) for key, entry in fit_manifest.items()
    ):
        raise ValueError("Phase 3 fit manifest violates chronological training")
    return {
        "opportunities": opportunities.height,
        "scored": scored.height,
        "aggregates": aggregates.height,
        "lagged": checked,
    }
