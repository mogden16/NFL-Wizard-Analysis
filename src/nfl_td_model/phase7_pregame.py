"""Closed-schema historical pregame projection; no result columns are loaded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from nfl_td_model.phase5 import HIERARCHY_FIELDS, hierarchy_predict
from nfl_td_model.phase7_collect import planned_games

SAFE_SOURCE = (
    "season", "game", "team", "opponent", "player_id", "player", "position",
    "prediction_time", "expected_team_td", "feature_available_at_max",
    "player_source_game_ids", "xtd_source_game_ids", "xtd_feature_available_at_max",
) + HIERARCHY_FIELDS
OUTPUT = Path("data/phase7/pregame_player_view.parquet")
FORBIDDEN = frozenset({"anytime_td", "actual_td_count", "final_score", "current_game_pbp",
                       "current_game_carries", "current_game_targets", "current_game_snaps",
                       "current_game_xtd", "closing_odds", "settlement_status", "p_market"})


def prepare_pregame_view() -> Path:
    """Write only audited lagged fields and frozen hierarchical probabilities."""
    if FORBIDDEN.intersection(SAFE_SOURCE):
        raise ValueError("Pregame projection contains result, price or settlement fields")
    p5 = json.loads(Path("reports/phase5_metrics.json").read_text(encoding="utf-8"))
    if list(p5["hierarchy_weights"]) != list(HIERARCHY_FIELDS):
        raise ValueError("Frozen Phase 5 hierarchy field order changed")
    features = pl.scan_parquet(
        "data/derived/phase5_2017_2024_player_features.parquet"
    ).filter(pl.col("season").is_in([2023, 2024])).select(SAFE_SOURCE).collect().sort(
        "season", "game", "team", "player_id"
    )
    if features.height != 12641 or sorted(features["season"].unique().to_list()) != [2023, 2024]:
        raise ValueError("Unexpected Phase 7 pregame population")
    if features.filter(pl.col("feature_available_at_max") >= pl.col("prediction_time")).height or features.filter(
        pl.col("xtd_feature_available_at_max") >= pl.col("prediction_time")
    ).height:
        raise ValueError("Historical feature availability crosses T-60")
    for row in features.select("game", "player_source_game_ids", "xtd_source_game_ids").iter_rows(named=True):
        if any(row[field] and row["game"] in row[field].split(";")
               for field in ("player_source_game_ids", "xtd_source_game_ids")):
            raise ValueError("Current-game source ID entered historical pregame view")
    weights = np.asarray(list(p5["hierarchy_weights"].values()), dtype=float)
    probabilities, expected = hierarchy_predict(features, weights)
    frozen = pl.scan_csv("reports/phase5_validation_predictions.csv").select(
        "game", "team", "player_id", "p_hierarchical", "expected_player_td"
    ).collect()
    comparison = features.filter(pl.col("season") == 2024).select("game", "team", "player_id").with_columns(
        pl.Series("p_recomputed", probabilities[np.asarray(features["season"] == 2024, dtype=bool)]),
        pl.Series("lambda_recomputed", expected[np.asarray(features["season"] == 2024, dtype=bool)]),
    ).join(frozen, on=["game", "team", "player_id"], how="left")
    if comparison.height != 6320 or not np.allclose(
        np.asarray(comparison["p_recomputed"], dtype=float),
        np.asarray(comparison["p_hierarchical"], dtype=float), atol=1e-10, rtol=0,
    ) or not np.allclose(
        np.asarray(comparison["lambda_recomputed"], dtype=float),
        np.asarray(comparison["expected_player_td"], dtype=float), atol=1e-10, rtol=0,
    ):
        raise ValueError("Hierarchical probabilities differ from frozen Phase 5 artifact")
    schedule = planned_games().select("season", "game", "event_id", "kickoff")
    view = features.select("season", "game", "team", "opponent", "player_id", "player",
                           "position", "prediction_time", "expected_team_td").with_columns(
        pl.Series("p_football", probabilities), pl.Series("expected_player_td", expected),
    ).join(schedule, on=["season", "game"], how="inner").sort(
        "season", "game", "team", "player_id"
    )
    if view.height != features.height or FORBIDDEN.intersection(view.columns):
        raise ValueError("Restricted pregame view changed or includes forbidden fields")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    view.write_parquet(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    Path("reports/phase7_pregame_view_manifest.json").write_text(
        json.dumps({"rows": view.height, "seasons": [2023, 2024],
                    "sha256": digest, "schema": view.schema.__str__()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return OUTPUT
