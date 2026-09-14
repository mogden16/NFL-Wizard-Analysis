"""Independent, read-only Phase 5 artifact and point-in-time verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl_td_model.phase5 import MODEL_FAMILIES, SAFE_MODEL_FIELDS


def verify_phase5() -> dict[str, Any]:
    """Verify hashes, chronology, source provenance, labels, and team coherence."""
    report = json.loads(Path("reports/phase5_metrics.json").read_text(encoding="utf-8"))
    feature_path = Path("data/derived/phase5_2017_2024_player_features.parquet")
    prediction_path = Path("reports/phase5_validation_predictions.csv")
    for name, path in (("feature_table", feature_path),
                       ("validation_predictions", prediction_path)):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != report["artifact_hashes"][name]:
            raise ValueError(f"Missing or altered Phase 5 {name}")
    if report["phase"] != 5 or not report["holdout_2025_untouched"] or not report["diagnostic_exhibition_slate_excluded"]:
        raise ValueError("Phase 5 firewall declaration is missing")
    if report["calibration"]["base_fit_seasons"] != [2022] or report["calibration"]["calibration_fit_season"] != 2023:
        raise ValueError("Phase 5 fitting/calibration chronology changed")
    if report["team_model_fit_seasons"] != {"2022": [2021], "2023": [2021, 2022],
                                            "2024": [2021, 2022, 2023]}:
        raise ValueError("Team environment model no longer uses prior-season fits")
    if set(report["controlled_comparisons"]) != set(MODEL_FAMILIES):
        raise ValueError("Controlled feature-family comparison is incomplete")
    if any(field not in SAFE_MODEL_FIELDS for family in MODEL_FAMILIES.values() for field in family):
        raise ValueError("Unsafe player-model field")
    features = pl.read_parquet(feature_path)
    predictions = pl.read_csv(prediction_path, try_parse_dates=True)
    if features.height != report["population"]["all_2017_2024_rows"] or predictions.height != report["population"]["validation_rows"]:
        raise ValueError("Phase 5 population size changed")
    if sorted(features["season"].unique().to_list()) != list(range(2017, 2025)) or predictions["season"].unique().to_list() != [2024]:
        raise ValueError("Protected season entered Phase 5 artifacts")
    if features.select("game", "team", "player_id").is_duplicated().any() or predictions.select("game", "team", "player_id").is_duplicated().any():
        raise ValueError("Duplicate player-game key")
    if not features["position"].is_in(["RB", "WR", "TE", "QB"]).all():
        raise ValueError("Unexpected player position")
    if features.filter(pl.col("season") <= 2021).filter(pl.col("expected_team_td").is_not_null()).height:
        raise ValueError("Warmup seasons have an in-sample team expectation")
    if features.filter(pl.col("season") >= 2022).filter(pl.col("expected_team_td").is_null()).height:
        raise ValueError("Formal player seasons lack out-of-sample team expectations")
    for name in ("feature_available_at_max", "xtd_feature_available_at_max", "odds_timestamp",
                 "spread_quote_time", "total_quote_time"):
        if features.filter(pl.col(name) >= pl.col("prediction_time")).height:
            raise ValueError(f"{name} is after the T-60 cutoff")
    for row in features.select("game", "player_source_game_ids", "xtd_source_game_ids").iter_rows(named=True):
        for source in ("player_source_game_ids", "xtd_source_game_ids"):
            if row[source] and row["game"] in row[source].split(";"):
                raise ValueError(f"Current game entered {source}")
    market_rows = predictions.filter(pl.col("p_market").is_not_null())
    if market_rows.height != report["market_benchmark"]["matched_players"] or market_rows["game"].n_unique() != report["market_benchmark"]["games"]:
        raise ValueError("Historical player-market benchmark coverage changed")
    if predictions.filter(pl.col("actual_td_count") > 0).height != predictions["anytime_td"].sum():
        raise ValueError("ATD binary labels disagree with rushing/receiving TD counts")
    opportunities = pl.read_parquet(
        "data/derived/phase3_2017_2024_opportunities.parquet",
        columns=["season", "game", "player_id", "actual_td"],
    ).filter(pl.col("season") == 2024).group_by("game", "player_id").agg(
        pl.col("actual_td").sum().alias("source_td")
    )
    crosscheck = predictions.join(opportunities, on=["game", "player_id"], how="left")
    if crosscheck.filter(pl.col("actual_td_count") != pl.col("source_td").fill_null(0)).height:
        raise ValueError("Phase 5 outcome labels differ from frozen Phase 3 opportunities")
    for probability in ("p_historical_td", "p_hierarchical", "p_logistic_raw", "p_logistic"):
        values = np.asarray(predictions[probability], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid {probability}")
    coherence = predictions.group_by("game", "team").agg(
        pl.col("expected_player_td").sum().alias("allocated"),
        pl.col("expected_team_td").first().alias("team_expected"),
    )
    if coherence.height != report["coherence"]["team_games"]:
        raise ValueError("Unexpected team-game coverage")
    gap = np.asarray(coherence["allocated"] - coherence["team_expected"], dtype=float)
    if not np.isfinite(gap).all() or np.max(np.abs(gap)) > 1e-7:
        raise ValueError("Hierarchical player TD allocation is not team-coherent")
    return {"feature_rows": features.height, "validation_rows": predictions.height,
            "team_games": coherence.height, "market_matches": market_rows.height}
