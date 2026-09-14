"""Independent Phase 6 hash, chronology, OOF, and champion verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl_td_model.phase5 import (
    HIERARCHY_FIELDS,
    MODEL_FAMILIES,
    fit_player_logistic,
    hierarchy_predict,
    logistic_predict,
)
from nfl_td_model.phase5_verify import verify_phase5
from nfl_td_model.phase6 import (
    BLENDS,
    DATA,
    POSITION_CORE,
    POSITION_XTD,
    POSITIONS,
    SAFE_FIELDS,
    TREE_CONFIGS,
    _fit_logistic,
    _fit_tree,
    _predict_logistic,
    _predict_tree,
    probability_metrics,
)


def verify_phase6() -> dict[str, Any]:
    """Rebuild 2023 component OOF scores without reading future outcomes."""
    verify_phase5()
    detail = json.loads(Path("reports/phase6_metrics.json").read_text(encoding="utf-8"))
    files = {
        "validation_predictions": Path("reports/phase6_validation_predictions.csv"),
        "oof_components": Path("reports/phase6_2023_oof_components.csv"),
    }
    for name, path in files.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != detail["artifact_hashes"][name]:
            raise ValueError(f"Missing or altered Phase 6 {name}")
    if detail["phase"] != 6 or not detail["holdout_2025_untouched"] or not detail["diagnostic_exhibition_slate_excluded"]:
        raise ValueError("Phase 6 research firewall declaration changed")
    if detail["chronology"]["component_fit_seasons"] != [2022] or detail["chronology"]["oof_and_calibration_season"] != 2023 or detail["chronology"]["validation_season"] != 2024:
        raise ValueError("Phase 6 chronological split changed")
    if set(detail["lightgbm_registry"]) != set(TREE_CONFIGS) or set(detail["blend_registry"]) != set(BLENDS):
        raise ValueError("Phase 6 experiment registry is incomplete")
    for name, config in TREE_CONFIGS.items():
        if detail["lightgbm_registry"][name]["parameters"] != config:
            raise ValueError("LightGBM configuration differs from preregistration")
    for name, weights in BLENDS.items():
        if detail["blend_registry"][name]["weights"] != list(weights):
            raise ValueError("Blend weights differ from preregistration")
    if any("route" in field or field in {"p_market", "anytime_td", "actual_td_count"}
           or field.startswith("current_game_") for field in SAFE_FIELDS):
        raise ValueError("Unsafe field entered Phase 6 model allowlist")
    feature = pl.read_parquet(DATA)
    if sorted(feature["season"].unique().to_list()) != list(range(2017, 2025)):
        raise ValueError("Phase 6 source crossed 2025/2026 firewall")
    train = feature.filter(pl.col("season") == 2022).sort("game", "team", "player_id")
    cal = feature.filter(pl.col("season") == 2023).sort("game", "team", "player_id")
    valid = pl.read_csv(files["validation_predictions"])
    oof = pl.read_csv(files["oof_components"])
    if (train.height, cal.height, valid.height, oof.height) != (6441, 6321, 6320, 6321):
        raise ValueError("Phase 6 player population changed")
    if valid["season"].unique().to_list() != [2024] or oof["season"].unique().to_list() != [2023] or oof["component_fit_max_season"].unique().to_list() != [2022]:
        raise ValueError("2023 OOF or 2024 validation chronology changed")
    if valid.select("game", "team", "player_id").is_duplicated().any() or oof.select("game", "team", "player_id").is_duplicated().any():
        raise ValueError("Duplicate Phase 6 player-game key")
    keys = ["game", "team", "player_id"]
    if not oof.select(keys).equals(cal.select(keys)):
        raise ValueError("2023 OOF rows do not match frozen historical universe")
    frozen_p5 = pl.read_csv("reports/phase5_validation_predictions.csv")
    if not valid.select(keys).equals(frozen_p5.select(keys)):
        raise ValueError("2024 Phase 6 rows do not match frozen Phase 5 predictions")
    if not valid["anytime_td"].equals(frozen_p5["anytime_td"]):
        raise ValueError("Phase 6 validation labels changed")
    if not np.allclose(np.asarray(valid["p_hierarchical"], dtype=float),
                       np.asarray(frozen_p5["p_hierarchical"], dtype=float), atol=1e-12, rtol=0):
        raise ValueError("Frozen hierarchical benchmark changed")
    p5_metrics = json.loads(Path("reports/phase5_metrics.json").read_text(encoding="utf-8"))
    if (detail["artifact_hashes"]["phase5_features"] != p5_metrics["artifact_hashes"]["feature_table"]
            or detail["artifact_hashes"]["phase5_predictions"]
            != p5_metrics["artifact_hashes"]["validation_predictions"]):
        raise ValueError("Phase 6 references altered Phase 5 artifact hashes")
    hierarchy_weights = np.asarray(list(p5_metrics["hierarchy_weights"].values()), dtype=float)
    if list(p5_metrics["hierarchy_weights"]) != list(HIERARCHY_FIELDS):
        raise ValueError("Hierarchy weight order changed")
    expected_hierarchy, _ = hierarchy_predict(cal, hierarchy_weights)
    pooled = fit_player_logistic(train, MODEL_FAMILIES["E_team_usage_xtd"])
    expected_pooled = logistic_predict(pooled, cal, MODEL_FAMILIES["E_team_usage_xtd"])
    expected_position = np.empty(cal.height)
    for position in POSITIONS:
        current_train = train.filter(pl.col("position") == position)
        current_cal = cal.filter(pl.col("position") == position)
        fields = POSITION_CORE[position] + POSITION_XTD[position]
        model = _fit_logistic(current_train, fields)
        mask = np.asarray(cal["position"] == position, dtype=bool)
        expected_position[mask] = _predict_logistic(model, current_cal, fields)
    expected_tree = _predict_tree(_fit_tree(train, "conservative"), cal)
    for name, expected in (
        ("p_hierarchical_oof", expected_hierarchy),
        ("p_pooled_logistic_oof_raw", expected_pooled),
        ("p_position_logistic_oof_raw", expected_position),
        ("p_lightgbm_conservative_oof_raw", expected_tree),
    ):
        if not np.allclose(np.asarray(oof[name], dtype=float), expected, atol=1e-10, rtol=0):
            raise ValueError(f"2023 {name} is not a 2022-only component prediction")
    labels = np.asarray(valid["anytime_td"], dtype=int)
    for column, metric_name in (("p_hierarchical", "phase5_hierarchical"),
                                ("p_position_logistic", "position_logistic"),
                                ("p_lightgbm", "lightgbm"), ("p_oof_blend", "oof_blend")):
        predicted = np.asarray(valid[column], dtype=float)
        if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
            raise ValueError(f"Invalid {column} probability")
        verified = probability_metrics(labels, predicted)
        if abs(verified["log_loss"] - detail["metrics"][metric_name]["log_loss"]) > 1e-10:
            raise ValueError(f"Reported {metric_name} log loss disagrees with predictions")
    expected_blend = min(detail["blend_registry"], key=lambda name: (
        detail["blend_registry"][name]["oof_metrics"]["log_loss"],
        detail["blend_registry"][name]["oof_metrics"]["brier"],
    ))
    if detail["selected_blend"] != expected_blend:
        raise ValueError("Blend was selected using something other than 2023 OOF metrics")
    expected_tree = min(detail["lightgbm_registry"], key=lambda name: (
        detail["lightgbm_registry"][name]["selected_metrics"]["log_loss"],
        detail["lightgbm_registry"][name]["selected_metrics"]["brier"],
    ))
    if detail["selected_tree_config"] != expected_tree:
        raise ValueError("LightGBM selection differs from recorded validation metrics")
    base_log_loss = detail["metrics"]["phase5_hierarchical"]["log_loss"]
    for name, verdict in detail["promotion"].items():
        if (verdict["gates"]["material_log_loss"]
                != (detail["metrics"][name]["log_loss"] <= base_log_loss - .005)
                or verdict["promoted"] != all(verdict["gates"].values())):
            raise ValueError("Phase 6 promotion record contradicts preregistered gates")
    passing = [name for name, verdict in detail["promotion"].items() if verdict["promoted"]]
    expected_champion = (min(passing, key=lambda name: detail["metrics"][name]["log_loss"])
                         if passing else "phase5_hierarchical")
    if detail["champion"] != expected_champion:
        raise ValueError("PHASE6_CHAMPION contradicts predeclared promotion gates")
    champion_source = {"phase5_hierarchical": "p_hierarchical",
                       "position_logistic": "p_position_logistic", "lightgbm": "p_lightgbm",
                       "oof_blend": "p_oof_blend"}[expected_champion]
    if not np.allclose(np.asarray(valid["p_phase6_champion"], dtype=float),
                       np.asarray(valid[champion_source], dtype=float), atol=1e-12, rtol=0):
        raise ValueError("Saved champion probabilities disagree with designation")
    if valid["p_market"].is_not_null().sum() != detail["market"]["rows"]:
        raise ValueError("Strict T-60 market benchmark coverage changed")
    return {"fit_rows": train.height, "oof_rows": oof.height,
            "validation_rows": valid.height, "champion": expected_champion,
            "market_matches": detail["market"]["rows"]}
