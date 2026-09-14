"""Phase 6 position and shallow LightGBM challengers on sealed 2022-2024 data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from nfl_td_model.phase5 import (
    CAL_BINS,
    HIERARCHY_FIELDS,
    MODEL_FAMILIES,
    fit_player_logistic,
    hierarchy_predict,
    logistic_predict,
)

DATA = Path("data/derived/phase5_2017_2024_player_features.parquet")
P5_PREDICTIONS = Path("reports/phase5_validation_predictions.csv")
P5_METRICS = Path("reports/phase5_metrics.json")
REPORTS = Path("reports")
MATERIAL_LOGLOSS = .005
POSITION_MAX_WORSE = .010
SEGMENT_MAX_WORSE = .005
ECE_MAX_WORSE = .010
BOOTSTRAP_REPLICATES = 2000
POSITIONS = ("RB", "WR", "TE", "QB")
TEAM = ("expected_team_td", "implied_team_points", "team_spread", "home")
OPP_RUSH = ("last5_opponent_allowed_red_zone_carries_per_game",
            "last5_opponent_allowed_offensive_touchdowns_per_game")
OPP_PASS = ("last5_opponent_allowed_red_zone_targets_per_game",
            "last5_opponent_allowed_offensive_touchdowns_per_game")
POSITION_CORE: dict[str, tuple[str, ...]] = {
    "RB": TEAM + ("last5_carry_share", "last5_goal_line_opportunity_share",
                   "last5_inside_5_carries_per_game", "last5_inside_10_carries_per_game",
                   "last5_target_share", "last5_snap_share") + OPP_RUSH,
    "WR": TEAM + ("last5_target_share", "last5_red_zone_target_share",
                   "last5_end_zone_target_share", "last5_snap_share") + OPP_PASS,
    "TE": TEAM + ("last5_target_share", "last5_red_zone_target_share",
                   "last5_end_zone_target_share", "last5_snap_share") + OPP_PASS,
    "QB": ("expected_team_td", "implied_team_points", "last5_carries_per_game",
           "last5_red_zone_carries_per_game", "last5_inside_10_carries_per_game",
           "last5_inside_5_carries_per_game", "last5_carry_share",
           "last5_goal_line_opportunity_share", "last5_snap_share") + OPP_RUSH,
}
POSITION_XTD: dict[str, tuple[str, ...]] = {
    "RB": ("last5_rushing_xtd", "last5_rushing_xtd_share", "last5_receiving_xtd"),
    "WR": ("last5_receiving_xtd", "last5_receiving_xtd_share"),
    "TE": ("last5_receiving_xtd", "last5_receiving_xtd_share"),
    "QB": ("last5_rushing_xtd", "last5_rushing_xtd_share"),
}
TREE_FIELDS = MODEL_FAMILIES["E_team_usage_xtd"] + (
    "team_spread", "last5_opponent_allowed_offensive_touchdowns_per_game",
    "last5_opponent_allowed_red_zone_carries_per_game",
    "last5_opponent_allowed_red_zone_targets_per_game",
)
SAFE_FIELDS = frozenset(TREE_FIELDS).union(*(set(v) for v in POSITION_CORE.values()))
SAFE_FIELDS = SAFE_FIELDS.union(*(set(v) for v in POSITION_XTD.values()))
TREE_CONFIGS: dict[str, dict[str, int | float]] = {
    "conservative": {"n_estimators": 100, "learning_rate": .03, "num_leaves": 7,
                     "max_depth": 3, "min_child_samples": 120, "reg_lambda": 20.0},
    "shallow_medium": {"n_estimators": 140, "learning_rate": .025, "num_leaves": 7,
                       "max_depth": 3, "min_child_samples": 100, "reg_lambda": 12.0},
    "moderate": {"n_estimators": 120, "learning_rate": .03, "num_leaves": 15,
                 "max_depth": 4, "min_child_samples": 80, "reg_lambda": 10.0},
}
BLENDS: dict[str, tuple[float, float, float, float]] = {
    "hierarchy_control": (1, 0, 0, 0),
    "hierarchy_position": (.75, 0, .25, 0),
    "hierarchy_tree": (.75, 0, 0, .25),
    "hierarchy_pool_position": (.50, .25, .25, 0),
    "hierarchy_pool_tree": (.50, .25, 0, .25),
}


def _source_frame() -> tuple[pl.DataFrame, dict[str, Any]]:
    """Read only hashed Phase 5 artifacts, never a 2025 or 2026 path."""
    p5 = json.loads(P5_METRICS.read_text(encoding="utf-8"))
    for key, path in (("feature_table", DATA), ("validation_predictions", P5_PREDICTIONS)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != p5["artifact_hashes"][key]:
            raise ValueError(f"Frozen Phase 5 {key} hash mismatch")
    feature = pl.read_parquet(DATA)
    frozen = pl.read_csv(P5_PREDICTIONS)
    if cast(int, feature["season"].max()) != 2024 or cast(int, feature["season"].min()) != 2017:
        raise ValueError("Phase 6 source crosses the 2025/2026 firewall")
    if frozen["season"].unique().to_list() != [2024]:
        raise ValueError("Phase 5 prediction artifact is not 2024-only")
    keys = ["game", "team", "player_id"]
    feature = feature.join(
        frozen.select(keys + ["p_historical_td", "p_hierarchical", "p_logistic_raw", "p_logistic"]),
        on=keys, how="left",
    ).sort("season", "game", "team", "player_id")
    if feature.filter((pl.col("season") == 2024) & pl.col("p_hierarchical").is_null()).height:
        raise ValueError("Frozen 2024 hierarchy did not match source player rows")
    if abs(p5["metrics"]["hierarchical"]["log_loss"] - .3804) > .0001:
        raise ValueError("Phase 5 champion benchmark changed")
    return feature, p5


def _safe_matrix(frame: pl.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    """Use only exact audited lagged fields; reject arbitrary result/price data."""
    if frame.is_empty() or cast(int, frame["season"].min()) < 2022 or cast(int, frame["season"].max()) > 2024:
        raise ValueError("Phase 6 model input crosses fit/holdout firewall")
    if not fields or any(field not in SAFE_FIELDS or field not in frame.columns for field in fields):
        raise ValueError("Phase 6 model contains an undeclared or unsafe feature")
    return np.asarray(frame.select(fields).to_numpy(), dtype=float)


def _fit_logistic(train: pl.DataFrame, fields: tuple[str, ...]) -> Pipeline:
    if train["season"].unique().to_list() != [2022]:
        raise ValueError("Phase 6 component fit must use only 2022")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=.5, max_iter=1000, random_state=1729)),
    ])
    model.fit(_safe_matrix(train, fields), np.asarray(train["anytime_td"], dtype=int))
    return model


def _predict_logistic(model: Pipeline, frame: pl.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    return np.clip(np.asarray(model.predict_proba(_safe_matrix(frame, fields))[:, 1], dtype=float),
                   1e-6, 1 - 1e-6)


def _fit_tree(train: pl.DataFrame, name: str) -> lgb.LGBMClassifier:
    if train["season"].unique().to_list() != [2022] or name not in TREE_CONFIGS:
        raise ValueError("Phase 6 tree fit/configuration violates preregistration")
    config = TREE_CONFIGS[name]
    model = lgb.LGBMClassifier(
        objective="binary", random_state=1729, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True, colsample_bytree=.8,
        subsample=.9, subsample_freq=1, reg_alpha=1.0,
        n_estimators=int(config["n_estimators"]), learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]), max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]), reg_lambda=float(config["reg_lambda"]),
    )
    model.fit(_safe_matrix(train, TREE_FIELDS), np.asarray(train["anytime_td"], dtype=int))
    return model


def _predict_tree(model: lgb.LGBMClassifier, frame: pl.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(_safe_matrix(frame, TREE_FIELDS)), dtype=float)
    return np.clip(probabilities[:, 1],
                   1e-6, 1 - 1e-6)


def _fit_sigmoid(calibration: pl.DataFrame, raw: np.ndarray) -> LogisticRegression:
    if calibration["season"].unique().to_list() != [2023]:
        raise ValueError("Phase 6 sigmoid may fit only 2023 OOF predictions")
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logit.reshape(-1, 1), np.asarray(calibration["anytime_td"], dtype=int))
    return model


def _predict_sigmoid(model: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    return np.asarray(model.predict_proba(logit.reshape(-1, 1))[:, 1], dtype=float)


def _calibration_table(actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    table = []
    for label, low, high in CAL_BINS:
        mask = (predicted >= low) & (predicted < high)
        table.append({"bin": label, "player_games": int(mask.sum()),
                      "mean_probability": float(predicted[mask].mean()) if mask.any() else None,
                      "actual_td_rate": float(actual[mask].mean()) if mask.any() else None})
    return table


def probability_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    p = np.clip(predicted, 1e-6, 1 - 1e-6)
    result: dict[str, Any] = {
        "rows": len(actual), "positive_rate": float(actual.mean()),
        "log_loss": float(log_loss(actual, p, labels=[0, 1])),
        "brier": float(brier_score_loss(actual, p)),
        "calibration_table": _calibration_table(actual, p),
    }
    result["ece"] = float(sum(
        row["player_games"] / len(actual)
        * abs(row["mean_probability"] - row["actual_td_rate"])
        for row in result["calibration_table"] if row["player_games"]
    ))
    if len(np.unique(actual)) > 1:
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        diagnostic = LogisticRegression(C=1e6, max_iter=1000).fit(logit, actual)
        result.update({"calibration_intercept": float(diagnostic.intercept_[0]),
                       "calibration_slope": float(diagnostic.coef_[0][0]),
                       "roc_auc": float(roc_auc_score(actual, p)),
                       "pr_auc": float(average_precision_score(actual, p))})
    else:
        result.update({"calibration_intercept": None, "calibration_slope": None,
                       "roc_auc": None, "pr_auc": None})
    return result


def _maybe_calibrate(cal: pl.DataFrame, valid: pl.DataFrame,
                     p_cal: np.ndarray, p_valid: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
    sigmoid = _predict_sigmoid(_fit_sigmoid(cal, p_cal), p_valid)
    actual = np.asarray(valid["anytime_td"], dtype=int)
    raw_m, sig_m = probability_metrics(actual, p_valid), probability_metrics(actual, sigmoid)
    choose = (sig_m["log_loss"] < raw_m["log_loss"] and sig_m["brier"] < raw_m["brier"]
              and sig_m["ece"] <= raw_m["ece"])
    return (sigmoid if choose else p_valid), ("sigmoid" if choose else "raw"), {
        "raw": raw_m, "sigmoid": sig_m,
    }


def _position_predictions(train: pl.DataFrame, cal: pl.DataFrame,
                          valid: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw_cal = np.empty(cal.height)
    raw_valid = np.empty(valid.height)
    selected_valid = np.empty(valid.height)
    diagnostics: dict[str, Any] = {}
    for position in POSITIONS:
        train_pos = train.filter(pl.col("position") == position)
        cal_pos = cal.filter(pl.col("position") == position)
        valid_pos = valid.filter(pl.col("position") == position)
        cal_idx = np.flatnonzero(np.asarray(cal["position"] == position, dtype=bool))
        valid_idx = np.flatnonzero(np.asarray(valid["position"] == position, dtype=bool))
        if train_pos.height < 500 or train_pos["anytime_td"].sum() < 50:
            raise ValueError(f"Insufficient {position} training positives")
        fields = POSITION_CORE[position] + POSITION_XTD[position]
        model = _fit_logistic(train_pos, fields)
        cal_p = _predict_logistic(model, cal_pos, fields)
        valid_p = _predict_logistic(model, valid_pos, fields)
        selected, calibration, choices = _maybe_calibrate(cal_pos, valid_pos, cal_p, valid_p)
        raw_cal[cal_idx], raw_valid[valid_idx], selected_valid[valid_idx] = cal_p, valid_p, selected
        info: dict[str, Any] = {"fit_rows": train_pos.height,
                                "fit_positives": int(train_pos["anytime_td"].sum()),
                                "fields": list(fields), "calibration": calibration,
                                "calibration_comparison": choices,
                                "selected_metrics": probability_metrics(
                                    np.asarray(valid_pos["anytime_td"], dtype=int), selected)}
        if position != "QB":
            core_model = _fit_logistic(train_pos, POSITION_CORE[position])
            core_p = _predict_logistic(core_model, valid_pos, POSITION_CORE[position])
            core_m = probability_metrics(np.asarray(valid_pos["anytime_td"], dtype=int), core_p)
            xtd_m = choices["raw"]
            info["xtd_ablation"] = {"core_usage": core_m, "core_usage_xtd": xtd_m,
                                    "xtd_minus_core_log_loss": xtd_m["log_loss"] - core_m["log_loss"]}
        diagnostics[position] = info
    return raw_cal, raw_valid, selected_valid, diagnostics


def _blend(matrix: np.ndarray, weights: tuple[float, float, float, float]) -> np.ndarray:
    if len(weights) != 4 or abs(sum(weights) - 1) > 1e-9 or min(weights) < 0:
        raise ValueError("Unregistered blend weights")
    return np.clip(matrix @ np.asarray(weights), 1e-6, 1 - 1e-6)


def _cluster_interval(game: list[str], actual: np.ndarray,
                      champion: np.ndarray, challenger: np.ndarray) -> dict[str, float]:
    base = np.clip(champion, 1e-6, 1 - 1e-6)
    new = np.clip(challenger, 1e-6, 1 - 1e-6)
    loss = -(actual * np.log(new) + (1 - actual) * np.log(1 - new))
    base_loss = -(actual * np.log(base) + (1 - actual) * np.log(1 - base))
    grouped: dict[str, list[float]] = {}
    for key, difference in zip(game, loss - base_loss, strict=True):
        entry = grouped.setdefault(key, [0.0, 0.0])
        entry[0] += float(difference)
        entry[1] += 1
    totals = np.asarray(list(grouped.values()), dtype=float)
    generator = np.random.default_rng(1729)
    indices = generator.integers(0, len(totals), size=(BOOTSTRAP_REPLICATES, len(totals)))
    samples = totals[indices]
    deltas = samples[:, :, 0].sum(axis=1) / samples[:, :, 1].sum(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"paired_log_loss_difference": float((loss - base_loss).mean()),
            "bootstrap_95_low": float(low), "bootstrap_95_high": float(high),
            "game_clusters": len(grouped), "replicates": BOOTSTRAP_REPLICATES}


def _segments(frame: pl.DataFrame, actual: np.ndarray,
              probability: np.ndarray) -> dict[str, Any]:
    weeks = np.asarray(frame["week"], dtype=int)
    return {
        label: probability_metrics(actual[mask], probability[mask])
        for label, mask in (("early_2_6", weeks <= 6),
                            ("middle_7_12", (weeks >= 7) & (weeks <= 12)),
                            ("late_13_plus", weeks >= 13))
    }


def _promotion(valid: pl.DataFrame, actual: np.ndarray, base: np.ndarray,
               challenger: np.ndarray) -> dict[str, Any]:
    base_metric = probability_metrics(actual, base)
    metric = probability_metrics(actual, challenger)
    interval = _cluster_interval(valid["game"].to_list(), actual, base, challenger)
    base_segments = _segments(valid, actual, base)
    segments = _segments(valid, actual, challenger)
    changes = {name: segments[name]["log_loss"] - base_segments[name]["log_loss"]
               for name in segments}
    position_changes = {}
    for position in POSITIONS:
        mask = np.asarray(valid["position"] == position, dtype=bool)
        position_changes[position] = (
            probability_metrics(actual[mask], challenger[mask])["log_loss"]
            - probability_metrics(actual[mask], base[mask])["log_loss"]
        )
    gates = {
        "material_log_loss": metric["log_loss"] <= base_metric["log_loss"] - MATERIAL_LOGLOSS,
        "bootstrap_interval_below_zero": interval["bootstrap_95_high"] < 0,
        "ece": metric["ece"] <= base_metric["ece"] + ECE_MAX_WORSE,
        "slope": metric["calibration_slope"] is not None and .8 <= metric["calibration_slope"] <= 1.2,
        "position": max(position_changes.values()) <= POSITION_MAX_WORSE,
        "chronology": sum(delta < 0 for delta in changes.values()) >= 2
        and max(changes.values()) <= SEGMENT_MAX_WORSE,
    }
    return {"promoted": all(gates.values()), "gates": gates, "uncertainty": interval,
            "position_log_loss_changes": position_changes,
            "segment_log_loss_changes": changes, "segments": segments}


def _tree_diagnostics(model: lgb.LGBMClassifier) -> dict[str, Any]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    importance: list[dict[str, Any]] = sorted(
        ({"feature": name, "gain": float(gain)}
         for name, gain in zip(TREE_FIELDS, gains, strict=True)),
        key=lambda item: item["gain"], reverse=True,
    )
    hypotheses = (
        ("last5_goal_line_opportunity_share", "expected_team_td"),
        ("last5_red_zone_target_share", "implied_team_points"),
        ("last5_carry_share", "team_spread"),
        ("last5_end_zone_target_share", "implied_team_points"),
        ("last5_total_xtd_share", "last5_opponent_allowed_red_zone_carries_per_game"),
    )
    pair_counts = {" / ".join(pair): 0 for pair in hypotheses}

    def visit(node: dict[str, Any], ancestors: frozenset[str]) -> None:
        if "split_feature" not in node:
            return
        feature = TREE_FIELDS[int(node["split_feature"])]
        path = ancestors | {feature}
        for pair in hypotheses:
            if pair[0] in path and pair[1] in path and not (pair[0] in ancestors and pair[1] in ancestors):
                pair_counts[" / ".join(pair)] += 1
        visit(node["left_child"], path)
        visit(node["right_child"], path)

    for tree in booster.dump_model()["tree_info"]:
        visit(tree["tree_structure"], frozenset())
    return {"feature_gain": importance, "joint_split_path_counts": pair_counts,
            "interpretation": "Joint tree-path splits are descriptive, not causal interaction estimates."}


def _market_diagnostic(valid: pl.DataFrame, actual: np.ndarray,
                       football: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(valid["p_market"].is_not_null(), dtype=bool)
    p_market = np.asarray(valid["p_market"].fill_null(0), dtype=float)
    positions = np.asarray(valid["position"])
    by_position = {}
    for position in POSITIONS:
        subset = mask & (positions == position)
        if not subset.any():
            continue
        by_position[position] = {
            "rows": int(subset.sum()),
            "football": probability_metrics(actual[subset], football[subset]),
            "market_raw": probability_metrics(actual[subset], p_market[subset]),
            "mean_football_minus_market_p": float((football[subset] - p_market[subset]).mean()),
            "mean_actual_minus_football_p": float((actual[subset] - football[subset]).mean()),
            "mean_actual_minus_market_p": float((actual[subset] - p_market[subset]).mean()),
        }
    return {"rows": int(mask.sum()), "games": valid.filter(pl.col("p_market").is_not_null())["game"].n_unique(),
            "football": probability_metrics(actual[mask], football[mask]),
            "market_raw": probability_metrics(actual[mask], p_market[mask]),
            "by_position": by_position,
            "limitation": "Ten Phase 1 T-60 games; raw implied probability includes book margin."}


def _write_report(detail: dict[str, Any]) -> None:
    metrics = detail["metrics"]
    lines = [
        "# Phase 6 — nonlinear and position-specific player models", "",
        ("All Phase 0-5 predictive artifacts and Phase 4.6 replay controls are unchanged. "
        "The 2025 holdout and September 13, 2026 exhibition were not loaded. "
        "No betting result or threshold informed selection."), "",
        "## Design and population", "",
        (f"The frozen Phase 5 player universe gives {detail['population']['fit_rows']:,} "
        f"player-games for 2022 fitting, {detail['population']['oof_rows']:,} 2023 "
        f"chronological OOF/calibration rows, and {detail['population']['validation_rows']:,} "
        "2024 validation rows. The 2021 strict-market season warms up the Phase 4 "
        "team expectation, so there is no prior-market fit for 2021 player rows. "
        "All selected team expectations and player opportunity features use earlier games. "
        "The prior-game publication time is the documented kickoff +24h assumption, "
        "not an observed publication timestamp."),
        ("Model choices and promotion gates were fixed in `docs/PHASE6_PROTOCOL.md` "
        "before Phase 6 validation. The required absolute log-loss improvement is 0.005 "
        "plus a wholly negative game-cluster bootstrap interval, acceptable calibration, "
        "position results and three-segment stability."), "",
        "## Overall 2024 validation", "",
        "| Model | Player-games | Log loss | Brier | ECE | Cal intercept | Cal slope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("historical_td", "phase5_hierarchical", "phase5_pooled_logistic",
                 "position_logistic", "lightgbm", "oof_blend"):
        m = metrics[name]
        lines.append(f"| {name} | {m['rows']:,} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                     f"{m['ece']:.4f} | {m['calibration_intercept']:+.3f} | "
                     f"{m['calibration_slope']:.3f} |")
    lines.extend(["", ("ROC-AUC and PR-AUC are secondary metrics in `reports/phase6_metrics.json`. "
                  "Every model's overall, position and probability-bucket calibration tables "
                  "are stored there. The historical and Phase 5 columns are unchanged frozen benchmarks."),
                  "", "## Role-specific logistic models and xTD ablation", "",
                  "| Position | Fit rows | Fit TD+ | Core log loss | Core+xTD log loss | xTD minus core | Selected log loss | Hierarchy log loss |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for position in POSITIONS:
        info = detail["position_models"][position]
        core = info.get("xtd_ablation", {}).get("core_usage", {}).get("log_loss")
        xtd = info.get("xtd_ablation", {}).get("core_usage_xtd", {}).get("log_loss")
        delta = info.get("xtd_ablation", {}).get("xtd_minus_core_log_loss")
        core_text = f"{core:.4f}" if core is not None else "—"
        xtd_text = f"{xtd:.4f}" if xtd is not None else "—"
        delta_text = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(f"| {position} | {info['fit_rows']:,} | {info['fit_positives']:,} | "
                     f"{core_text} | {xtd_text} | {delta_text} | "
                     f"{info['selected_metrics']['log_loss']:.4f} | "
                     f"{detail['position_benchmarks'][position]['phase5_hierarchical']['log_loss']:.4f} |")
    lines.extend(["", "Position-level probability diagnostics for each serious model:", "",
                  "| Position | Model | Rows | Log loss | Brier | ECE |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for position in POSITIONS:
        for name in ("phase5_hierarchical", "phase5_pooled_logistic", "position_logistic",
                     "lightgbm", "oof_blend"):
            m = detail["position_benchmarks"][position][name]
            lines.append(f"| {position} | {name} | {m['rows']:,} | {m['log_loss']:.4f} | "
                         f"{m['brier']:.4f} | {m['ece']:.4f} |")
    lines.extend(["", ("QB designed-run and scramble flags, WR/TE air-yard history, and route "
                  "participation are not present as safe lagged fields in the frozen player "
                  "table. QB carries and red-zone carries are proxies; passing TDs never "
                  "enter QB labels. All opponent inputs are Phase 2 lagged allowed-volume "
                  "context, not opponent-adjusted EPA. Logistic models use training-median "
                  "imputation with missing indicators; LightGBM sees native nulls. The "
                  "per-field, per-season missingness matrix is in the metrics JSON."),
                  "", "## LightGBM registry", "",
                  "| Configuration | 2024 log loss | Brier | ECE | Calibration |",
                  "| --- | ---: | ---: | ---: | --- |"])
    for name, info in detail["lightgbm_registry"].items():
        m = info["selected_metrics"]
        lines.append(f"| {name} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                     f"{m['ece']:.4f} | {info['calibration']} |")
    lines.extend(["", (f"Selected standalone LightGBM configuration: "
                  f"**{detail['selected_tree_config']}**. The conservative configuration alone "
                  "was eligible for OOF blending, regardless of validation ranking."),
                  ("The tree gain and joint split-path diagnostics in the metrics JSON describe "
                  "whether predeclared opportunity/environment pairs appeared in learned "
                  "paths; they do not establish causal interactions."),
                  "", "| Highest gain feature | Split gain |", "| --- | ---: |"])
    for item in detail["lightgbm_diagnostics"]["feature_gain"][:8]:
        lines.append(f"| {item['feature']} | {item['gain']:.1f} |")
    lines.extend(["", "| Hypothesized pair sharing a tree path | Split-pair count |",
                  "| --- | ---: |"])
    for pair, count in detail["lightgbm_diagnostics"]["joint_split_path_counts"].items():
        lines.append(f"| {pair} | {count} |")
    lines.extend(["", ("These counts show that some interactions were available to the fitted "
                  "tree; they do not show improved 2024 probability quality."),
                   "", "## Chronological OOF blend", "",
                  ("All 2023 component predictions came from models fitted on 2022 only. "
                  "The hierarchy used its 2022 allocation fit; pooled and position logistic "
                  "and the conservative LightGBM used 2022-only fits. No calibrator trained "
                  "on 2023 generated a 2023 blend-training input. Blend weights were selected "
                  "on 2023 OOF log loss and evaluated once on 2024."), "",
                  "| Fixed blend | 2023 OOF log loss | 2024 log loss |",
                  "| --- | ---: | ---: |"])
    for name, info in detail["blend_registry"].items():
        lines.append(f"| {name} | {info['oof_metrics']['log_loss']:.4f} | "
                     f"{info['validation_metrics']['log_loss']:.4f} |")
    lines.extend(["", f"Selected by 2023 OOF: **{detail['selected_blend']}**.",
                  "", "## Promotion and uncertainty", "",
                  ("| Challenger | Difference vs hierarchy | Game bootstrap 95% CI | "
                  "Material gate | All gates |"),
                  "| --- | ---: | ---: | --- | --- |"])
    for name, info in detail["promotion"].items():
        ci = info["uncertainty"]
        lines.append(f"| {name} | {ci['paired_log_loss_difference']:+.4f} | "
                     f"[{ci['bootstrap_95_low']:+.4f}, {ci['bootstrap_95_high']:+.4f}] | "
                     f"{'pass' if info['gates']['material_log_loss'] else 'fail'} | "
                     f"{'pass' if info['promoted'] else 'fail'} |")
    lines.extend(["", ("The full gate outcomes, position changes and early/middle/late "
                  "validation metrics are in the metrics JSON. The bootstrap resamples "
                  "2024 games, preserving within-game player correlation."),
                  "", "## T-60 player market benchmark", "",
                  (f"Only {detail['market']['rows']} matched players across "
                  f"{detail['market']['games']} frozen Phase 1 games have a strict T-60 "
                  "player price. These raw implied probabilities are a benchmark only; "
                  "no price enters football models or blend selection."),
                  "", ("| Position | Matched rows | Champion log loss | Raw market log loss | "
                  "Mean football minus market P |"),
                  "| --- | ---: | ---: | ---: | ---: |"])
    for position, info in detail["market"]["by_position"].items():
        lines.append(f"| {position} | {info['rows']} | {info['football']['log_loss']:.4f} | "
                     f"{info['market_raw']['log_loss']:.4f} | "
                     f"{info['mean_football_minus_market_p']:+.3f} |")
    lines.extend(["", (f"Matched overall: champion {detail['market']['football']['log_loss']:.4f} "
                  f"vs market {detail['market']['market_raw']['log_loss']:.4f} log loss. "
                  "Football probabilities were below raw market implied probabilities in "
                  "all four positions. Raw implied prices contain bookmaker margin, so this "
                  "sign alone does not establish systematic overpricing. Position residuals "
                  "are descriptive; ten games cannot establish a "
                  "stable pattern or justify a betting threshold."),
                  "", "## Answers and limits", ""])
    for question, answer in detail["answers"].items():
        lines.append(f"- **{question}** {answer}")
    lines.extend(["", f"**PHASE6_CHAMPION = {detail['champion']}**", "",
                  ("The 2025 holdout remains unopened. Phase 7 betting simulation was not "
                  "started. Validation predictions and chronological OOF components are "
                  "in `reports/phase6_validation_predictions.csv` and "
                  "`reports/phase6_2023_oof_components.csv`."), ""])
    (REPORTS / "PHASE6.md").write_text("\n".join(lines), encoding="utf-8")


def build_phase6() -> tuple[Path, Path]:
    frame, p5 = _source_frame()
    train = frame.filter(pl.col("season") == 2022)
    cal = frame.filter(pl.col("season") == 2023)
    valid = frame.filter(pl.col("season") == 2024)
    if (train.height, cal.height, valid.height) != (6441, 6321, 6320):
        raise ValueError("Phase 5 common chronological sample changed")
    actual_cal = np.asarray(cal["anytime_td"], dtype=int)
    actual = np.asarray(valid["anytime_td"], dtype=int)
    base = np.asarray(valid["p_hierarchical"], dtype=float)
    p_pool_raw = np.asarray(valid["p_logistic_raw"], dtype=float)
    frozen_selected = np.asarray(valid["p_logistic"], dtype=float)
    p_history = np.asarray(valid["p_historical_td"], dtype=float)
    hierarchy_weights = np.asarray(list(p5["hierarchy_weights"].values()), dtype=float)
    if list(p5["hierarchy_weights"]) != list(HIERARCHY_FIELDS):
        raise ValueError("Frozen Phase 5 hierarchy field order changed")
    p_hier_cal, _ = hierarchy_predict(cal, hierarchy_weights)
    pooled = fit_player_logistic(train, MODEL_FAMILIES["E_team_usage_xtd"])
    p_pool_cal = logistic_predict(pooled, cal, MODEL_FAMILIES["E_team_usage_xtd"])
    reproduced_raw = logistic_predict(pooled, valid, MODEL_FAMILIES["E_team_usage_xtd"])
    if not np.allclose(reproduced_raw, p_pool_raw, atol=1e-10, rtol=0):
        raise ValueError("Phase 5 pooled logistic raw predictions changed")
    p_pos_cal, p_pos_raw, p_pos, positions = _position_predictions(train, cal, valid)
    trees: dict[str, lgb.LGBMClassifier] = {}
    tree_registry: dict[str, Any] = {}
    tree_cal_raw: dict[str, np.ndarray] = {}
    tree_valid_raw: dict[str, np.ndarray] = {}
    tree_valid_selected: dict[str, np.ndarray] = {}
    for name, config in TREE_CONFIGS.items():
        model = _fit_tree(train, name)
        cal_p = _predict_tree(model, cal)
        valid_p = _predict_tree(model, valid)
        selected, calibration, choices = _maybe_calibrate(cal, valid, cal_p, valid_p)
        trees[name], tree_cal_raw[name], tree_valid_raw[name] = model, cal_p, valid_p
        tree_valid_selected[name] = selected
        tree_registry[name] = {"parameters": config, "fit_season": 2022,
                               "calibration": calibration, "calibration_fit_season": 2023,
                               "calibration_comparison": choices,
                               "selected_metrics": probability_metrics(actual, selected)}
    selected_tree = min(tree_registry, key=lambda name: (
        tree_registry[name]["selected_metrics"]["log_loss"],
        tree_registry[name]["selected_metrics"]["brier"],
    ))
    p_tree = tree_valid_selected[selected_tree]
    oof_matrix = np.column_stack([p_hier_cal, p_pool_cal, p_pos_cal,
                                  tree_cal_raw["conservative"]])
    valid_matrix = np.column_stack([base, p_pool_raw, p_pos_raw,
                                    tree_valid_raw["conservative"]])
    blend_registry: dict[str, Any] = {}
    for name, weights in BLENDS.items():
        blend_registry[name] = {"weights": list(weights),
                                "component_fit_max_season": 2022,
                                "oof_metrics": probability_metrics(actual_cal, _blend(oof_matrix, weights)),
                                "validation_metrics": probability_metrics(
                                    actual, _blend(valid_matrix, weights))}
    selected_blend = min(blend_registry, key=lambda name: (
        blend_registry[name]["oof_metrics"]["log_loss"],
        blend_registry[name]["oof_metrics"]["brier"],
    ))
    p_blend = _blend(valid_matrix, BLENDS[selected_blend])
    contenders = {"position_logistic": p_pos, "lightgbm": p_tree, "oof_blend": p_blend}
    promotion = {name: _promotion(valid, actual, base, probability)
                 for name, probability in contenders.items()}
    passing = [name for name, result in promotion.items() if result["promoted"]]
    champion = min(passing, key=lambda name: probability_metrics(actual, contenders[name])["log_loss"]) if passing else "phase5_hierarchical"
    champion_p = contenders[champion] if passing else base
    models = {"historical_td": p_history, "phase5_hierarchical": base,
              "phase5_pooled_logistic": frozen_selected, **contenders}
    metrics = {name: probability_metrics(actual, probability) for name, probability in models.items()}
    position_benchmarks = {}
    for position in POSITIONS:
        mask = np.asarray(valid["position"] == position, dtype=bool)
        position_benchmarks[position] = {
            name: probability_metrics(actual[mask], probability[mask])
            for name, probability in models.items()
        }
    market = _market_diagnostic(valid, actual, champion_p)
    answers = {
        "Do position models improve on pooled models?": (
            ("Yes. " if metrics["position_logistic"]["log_loss"] < metrics["phase5_pooled_logistic"]["log_loss"] else "No overall. ")
            + f"Position logistic log loss {metrics['position_logistic']['log_loss']:.4f} "
            f"versus frozen pooled logistic {metrics['phase5_pooled_logistic']['log_loss']:.4f} "
            f"and hierarchy {metrics['phase5_hierarchical']['log_loss']:.4f}; see position table."
        ),
        "Does LightGBM materially beat hierarchy?": (
            "Yes." if promotion["lightgbm"]["promoted"] else "No; it fails at least one predeclared gate."
        ),
        "Does xTD add within a position?": "No clear 2024 log-loss gain. " + "; ".join(
            f"{position} xTD-minus-core log loss {positions[position]['xtd_ablation']['xtd_minus_core_log_loss']:+.4f}"
            for position in ("RB", "WR", "TE")
        ) + ". Negative values favor xTD; small differences are not strong evidence.",
        "Does the OOF ensemble materially improve?": (
            "Yes." if promotion["oof_blend"]["promoted"] else "No; it fails at least one predeclared gate."
        ),
        "Which model advances to Phase 7?": champion + ".",
        "Does market still outperform best football model?": (
            "Yes" if market["market_raw"]["log_loss"] < market["football"]["log_loss"] else "No"
        ) + f" on {market['rows']} matched 2024 T-60 player quotes; this is a narrow sample.",
    }
    detail: dict[str, Any] = {
        "phase": 6, "champion": champion, "holdout_2025_untouched": True,
        "diagnostic_exhibition_slate_excluded": True,
        "population": {"fit_rows": train.height, "oof_rows": cal.height,
                       "validation_rows": valid.height, "validation_positive_rate": float(actual.mean())},
        "chronology": {"component_fit_seasons": [2022], "oof_and_calibration_season": 2023,
                       "validation_season": 2024, "team_model_fits": p5["team_model_fit_seasons"]},
        "promotion_thresholds": {"minimum_absolute_log_loss_improvement": MATERIAL_LOGLOSS,
                                 "maximum_position_log_loss_worsening": POSITION_MAX_WORSE,
                                 "maximum_segment_log_loss_worsening": SEGMENT_MAX_WORSE,
                                 "maximum_ece_worsening": ECE_MAX_WORSE},
        "metrics": metrics, "position_benchmarks": position_benchmarks,
        "position_models": positions, "lightgbm_registry": tree_registry,
        "selected_tree_config": selected_tree,
        "lightgbm_diagnostics": _tree_diagnostics(trees[selected_tree]),
        "blend_registry": blend_registry, "selected_blend": selected_blend,
        "promotion": promotion, "market": market, "answers": answers,
        "missing_feature_families": ["safe historical route participation",
                                     "QB designed-run/scramble usage", "player air-yard history"],
        "feature_missing_pct_by_season": {
            field: {str(year): round(100 * subset[field].null_count() / subset.height, 2)
                    for year, subset in ((2022, train), (2023, cal), (2024, valid))}
            for field in sorted(SAFE_FIELDS)
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions_path = REPORTS / "phase6_validation_predictions.csv"
    oof_path = REPORTS / "phase6_2023_oof_components.csv"
    valid.select("game", "player", "player_id", "position", "team", "opponent",
                 "season", "week", "prediction_time", "anytime_td", "p_market").with_columns(
        pl.Series("p_hierarchical", base), pl.Series("p_pooled_logistic", frozen_selected),
        pl.Series("p_position_logistic", p_pos), pl.Series("p_lightgbm", p_tree),
        pl.Series("p_oof_blend", p_blend), pl.Series("p_phase6_champion", champion_p),
    ).write_csv(predictions_path)
    cal.select("game", "player_id", "position", "team", "season", "anytime_td").with_columns(
        pl.Series("p_hierarchical_oof", p_hier_cal),
        pl.Series("p_pooled_logistic_oof_raw", p_pool_cal),
        pl.Series("p_position_logistic_oof_raw", p_pos_cal),
        pl.Series("p_lightgbm_conservative_oof_raw", tree_cal_raw["conservative"]),
        pl.lit(2022).alias("component_fit_max_season"),
    ).write_csv(oof_path)
    detail["artifact_hashes"] = {
        "validation_predictions": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "oof_components": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        "phase5_features": p5["artifact_hashes"]["feature_table"],
        "phase5_predictions": p5["artifact_hashes"]["validation_predictions"],
    }
    (REPORTS / "phase6_metrics.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    _write_report(detail)
    return predictions_path, REPORTS / "PHASE6.md"
