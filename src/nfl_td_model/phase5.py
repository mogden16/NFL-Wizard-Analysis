"""Phase 5 chronological player anytime-TD probability baselines.

Only 2017-2024 frozen sources are read. 2025 and the 2026 exhibition are
excluded by filename, season guards, and fitting/calibration API checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from scipy.optimize import minimize  # type: ignore[import-untyped]
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

from nfl_td_model.config import Settings
from nfl_td_model.market_math import american_to_decimal
from nfl_td_model.odds import extract_market_rows, parse_time
from nfl_td_model.phase1 import normalize_name
from nfl_td_model.phase4_models import MARKET, fit_count_model

DERIVED = Path("data/derived")
REPORTS = Path("reports")
POSITIONS = ("RB", "WR", "TE", "QB")
HIERARCHY_FIELDS = (
    "last5_total_xtd_share", "last5_rushing_xtd_share", "last5_receiving_xtd_share",
    "last5_goal_line_opportunity_share", "last5_red_zone_target_share",
    "last5_end_zone_target_share", "last5_carry_share", "last5_target_share",
    "last5_snap_share",
)
USAGE = (
    "last5_carry_share", "last5_target_share", "last5_touch_share",
    "last5_snap_share", "last5_goal_line_opportunity_share",
    "last5_inside_10_carries_per_game", "last5_inside_5_carries_per_game",
    "last5_red_zone_target_share", "last5_end_zone_target_share",
)
XTD = (
    "last5_rushing_xtd", "last5_receiving_xtd", "last5_total_xtd",
    "last5_total_xtd_per_opportunity", "last5_rushing_xtd_share",
    "last5_receiving_xtd_share", "last5_total_xtd_share",
)
POSITION_DUMMIES = ("is_rb", "is_wr", "is_te", "is_qb")
MODEL_FAMILIES = {
    "A_actual_td_only": ("last8_td_rate",) + POSITION_DUMMIES,
    "B_xtd_only": XTD + POSITION_DUMMIES,
    "C_usage_actual_td": USAGE + ("last8_td_rate",) + POSITION_DUMMIES,
    "D_usage_xtd": USAGE + XTD + POSITION_DUMMIES,
    "E_team_usage_xtd": ("expected_team_td", "implied_team_points", "home")
    + USAGE + XTD + POSITION_DUMMIES,
    "F_team_usage_xtd_actual_td": ("expected_team_td", "implied_team_points", "home")
    + USAGE + XTD + ("last8_td_rate",) + POSITION_DUMMIES,
}
SAFE_MODEL_FIELDS = frozenset(field for family in MODEL_FAMILIES.values() for field in family)
CAL_BINS = (("<10%", 0, .1), ("10-20%", .1, .2), ("20-30%", .2, .3),
            ("30-40%", .3, .4), ("40-50%", .4, .5), ("50-60%", .5, .6),
            ("60%+", .6, 1.01))


def _safe_season_file(season: int) -> Path:
    if not 2017 <= season <= 2024:
        raise ValueError("Phase 5 may read only 2017-2024 player features")
    return DERIVED / f"phase2_{season}_player_features.parquet"


def _candidate_universe() -> pl.DataFrame:
    """Use prior same-team appearances, never current-game participant status."""
    seasons = []
    for season in range(2017, 2025):
        frame = pl.read_parquet(_safe_season_file(season)).with_columns(
            pl.lit(season).alias("season"),
            pl.col("game").str.slice(5, 2).cast(pl.Int32).alias("week"),
        )
        frame = frame.with_columns(
            pl.col("player_source_game_ids").str.split(";").list.last().alias("last_prior_player_game"),
            pl.col("team_source_game_ids").str.split(";").list.tail(3).alias("last3_team_games"),
        )
        role = (
            (pl.col("last3_carries_per_game").fill_null(0)
             + pl.col("last3_targets_per_game").fill_null(0) > 0)
            | ((pl.col("position") == "QB") & (pl.col("last3_snaps_per_game").fill_null(0) >= 20))
        )
        frame = frame.filter(
            pl.col("position").is_in(POSITIONS)
            & pl.col("last3_team_games").list.contains(pl.col("last_prior_player_game"))
            & role
        )
        seasons.append(frame.drop("last3_team_games"))
    return pl.concat(seasons).sort("season", "game", "team", "player_id")


def historical_td_features(row: dict[str, Any], td_by_game_player: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Historical scoring rates use only the Phase 2 audited prior-game IDs."""
    ids = [game for game in row["player_source_game_ids"].split(";") if game]
    if row["game"] in ids:
        raise ValueError("Current game entered historical TD source list")
    output: dict[str, Any] = {}
    for window, recent in (("season", ids), ("last5", ids[-5:]), ("last8", ids[-8:])):
        outcomes = [int(td_by_game_player.get((game, row["player_id"]), 0) > 0) for game in recent]
        output[f"{window}_td_games"] = len(outcomes)
        output[f"{window}_td_wins"] = sum(outcomes)
        output[f"{window}_td_rate"] = sum(outcomes) / len(outcomes) if outcomes else None
    recent = ids[-8:]
    values = [int(td_by_game_player.get((game, row["player_id"]), 0) > 0) for game in recent]
    weights = [0.5 ** ((len(values) - i - 1) / 3) for i in range(len(values))]
    output["ewma_td_games"] = len(values)
    output["ewma_td_wins_weighted"] = sum(v * w for v, w in zip(values, weights, strict=True))
    output["ewma_td_weight"] = sum(weights)
    output["ewma_td_rate"] = output["ewma_td_wins_weighted"] / sum(weights) if weights else None
    return output


def _td_labels() -> dict[tuple[str, str], int]:
    opportunities = pl.read_parquet(
        DERIVED / "phase3_2017_2024_opportunities.parquet",
        columns=["season", "game", "player_id", "actual_td"],
    )
    if opportunities["season"].max() != 2024:
        raise ValueError("Unexpected opportunity label seasons")
    frame = opportunities.group_by("game", "player_id").agg(pl.col("actual_td").sum())
    return {(r["game"], r["player_id"]): r["actual_td"] for r in frame.to_dicts()}


def _market_benchmark() -> dict[tuple[str, str], dict[str, Any]]:
    """Use only the ten already-frozen Phase 1 event snapshots; no new credits."""
    with (REPORTS / "phase1_2024_week4_audit.csv").open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    event_games = {row["odds_event_id"]: (row["game"], row["prediction_time"]) for row in audit}
    hashes = json.loads((REPORTS / "phase1_odds_hashes.json").read_text(encoding="utf-8"))
    benchmark: dict[tuple[str, str], dict[str, Any]] = {}
    for event_id, (game, cutoff_text) in event_games.items():
        path = Path("data/raw/the_odds_api") / f"{hashes[event_id]}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[event_id]:
            raise ValueError("Frozen Phase 1 market snapshot was modified")
        cutoff = parse_time(cutoff_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        prices: dict[str, list[int]] = defaultdict(list)
        for quote in extract_market_rows(payload, cutoff):
            if quote["market"] == "player_anytime_td" and quote["name"] == "Yes":
                prices[normalize_name(str(quote["description"]))].append(int(quote["price"]))
        for name, odds in prices.items():
            decimals = sorted(american_to_decimal(price) for price in odds)
            median_decimal = float(np.median(decimals))
            benchmark[(game, name)] = {
                "p_market": 1 / median_decimal, "market_books": len(odds),
                "market_median_decimal": median_decimal,
            }
    return benchmark


def _out_of_sample_team_expectations(strict: pl.DataFrame) -> pl.DataFrame:
    """Score each team season with the frozen Phase 4 architecture fit earlier."""
    scored_seasons = []
    for score_season in (2022, 2023, 2024):
        prior = strict.filter(pl.col("season") < score_season)
        current = strict.filter(pl.col("season") == score_season)
        if prior.is_empty() or current.is_empty() or cast(int, prior["season"].max()) >= score_season:
            raise ValueError("Team environment fit is not strictly earlier than score season")
        model = fit_count_model(prior, MARKET, "poisson")
        scored_seasons.append(current.with_columns(
            pl.Series("expected_team_td", model.predict(current))
        ))
    return pl.concat(scored_seasons)


def assemble_frame() -> pl.DataFrame:
    """Join labels only after all feature sources have been selected and audited."""
    candidates = _candidate_universe()
    labels = _td_labels()
    rates = [historical_td_features(row, labels) for row in candidates.select(
        "game", "player_id", "player_source_game_ids"
    ).to_dicts()]
    candidates = pl.concat([candidates, pl.DataFrame(rates)], how="horizontal")
    xtd = pl.read_parquet(DERIVED / "phase3_2018_2024_lagged_xtd_features.parquet")
    keep = ["game", "team", "player_id", "xtd_source_game_ids", "xtd_history_games",
            "xtd_feature_available_at_max"] + list({*XTD, *HIERARCHY_FIELDS} & set(xtd.columns))
    candidates = candidates.join(xtd.select(keep), on=["game", "team", "player_id"], how="left")
    # Verify source ids and assumed publication cutoffs before any model matrix.
    for row in candidates.select(
        "game", "prediction_time", "feature_available_at_max", "xtd_source_game_ids",
        "xtd_feature_available_at_max"
    ).iter_rows(named=True):
        if row["feature_available_at_max"] is None or row["feature_available_at_max"] >= row["prediction_time"]:
            raise ValueError("Phase 2 feature availability crosses prediction time")
        if row["xtd_source_game_ids"] and row["game"] in row["xtd_source_game_ids"].split(";"):
            raise ValueError("Current-game xTD entered predictor")
        if row["xtd_feature_available_at_max"] and row["xtd_feature_available_at_max"] >= row["prediction_time"]:
            raise ValueError("Phase 3 xTD availability crosses prediction time")
    strict = pl.read_parquet(DERIVED / "phase4_strict_market_2021_2024.parquet")
    if strict["season"].max() != 2024:
        raise ValueError("2025 entered strict market source")
    # 2021 warms up the strict market model; no team expectation is in-sample.
    strict = _out_of_sample_team_expectations(strict)
    market_fields = strict.select(
        pl.col("game_id").alias("game"), "team", "implied_team_points", "team_spread",
        "game_total", "home", "expected_team_td", "odds_timestamp", "spread_quote_time",
        "total_quote_time",
    )
    candidates = candidates.join(market_fields, on=["game", "team"], how="left")
    benchmark = _market_benchmark()
    market_keys = ("p_market", "market_books", "market_median_decimal")
    extras = [
        {key: benchmark.get((row["game"], normalize_name(row["player"])), {}).get(key)
         for key in market_keys}
        for row in candidates.select("game", "player").to_dicts()
    ]
    candidates = pl.concat([candidates, pl.DataFrame(extras, infer_schema_length=None).select(
        pl.col("p_market").cast(pl.Float64), pl.col("market_books").cast(pl.Int64),
        pl.col("market_median_decimal").cast(pl.Float64)
    )], how="horizontal")
    candidates = candidates.with_columns(
        pl.Series("actual_td_count", [labels.get((r["game"], r["player_id"]), 0)
                                      for r in candidates.select("game", "player_id").to_dicts()]),
    ).with_columns((pl.col("actual_td_count") > 0).cast(pl.Int8).alias("anytime_td"))
    for position in POSITIONS:
        candidates = candidates.with_columns(
            (pl.col("position") == position).cast(pl.Int8).alias(f"is_{position.lower()}")
        )
    return candidates.sort("season", "game", "team", "player_id")


def guard_model_input(frame: pl.DataFrame, fields: tuple[str, ...]) -> None:
    """A single gate for fit, calibration, validation and live definitions."""
    if frame.is_empty() or cast(int, frame["season"].min()) < 2017 or cast(int, frame["season"].max()) > 2024:
        raise ValueError("Phase 5 model input crosses the 2025/2026 holdout firewall")
    if not fields or any(field not in SAFE_MODEL_FIELDS or field not in frame.columns for field in fields):
        raise ValueError("Only declared lagged football features may enter a Phase 5 model")


def _matrix(frame: pl.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    guard_model_input(frame, fields)
    return np.asarray(frame.select(fields).to_numpy(), dtype=float)


def fit_player_logistic(frame: pl.DataFrame, fields: tuple[str, ...]) -> Pipeline:
    if cast(int, frame["season"].max()) != 2022 or cast(int, frame["season"].min()) != 2022:
        raise ValueError("Phase 5 logistic fit must use 2022 only")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=1000, random_state=1729)),
    ])
    model.fit(_matrix(frame, fields), np.asarray(frame["anytime_td"], dtype=int))
    return model


def logistic_predict(model: Pipeline, frame: pl.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    return np.clip(np.asarray(model.predict_proba(_matrix(frame, fields))[:, 1], dtype=float), 1e-6, 1 - 1e-6)


def _hierarchy_inputs(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = _matrix(frame, HIERARCHY_FIELDS)
    available = np.isfinite(values).astype(float)
    values = np.nan_to_num(values, nan=0.0)
    codes: dict[tuple[str, str], int] = {}
    groups = []
    for row in frame.select("game", "team").iter_rows():
        key = cast(tuple[str, str], row)
        groups.append(codes.setdefault(key, len(codes)))
    expected = np.asarray(frame["expected_team_td"], dtype=float)
    if not np.isfinite(expected).all():
        raise ValueError("Hierarchical model requires strict Phase 4 team expectation")
    return values, available, np.asarray(groups, dtype=int), expected


def hierarchy_from_arrays(
    values: np.ndarray, available: np.ndarray, groups: np.ndarray,
    expected_team_td: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each team-game independently; missing families are omitted."""
    usable_weight = available @ weights
    raw = np.divide(values @ weights, usable_weight, out=np.ones(len(values)), where=usable_weight > 0)
    totals = np.bincount(groups, weights=raw, minlength=int(groups.max()) + 1)
    if np.any(totals == 0):
        raw[totals[groups] == 0] = 1.0
        totals = np.bincount(groups, weights=raw, minlength=len(totals))
    share = np.divide(raw, totals[groups], out=np.zeros(len(raw)), where=totals[groups] > 0)
    lam = expected_team_td * share
    return 1 - np.exp(-lam), lam


def hierarchy_predict(frame: pl.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return hierarchy_from_arrays(*_hierarchy_inputs(frame), weights)


def fit_hierarchy(frame: pl.DataFrame) -> np.ndarray:
    if cast(int, frame["season"].min()) != 2022 or cast(int, frame["season"].max()) != 2022:
        raise ValueError("Hierarchical allocation weights fit on 2022 only")
    inputs = _hierarchy_inputs(frame)
    labels = np.asarray(frame["anytime_td"], dtype=float)
    initial = np.full(len(HIERARCHY_FIELDS), 1 / len(HIERARCHY_FIELDS))

    def objective(weights: np.ndarray) -> float:
        probabilities, _ = hierarchy_from_arrays(*inputs, weights)
        return float(np.mean((probabilities - labels) ** 2))

    result = minimize(
        objective, initial, method="SLSQP", bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 100, "ftol": 1e-9},
    )
    if not result.success or np.any(result.x < -1e-8):
        raise RuntimeError(f"Hierarchical training did not converge: {result.message}")
    return np.clip(np.asarray(result.x, dtype=float), 0, 1) / np.clip(result.x.sum(), 1e-12, None)


def _probability_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    p = np.clip(predicted, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    diagnostic = LogisticRegression(C=1e6, max_iter=1000).fit(logit, actual)
    buckets = calibration_table(actual, p)
    ece = sum(b["player_games"] / len(actual) * abs(b["mean_probability"] - b["actual_td_rate"])
              for b in buckets if b["player_games"])
    return {
        "log_loss": float(log_loss(actual, p)), "brier": float(brier_score_loss(actual, p)),
        "calibration_intercept": float(diagnostic.intercept_[0]),
        "calibration_slope": float(diagnostic.coef_[0][0]),
        "ece": float(ece), "roc_auc": float(roc_auc_score(actual, p)),
        "pr_auc": float(average_precision_score(actual, p)),
    }


def calibration_table(actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for name, low, high in CAL_BINS:
        mask = (predicted >= low) & (predicted < high)
        output.append({
            "bin": name, "player_games": int(mask.sum()),
            "mean_probability": float(predicted[mask].mean()) if mask.any() else None,
            "actual_td_rate": float(actual[mask].mean()) if mask.any() else None,
        })
    return output


def _historical_probabilities(
    frame: pl.DataFrame, window: str, priors: dict[str, float], strength: float = 3.0,
) -> np.ndarray:
    rows = frame.select("position", f"{window}_td_games",
                        f"{window}_td_wins" if window != "ewma" else "ewma_td_wins_weighted",
                        *([] if window != "ewma" else ["ewma_td_weight"])).to_dicts()
    return np.asarray([
        (float(row[f"{window}_td_wins" if window != "ewma" else "ewma_td_wins_weighted"])
         + strength * priors[row["position"]])
        / (float(row[f"{window}_td_games" if window != "ewma" else "ewma_td_weight"]) + strength)
        for row in rows
    ], dtype=float)


def _coefficient_diagnostics(model: Pipeline, train: pl.DataFrame, fields: tuple[str, ...]) -> dict[str, Any]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    fitted = model.named_steps["model"]
    design = np.asarray(scaler.transform(imputer.transform(_matrix(train, fields))), dtype=float)
    names = list(imputer.get_feature_names_out(fields))
    prediction = np.asarray(fitted.predict_proba(design)[:, 1], dtype=float)
    augmented = np.column_stack([np.ones(len(design)), design])
    fisher = augmented.T @ (augmented * (prediction * (1 - prediction))[:, None])
    ridge = np.eye(fisher.shape[0]) / .5
    ridge[0, 0] = 0
    standard_errors = np.sqrt(np.maximum(np.diag(np.linalg.pinv(fisher + ridge)), 0))
    coefficients = [
        {"feature": name, "standardized_coefficient": float(coef),
         "approx_se": float(se), "approx_95pct_low": float(coef - 1.96 * se),
         "approx_95pct_high": float(coef + 1.96 * se)}
        for name, coef, se in zip(names, fitted.coef_[0], standard_errors[1:], strict=True)
    ]
    # Near-duplicate fields are reported rather than silently removed after validation.
    varying = np.std(design, axis=0) > 1e-8
    correlations = np.corrcoef(design[:, varying], rowvar=False)
    active_names = [name for name, keep in zip(names, varying, strict=True) if keep]
    pairs = sorted(
        ({"a": active_names[i], "b": active_names[j], "abs_correlation": float(abs(correlations[i, j]))}
         for i in range(len(active_names)) for j in range(i + 1, len(active_names))
         if abs(correlations[i, j]) >= .85),
        key=lambda row: row["abs_correlation"], reverse=True,
    )[:12]
    return {"coefficients": coefficients, "high_correlation_pairs": pairs,
            "uncertainty_note": "Approximate penalized Fisher SE; descriptive, not a causal interval."}


def _sigmoid_fit(calibration: pl.DataFrame, raw: np.ndarray) -> LogisticRegression:
    if calibration["season"].min() != 2023 or calibration["season"].max() != 2023:
        raise ValueError("Sigmoid calibration requires only 2023 observations")
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    fitted = LogisticRegression(C=1e6, max_iter=1000)
    fitted.fit(logits.reshape(-1, 1), np.asarray(calibration["anytime_td"], dtype=int))
    return fitted


def _sigmoid_predict(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    return np.asarray(calibrator.predict_proba(logits.reshape(-1, 1))[:, 1], dtype=float)


def _write_report(detail: dict[str, Any]) -> None:
    headline = detail["metrics"]
    lines = [
        "# Phase 5 — player anytime-touchdown probability baselines", "",
        "Phase 1-4.6 inputs and predictive architectures are frozen. No 2025 file or",
        "September 13, 2026 outcome/price/result was read. The 2026 exhibition",
        "remains excluded from all modeling decisions. This phase fits no tree or",
        "ensemble and performs no betting-return or threshold optimization.", "",
        "## Population and chronology", "",
        f"The prior-game-only universe contains **{detail['population']['all_2017_2024_rows']:,}** player-games in 2017-2024. It uses a player who appeared for the same team in one of the previous three team games and had a carry or target in that prior three-game window; a QB with at least 20 prior snaps also qualifies. No current-game participation, carries, targets, snaps or TD is used to choose a row. Rookie debuts and first-team games are consequently absent. RB, WR, TE and QB are included. FB is omitted because the frozen Phase 2 feature pipeline has no FB rows or comparable lagged red-zone/snap fields; adding an incomplete FB feature definition solely for this phase would make model comparisons inconsistent.",
        f"The common strict-market comparison contains **{detail['population']['fit_rows']:,}** 2022 base-fit rows, **{detail['population']['calibration_rows']:,}** 2023 calibration-only rows, and **{detail['population']['validation_rows']:,}** 2024 validation rows (TD rate **{detail['population']['validation_positive_rate']:.2%}**). 2017-2021 contributes to historical scoring priors and coverage diagnostics. Phase 3 chronological xTD is unavailable in 2017; strict T-60 team market inputs begin in 2021, which is a warmup season for the Phase 4 team market model. Team expectations are scored out of sample with an expanding prior-season fit: 2021 for 2022, 2021-2022 for 2023, and 2021-2023 for 2024. The 2025 holdout is not loaded.",
        "Historical prior-game statistics retain the **assumed kickoff + 24h** availability proxy. It is not an observed publication timestamp. Phase 3 xTD was scored with models fit on earlier seasons, and the current game's xTD is excluded.", "",
        "## 2024 common-row validation", "",
        "| Probability | Player-games | Log loss | Brier | ECE | Cal intercept | Cal slope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("historical_td", "hierarchical", "logistic_raw", "logistic_selected"):
        metric = headline[name]
        lines.append(f"| {name} | {metric['rows']:,} | {metric['log_loss']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} | {metric['calibration_intercept']:+.3f} | {metric['calibration_slope']:.3f} |")
    lines.extend(["", f"Historical TD rate uses the {detail['selected_historical_window']} candidate selected on 2024 validation. The logistic calibration choice is {detail['calibration']['selected']}; its sigmoid was fitted only to 2023 base-model predictions, then compared with raw probabilities on 2024. ROC-AUC and PR-AUC, which are secondary, and full metrics for all four TD-rate windows are in the metrics JSON."])
    lines.extend(["", "The 2024 market comparison is limited to **"
                  + str(detail["market_benchmark"]["matched_players"])
                  + "** player-games across ten frozen Phase 1 week-four games. No new historical-odds credits were spent. These already-captured T-60 quotes are a separate raw implied-probability benchmark, never a football-model input; coverage is too small for empirical market calibration.",
                  "", "| Model on quoted subset | Log loss | Brier | ECE |", "| --- | ---: | ---: | ---: | ---: |"])
    for name, metric in detail["market_benchmark"]["metrics"].items():
        lines.append(f"| {name} | {metric['log_loss']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |")
    lines.extend(["", "## Controlled logistic feature-family tests", "",
                  "All six models use identical 2024 rows, the same 2022 base-fit period and the same fixed logistic regularization. The 2023 rows are reserved for calibration; feature-family results below use raw 2024 predictions. The reduced base-fit period is necessary because 2021 has no prior strict-market season from which to generate an out-of-sample team expectation.",
                  "", "| Family | Log loss | Brier | ECE |", "| --- | ---: | ---: | ---: | ---: |"])
    for name, metric in detail["controlled_comparisons"].items():
        lines.append(f"| {name} | {metric['log_loss']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |")
    lines.extend(["", "Selected logistic stability within 2024:", "",
                  "| Segment | Log loss | Brier | ECE |", "| --- | ---: | ---: | ---: |"])
    for name, metric in detail["stability"].items():
        lines.append(f"| {name} | {metric['log_loss']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |")
    lines.extend(["", "## Acceptance answers", ""])
    for question, answer in detail["answers"].items():
        lines.append(f"- **{question}** {answer}")
    lines.extend(["", "## Hierarchical allocation and team coherence", "",
                  "Nonnegative allocation weights were fit to 2022 player TD labels by training Brier loss. The fixed feature families were declared before fitting. The weighted role score omits unavailable families, then each team-game normalizes player shares to one. No September 13 result or Phase 4.5 weight was used.",
                  "", "| Lagged feature | Fitted weight |", "| --- | ---: |"])
    for key, weight in detail["hierarchy_weights"].items():
        lines.append(f"| {key} | {weight:.4f} |")
    coherence = detail["coherence"]
    lines.extend(["", f"Across {coherence['team_games']} validation team-games, hierarchical player expected TD sums differed from Phase 4 expected team TD by at most {coherence['hierarchical_max_abs_gap']:.6f}. The independent logistic model's Poisson-equivalent player lambda sum minus team expectation averaged {coherence['logistic_mean_gap']:+.3f} TD, with mean absolute gap {coherence['logistic_mean_abs_gap']:.3f}; {coherence['logistic_frac_abs_gap_gt_half']:.1%} of team-games exceeded a 0.5-TD absolute gap. No coherence adjustment was applied to logistic predictions.",
                  "", "## Calibration by position", "",
                  "| Position | Rows | TD rate | Mean P | Log loss | Brier | ECE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for pos, info in detail["position_diagnostics"].items():
        metric = info["metrics"]
        lines.append(f"| {pos} | {info['rows']:,} | {info['td_rate']:.2%} | {info['mean_probability']:.2%} | {metric['log_loss']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |")
    lines.extend(["", "## Selected-logistic calibration buckets", "",
                  "| Predicted P | Player-games | Mean P | Actual TD rate |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for bucket in detail["calibration_table"]:
        lines.append(f"| {bucket['bin']} | {bucket['player_games']:,} | {bucket['mean_probability']:.2%} | {bucket['actual_td_rate']:.2%} |" if bucket["player_games"] else f"| {bucket['bin']} | 0 | — | — |")
    lines.extend(["", "## Logistic coefficients and feature diagnostics", "",
                  "Coefficients are for standardized, imputed inputs. Approximate Fisher intervals ignore model-selection uncertainty and are descriptive. Correlated feature groups can cause unstable individual signs.",
                  "", "| Feature | Standardized coefficient | Approx. 95% interval |",
                  "| --- | ---: | ---: |"])
    for item in detail["coefficient_diagnostics"]["coefficients"]:
        lines.append(f"| {item['feature']} | {item['standardized_coefficient']:+.3f} | [{item['approx_95pct_low']:+.3f}, {item['approx_95pct_high']:+.3f}] |")
    lines.extend(["", "Highest absolute feature correlations: " + ", ".join(
        f"{p['a']} / {p['b']} {p['abs_correlation']:.2f}"
        for p in detail["coefficient_diagnostics"]["high_correlation_pairs"][:8]
    ) + ".", "The standalone expected-team-TD coefficient is negative while implied points is positive. These two environment signals are correlated, and the expected-team-TD interval crosses zero; the combined effect should not be read as a causal decrease. Goal-line and red-zone target signs are small and uncertain. All xTD missing indicators are identical because the source is joined as one family; their individual coefficients are not separately interpretable.",
                  "", "## Missingness and limitations", "",
                  "Route participation is unavailable and excluded, never imputed as zero. All selected numeric families retain nulls in the feature table; the logistic pipeline adds missing indicators and imputes from the training distribution. FB, rookie debuts, players changing teams midseason before an appearance, and long-inactive players are outside this prior-known universe. Historical pregame active rosters are unavailable, so the prior-appearance rule is a proxy, not proof of actual active status.",
                  "The market comparison covers only ten games and may be too small to resolve a close model difference. 2024 is validation/model selection, not an untouched final test. Some xTD feature missingness reflects no earlier scored opportunity rather than zero scoring quality. 2025 remains the untouched historical holdout.",
                  "", "| Feature | 2022 null | 2023 null | 2024 null |",
                  "| --- | ---: | ---: | ---: |"])
    for field, by_year in detail["feature_missing_pct_by_season"].items():
        lines.append(f"| {field} | {by_year['2022']:.2f}% | {by_year['2023']:.2f}% | {by_year['2024']:.2f}% |")
    lines.extend(["", "## Representative 2024 observations", "",
                  "These are diagnostic examples, including missed outcomes; a single binary result does not validate an individual probability. Early-week historical TD rates can rest on only one prior game.",
                  "", "| Pattern | Game | Player | Prior last-8 TD rate | Prior last-5 xTD | History P | Hierarchy P | Logistic P | Actual TD |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    example_labels = {
        "low_actual_high_xtd_scored": "Low TD / high xTD, scored",
        "low_actual_high_xtd_missed": "Low TD / high xTD, missed",
        "high_actual_low_xtd_missed": "High TD / low xTD, missed",
        "high_actual_low_xtd_scored": "High TD / low xTD, scored",
        "logistic_hierarchy_disagreements": "Logistic / hierarchy disagree",
    }
    for key, label in example_labels.items():
        for row in detail["examples"][key][:2]:
            lines.append(f"| {label} | {row['game']} | {row['player']} | {row['last8_td_rate']:.1%} | {row['last5_total_xtd']:.3f} | {row['p_historical_td']:.1%} | {row['p_hierarchical']:.1%} | {row['p_logistic']:.1%} | {row['anytime_td']} |")
    lines.extend(["", "The complete 2024 prediction artifact is `reports/phase5_validation_predictions.csv`; machine-readable metrics and all selected examples are in `reports/phase5_metrics.json`.", ""])
    (REPORTS / "PHASE5.md").write_text("\n".join(lines), encoding="utf-8")


def build_phase5(_settings: Settings | None = None) -> tuple[Path, Path]:
    frame = assemble_frame()
    if frame["season"].max() != 2024 or frame["season"].min() != 2017:
        raise ValueError("Incorrect Phase 5 data window")
    formal = frame.filter(pl.col("season").is_between(2022, 2024))
    train = formal.filter(pl.col("season") == 2022)
    calibration = formal.filter(pl.col("season") == 2023)
    validation = formal.filter(pl.col("season") == 2024)
    if min(train.height, calibration.height, validation.height) == 0:
        raise ValueError("Incomplete chronological split")
    labels = np.asarray(validation["anytime_td"], dtype=int)
    prior_train = frame.filter(pl.col("season") <= 2021)
    priors = {pos: float(cast(float, prior_train.filter(pl.col("position") == pos)["anytime_td"].mean()))
              for pos in POSITIONS}
    history_versions = {
        window: _historical_probabilities(validation, window, priors)
        for window in ("season", "last5", "last8", "ewma")
    }
    history_metrics = {name: _probability_metrics(labels, p) for name, p in history_versions.items()}
    best_history = min(history_metrics, key=lambda name: (history_metrics[name]["log_loss"],
                                                            history_metrics[name]["brier"]))
    p_history = history_versions[best_history]
    hierarchy_weights = fit_hierarchy(train)
    p_hier, lambda_hier = hierarchy_predict(validation, hierarchy_weights)
    family_models: dict[str, Pipeline] = {}
    family_probs: dict[str, np.ndarray] = {}
    family_metrics: dict[str, dict[str, float]] = {}
    for name, fields in MODEL_FAMILIES.items():
        fitted = fit_player_logistic(train, fields)
        predicted = logistic_predict(fitted, validation, fields)
        family_models[name], family_probs[name] = fitted, predicted
        family_metrics[name] = _probability_metrics(labels, predicted)
    e = family_metrics["E_team_usage_xtd"]
    f = family_metrics["F_team_usage_xtd_actual_td"]
    selected_family = "F_team_usage_xtd_actual_td" if (
        f["log_loss"] <= .99 * e["log_loss"] and f["brier"] <= e["brier"]
    ) else "E_team_usage_xtd"
    fields = MODEL_FAMILIES[selected_family]
    selected_model = family_models[selected_family]
    raw = family_probs[selected_family]
    cal_raw = logistic_predict(selected_model, calibration, fields)
    calibrator = _sigmoid_fit(calibration, cal_raw)
    sigmoid = _sigmoid_predict(calibrator, raw)
    raw_metric = _probability_metrics(labels, raw)
    sigmoid_metric = _probability_metrics(labels, sigmoid)
    use_sigmoid = bool(sigmoid_metric["log_loss"] < raw_metric["log_loss"]
                       and sigmoid_metric["brier"] < raw_metric["brier"]
                       and sigmoid_metric["ece"] <= raw_metric["ece"])
    p_logistic = sigmoid if use_sigmoid else raw
    selected_metric = sigmoid_metric if use_sigmoid else raw_metric
    match_mask = np.asarray(validation["p_market"].is_not_null(), dtype=bool)
    market_labels = labels[match_mask]
    if market_labels.size < 50:
        raise ValueError("Insufficient frozen Phase 1 player market benchmark")
    market_probs = np.asarray(validation["p_market"].fill_null(0), dtype=float)[match_mask]
    market_metrics = {
        "historical_td": _probability_metrics(market_labels, p_history[match_mask]),
        "hierarchical": _probability_metrics(market_labels, p_hier[match_mask]),
        "logistic": _probability_metrics(market_labels, p_logistic[match_mask]),
        "market_raw": _probability_metrics(market_labels, market_probs),
    }
    validation = validation.with_columns(
        pl.Series("p_historical_td", p_history), pl.Series("p_hierarchical", p_hier),
        pl.Series("p_logistic_raw", raw), pl.Series("p_logistic", p_logistic),
        pl.Series("expected_player_td", lambda_hier),
    )
    # Lambda is a diagnostic transform; no post-hoc coherence correction is made.
    sums = validation.with_columns(
        pl.Series("logistic_implied_lambda", -np.log1p(-np.clip(p_logistic, 1e-9, 1 - 1e-9)))
    ).group_by("game", "team").agg(
        pl.col("expected_player_td").sum().alias("sum_player_td"),
        pl.col("logistic_implied_lambda").sum().alias("sum_logistic_lambda"),
        pl.col("expected_team_td").first(),
    )
    gap_hier = np.asarray(sums["sum_player_td"] - sums["expected_team_td"], dtype=float)
    gap_log = np.asarray(sums["sum_logistic_lambda"] - sums["expected_team_td"], dtype=float)
    position_metrics = {}
    for pos in POSITIONS:
        mask = np.asarray(validation["position"] == pos, dtype=bool)
        position_metrics[pos] = {"rows": int(mask.sum()), "td_rate": float(labels[mask].mean()),
                                 "mean_probability": float(p_logistic[mask].mean()),
                                 "metrics": _probability_metrics(labels[mask], p_logistic[mask])}
    selected_cols = sorted(set(USAGE + XTD + HIERARCHY_FIELDS
                               + ("last8_td_rate", "expected_team_td", "implied_team_points")))
    missingness = {field: {str(year): round(100 * subset[field].null_count() / subset.height, 2)
                           for year, subset in ((year, formal.filter(pl.col("season") == year))
                                                for year in range(2022, 2025))}
                   for field in selected_cols}
    comparison = {
        "xtd_only_vs_actual_td_only_logloss_delta": family_metrics["B_xtd_only"]["log_loss"] - family_metrics["A_actual_td_only"]["log_loss"],
        "usage_xtd_vs_usage_actual_logloss_delta": family_metrics["D_usage_xtd"]["log_loss"] - family_metrics["C_usage_actual_td"]["log_loss"],
        "actual_incremental_logloss_delta": f["log_loss"] - e["log_loss"],
    }
    metrics = {
        "historical_td": {"rows": len(labels), **history_metrics[best_history]},
        "hierarchical": {"rows": len(labels), **_probability_metrics(labels, p_hier)},
        "logistic_raw": {"rows": len(labels), **raw_metric},
        "logistic_selected": {"rows": len(labels), **selected_metric},
    }
    market_better = market_metrics["logistic"]["log_loss"] < market_metrics["market_raw"]["log_loss"]
    answers = {
        "Does xTD beat recent actual TD?": ("Yes" if comparison["usage_xtd_vs_usage_actual_logloss_delta"] < 0 else "No") + f" with usage controlled: D minus C log loss {comparison['usage_xtd_vs_usage_actual_logloss_delta']:+.5f}. xTD alone minus actual TD alone is {comparison['xtd_only_vs_actual_td_only_logloss_delta']:+.5f}; conclusions differ by comparison.",
        "Does hierarchy beat historical TD?": "Yes on log loss." if metrics["hierarchical"]["log_loss"] < metrics["historical_td"]["log_loss"] else "No on log loss.",
        "Does logistic beat hierarchy?": "Yes on log loss." if metrics["logistic_selected"]["log_loss"] < metrics["hierarchical"]["log_loss"] else "No on log loss.",
        "Overall calibration": f"Selected logistic ECE {selected_metric['ece']:.4f}, intercept {selected_metric['calibration_intercept']:+.3f}, slope {selected_metric['calibration_slope']:.3f}.",
        "Position calibration": "; ".join(f"{pos} mean P {position_metrics[pos]['mean_probability']:.1%} vs actual {position_metrics[pos]['td_rate']:.1%}" for pos in POSITIONS) + ".",
        "Does football beat market?": ("Yes" if market_better else "No") + f" on the {market_labels.size}-row matched 2024 subset by log loss; this limited sample is not conclusive.",
        "Does independent logistic violate team coherence?": f"No large systematic gap: mean lambda-minus-team {float(gap_log.mean()):+.3f} TD. Mean absolute team-game gap {float(np.abs(gap_log).mean()):.3f} TD; {float(np.mean(np.abs(gap_log) > .5)):.1%} of team-games differ by over 0.5 TD.",
        "Does actual TD add after xTD and role?": f"F minus E log loss {comparison['actual_incremental_logloss_delta']:+.5f}; this is too small to justify adding the feature to the selected model.",
    }
    examples = {}
    rows = validation.select("game", "player", "position", "team", "anytime_td",
                             "last8_td_rate", "last5_total_xtd", "p_historical_td",
                             "p_hierarchical", "p_logistic").to_dicts()
    low_td_high_xtd = [r for r in rows if (r["last8_td_rate"] or 0) <= .1
                       and (r["last5_total_xtd"] or 0) >= .2]
    high_td_low_xtd = [r for r in rows if (r["last8_td_rate"] or 0) >= .3
                       and (r["last5_total_xtd"] or 0) <= .1]
    examples["low_actual_high_xtd_scored"] = sorted((r for r in low_td_high_xtd if r["anytime_td"]), key=lambda r: r["last5_total_xtd"], reverse=True)[:4]
    examples["low_actual_high_xtd_missed"] = sorted((r for r in low_td_high_xtd if not r["anytime_td"]), key=lambda r: r["last5_total_xtd"], reverse=True)[:4]
    examples["high_actual_low_xtd_missed"] = sorted((r for r in high_td_low_xtd if not r["anytime_td"]), key=lambda r: r["last8_td_rate"], reverse=True)[:4]
    examples["high_actual_low_xtd_scored"] = sorted((r for r in high_td_low_xtd if r["anytime_td"]), key=lambda r: r["last8_td_rate"], reverse=True)[:4]
    examples["logistic_hierarchy_disagreements"] = sorted(rows, key=lambda r: abs(r["p_logistic"] - r["p_hierarchical"]), reverse=True)[:10]
    detail: dict[str, Any] = {
        "phase": 5, "holdout_2025_untouched": True, "diagnostic_exhibition_slate_excluded": True,
        "population": {"all_2017_2024_rows": frame.height, "fit_rows": train.height,
                       "calibration_rows": calibration.height, "validation_rows": validation.height,
                       "validation_positive_rate": float(labels.mean()),
                       "by_season": {str(y): frame.filter(pl.col("season") == y).height for y in range(2017, 2025)}},
        "team_model_fit_seasons": {"2022": [2021], "2023": [2021, 2022],
                                   "2024": [2021, 2022, 2023]},
        "historical_td_windows": history_metrics, "selected_historical_window": best_history,
        "hierarchy_weights": dict(zip(HIERARCHY_FIELDS, hierarchy_weights.tolist(), strict=True)),
        "controlled_comparisons": family_metrics, "selected_logistic_family": selected_family,
        "calibration": {"raw": raw_metric, "sigmoid": sigmoid_metric, "selected": "sigmoid" if use_sigmoid else "raw",
                        "calibration_fit_season": 2023, "base_fit_seasons": [2022]},
        "metrics": metrics, "comparison_deltas": comparison,
        "market_benchmark": {"matched_players": int(market_labels.size),
                             "games": int(validation.filter(pl.col("p_market").is_not_null())["game"].n_unique()),
                             "metrics": market_metrics},
        "coherence": {"team_games": sums.height, "hierarchical_max_abs_gap": float(np.abs(gap_hier).max()),
                      "hierarchical_mean_gap": float(gap_hier.mean()),
                      "logistic_mean_gap": float(gap_log.mean()),
                      "logistic_mean_abs_gap": float(np.abs(gap_log).mean()),
                      "logistic_frac_abs_gap_gt_half": float(np.mean(np.abs(gap_log) > .5))},
        "position_diagnostics": position_metrics,
        "calibration_table": calibration_table(labels, p_logistic),
        "coefficient_diagnostics": _coefficient_diagnostics(selected_model, train, fields),
        "feature_missing_pct_by_season": missingness, "examples": examples, "answers": answers,
        "stability": {label: _probability_metrics(labels[mask], p_logistic[mask])
                      for label, mask in (("weeks_2_to_9", np.asarray(validation["week"] <= 9)),
                                          ("weeks_10_plus", np.asarray(validation["week"] >= 10)))},
    }
    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    feature_file = DERIVED / "phase5_2017_2024_player_features.parquet"
    prediction_file = REPORTS / "phase5_validation_predictions.csv"
    frame.write_parquet(feature_file)
    output_cols = ["game", "player", "player_id", "position", "team", "opponent", "season",
                   "week", "prediction_time", "anytime_td", "actual_td_count", "p_historical_td",
                   "p_hierarchical", "p_logistic_raw", "p_logistic", "p_market",
                   "expected_player_td", "expected_team_td", "last8_td_rate", "last5_total_xtd",
                   "last5_total_xtd_share", "last5_goal_line_opportunity_share",
                   "last5_carry_share", "last5_target_share", "last5_snap_share"]
    validation.select(output_cols).write_csv(prediction_file)
    detail["artifact_hashes"] = {
        "feature_table": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        "validation_predictions": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
    }
    (REPORTS / "phase5_metrics.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    _write_report(detail)
    return prediction_file, REPORTS / "PHASE5.md"
