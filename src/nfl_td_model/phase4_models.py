"""Chronological team offensive touchdown count models and diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import PoissonRegressor  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

MARKET = ("implied_team_points", "team_spread", "game_total", "home")
EFFICIENCY = (
    "last5_off_epa_per_play",
    "last5_off_pass_epa",
    "last5_off_rush_epa",
    "last5_off_success_rate",
    "last5_def_epa_allowed",
    "last5_def_pass_epa_allowed",
    "last5_def_rush_epa_allowed",
    "last5_def_success_allowed",
)
PACE_REDZONE = (
    "last5_off_plays",
    "last5_off_pass_rate",
    "last5_off_red_zone_trips",
    "last5_off_red_zone_td_conversion",
    "last5_def_red_zone_td_allowed",
)
ENVIRONMENT = ("rest_days",)
XTD = ("last5_off_xtd", "last5_def_xtd_allowed", "last5_off_red_zone_xtd")
ABLATIONS = {
    "A_market_only": MARKET,
    "B_market_efficiency": MARKET + EFFICIENCY,
    "C_market_efficiency_pace_redzone": MARKET + EFFICIENCY + PACE_REDZONE,
    "D_plus_rest": MARKET + EFFICIENCY + PACE_REDZONE + ENVIRONMENT,
    "C_plus_xtd": MARKET + EFFICIENCY + PACE_REDZONE + XTD,
}
CALIBRATION_BINS = (-math.inf, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, math.inf)
CALIBRATION_NAMES = ("<1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0", "4.0+")


def _matrix(frame: pl.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    return np.asarray(frame.select(fields).to_numpy(), dtype=float)


def count_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.clip(predicted, 1e-6, None)
    table = calibration_table(actual, predicted)
    calibration_gap = sum(
        row["team_games"] / len(actual) * abs(row["mean_predicted"] - row["mean_actual"])
        for row in table
        if row["team_games"]
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "poisson_deviance": float(mean_poisson_deviance(actual, predicted)),
        "mean_prediction_bias": float(np.mean(predicted - actual)),
        "calibration_gap": float(calibration_gap),
    }


def calibration_table(actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for label, low, high in zip(
        CALIBRATION_NAMES, CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:], strict=True
    ):
        mask = (predicted >= low) & (predicted < high)
        rows.append(
            {
                "range": label,
                "team_games": int(mask.sum()),
                "mean_predicted": float(predicted[mask].mean()) if mask.any() else None,
                "mean_actual": float(actual[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def count_distribution(expected_tds: float) -> tuple[float, float, float, float, float]:
    """Poisson reference count probabilities at a nonnegative expected TD mean."""
    if expected_tds < 0:
        raise ValueError("Expected touchdowns cannot be negative")
    p0 = math.exp(-expected_tds)
    p1 = p0 * expected_tds
    p2 = p1 * expected_tds / 2
    p3 = p2 * expected_tds / 3
    return p0, p1, p2, p3, max(0.0, 1 - p0 - p1 - p2 - p3)


@dataclass
class FittedCountModel:
    architecture: str
    fields: tuple[str, ...]
    pipeline: Pipeline | LGBMRegressor
    fit_max_season: int

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.clip(
            np.asarray(self.pipeline.predict(_matrix(frame, self.fields)), dtype=float), 1e-6, None
        )

    def important_features(self, limit: int = 10) -> list[dict[str, Any]]:
        if self.architecture == "lightgbm":
            importance = np.asarray(self.pipeline.feature_importances_, dtype=float)
            names = self.fields
        else:
            pipeline = self.pipeline
            coefficients = np.asarray(pipeline.named_steps["model"].coef_, dtype=float)
            names = tuple(pipeline.named_steps["imputer"].get_feature_names_out(self.fields))
            importance = np.abs(coefficients)
        order = np.argsort(importance)[::-1][:limit]
        return [
            {"feature": names[index], "importance": float(importance[index])} for index in order
        ]


def fit_count_model(
    train: pl.DataFrame, fields: tuple[str, ...], architecture: str
) -> FittedCountModel:
    """Fit only on 2017–2023 outcomes; 2025 is forbidden at the API boundary."""
    if train.is_empty() or cast(int, train["season"].max()) > 2023:
        raise ValueError("Phase 4 fit includes 2024 validation or 2025 holdout")
    x = _matrix(train, fields)
    y = np.asarray(train["actual_offensive_td"], dtype=float)
    if architecture == "poisson":
        model: Pipeline | LGBMRegressor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=0.15, max_iter=500)),
            ]
        )
    elif architecture == "lightgbm":
        model = LGBMRegressor(
            objective="poisson",
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=3,
            min_child_samples=60,
            reg_lambda=4.0,
            random_state=1729,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            importance_type="gain",
        )
    else:
        raise ValueError("Unknown count architecture")
    model.fit(x, y)
    return FittedCountModel(architecture, fields, model, cast(int, train["season"].max()))


def validate_model(
    train: pl.DataFrame, validation: pl.DataFrame, fields: tuple[str, ...], architecture: str
) -> tuple[FittedCountModel, np.ndarray, dict[str, Any]]:
    if (
        validation.is_empty()
        or validation["season"].min() != 2024
        or validation["season"].max() != 2024
    ):
        raise ValueError("Phase 4 validation must be 2024 only")
    model = fit_count_model(train, fields, architecture)
    predicted = model.predict(validation)
    actual = np.asarray(validation["actual_offensive_td"], dtype=float)
    return (
        model,
        predicted,
        {
            "train_rows": train.height,
            "train_seasons": sorted(train["season"].unique().to_list()),
            "validation_rows": validation.height,
            "fit_max_season": model.fit_max_season,
            "metrics": count_metrics(actual, predicted),
            "calibration": calibration_table(actual, predicted),
            "important_features": model.important_features(),
        },
    )


def materially_better(baseline: dict[str, float], challenger: dict[str, float]) -> bool:
    """Require 1% improvement in RMSE and deviance without worse MAE."""
    return bool(
        challenger["rmse"] <= baseline["rmse"] * 0.99
        and challenger["poisson_deviance"] <= baseline["poisson_deviance"] * 0.99
        and challenger["mae"] <= baseline["mae"]
    )
