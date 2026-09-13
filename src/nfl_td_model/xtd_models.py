"""Chronological play-level probability models and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from nfl_td_model.xtd_data import (
    RECEIVING_CATEGORICAL,
    RECEIVING_NUMERIC,
    RUSH_CATEGORICAL,
    RUSH_NUMERIC,
)

SEED = 1729
BINS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)
BIN_LABELS = ("0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+")


def feature_dicts(frame: pl.DataFrame, kind: str) -> list[dict[str, float | str]]:
    """Encode missingness explicitly and exclude player/game identity and outcomes."""
    numeric = RUSH_NUMERIC if kind == "rushing" else RECEIVING_NUMERIC
    categorical = RUSH_CATEGORICAL if kind == "rushing" else RECEIVING_CATEGORICAL
    result: list[dict[str, float | str]] = []
    for row in frame.select(*numeric, *categorical).iter_rows(named=True):
        features: dict[str, float | str] = {}
        for key in numeric:
            value = row[key]
            features[key] = float(value) if value is not None else 0.0
            features[f"{key}_missing"] = float(value is None)
        for key in categorical:
            features[key] = str(row[key] or "Unknown")
        result.append(features)
    return result


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Proper scoring rules plus a diagnostic weighted calibration gap."""
    return {
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
        "calibration_gap": float(
            sum(
                len(mask.nonzero()[0]) / len(y) * abs(float(probs[mask].mean() - y[mask].mean()))
                for low, high in pairwise(BINS)
                if (mask := (probs >= low) & (probs < high if high < 1 else probs <= high)).any()
            )
        ),
    }


def calibration_table(y: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    """Reliability bins, including empty bins so reports have stable shape."""
    rows = []
    for label, low, high in zip(BIN_LABELS, BINS[:-1], BINS[1:], strict=True):
        mask = (probs >= low) & (probs < high if high < 1 else probs <= high)
        rows.append(
            {
                "bin": label,
                "observations": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()) if mask.any() else None,
                "actual_td_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return rows


@dataclass
class FittedXTD:
    kind: str
    architecture: str
    pipeline: Pipeline
    calibrator: LogisticRegression | None
    calibration_selected: bool
    fit_max_season: int
    calibration_season: int | None

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        raw = np.asarray(
            self.pipeline.predict_proba(feature_dicts(frame, self.kind))[:, 1], dtype=float
        )
        if self.calibrator is not None:
            raw = np.asarray(self.calibrator.predict_proba(_logit(raw))[:, 1], dtype=float)
        return np.clip(raw, 1e-6, 1 - 1e-6)

    def top_features(self, limit: int = 12) -> list[dict[str, float | str]]:
        vectorizer: DictVectorizer = self.pipeline.named_steps["vector"]
        names = vectorizer.get_feature_names_out()
        model: Any = self.pipeline.named_steps["model"]
        if self.architecture == "logistic":
            values = np.abs(model.coef_[0])
        else:
            values = model.feature_importances_
        order = np.argsort(values)[::-1][:limit]
        return [{"feature": str(names[i]), "importance": float(values[i])} for i in order]


def fit_chronological(
    opportunities: pl.DataFrame, kind: str, architecture: str, target_season: int
) -> tuple[FittedXTD, dict[str, Any]]:
    """Fit before target season and test optional sigmoid calibration on prior year."""
    if target_season < 2018 or target_season > 2024:
        raise ValueError("Phase 3 fitting is limited to historical targets 2018–2024")
    pool = opportunities.filter(pl.col("opportunity_type") == kind)
    if target_season == 2018:
        train = pool.filter(pl.col("season") == 2017)
        calibration = None
    else:
        train = pool.filter(pl.col("season") <= target_season - 2)
        calibration = pool.filter(pl.col("season") == target_season - 1)
    fit_max = int(cast(int, train["season"].max())) if not train.is_empty() else target_season
    if train.is_empty() or fit_max >= target_season:
        raise ValueError("Model fitting includes target or future season")
    if architecture == "logistic":
        pipeline = Pipeline(
            [
                ("vector", DictVectorizer(sparse=True)),
                ("scale", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        )
    elif architecture == "lightgbm":
        pipeline = Pipeline(
            [
                ("vector", DictVectorizer(sparse=True)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=240,
                        learning_rate=0.035,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=120,
                        reg_lambda=2.0,
                        random_state=SEED,
                        n_jobs=1,
                        verbosity=-1,
                        deterministic=True,
                        force_col_wise=True,
                        importance_type="gain",
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown xTD architecture: {architecture}")
    pipeline.fit(feature_dicts(train, kind), np.asarray(train["actual_td"], dtype=int))
    calibrator = None
    diagnostic: dict[str, Any] = {
        "train_rows": train.height,
        "fit_max_season": fit_max,
    }
    if calibration is not None:
        early = calibration.filter(pl.col("week") <= 9)
        late = calibration.filter(pl.col("week") > 9)
        y_early = np.asarray(early["actual_td"], dtype=int)
        y_late = np.asarray(late["actual_td"], dtype=int)
        raw_early = np.asarray(pipeline.predict_proba(feature_dicts(early, kind))[:, 1])
        raw_late = np.asarray(pipeline.predict_proba(feature_dicts(late, kind))[:, 1])
        candidate = LogisticRegression(C=1e6, max_iter=200, random_state=SEED)
        candidate.fit(_logit(raw_early), y_early)
        calibrated_late = candidate.predict_proba(_logit(raw_late))[:, 1]
        raw_metrics = metrics(y_late, raw_late)
        calibrated_metrics = metrics(y_late, calibrated_late)
        selected = (
            calibrated_metrics["log_loss"] < raw_metrics["log_loss"]
            and calibrated_metrics["brier"] < raw_metrics["brier"]
        )
        if selected:
            calibrator = candidate
        diagnostic["calibration"] = {
            "season": target_season - 1,
            "fit_weeks": "1-9",
            "check_weeks": "10+",
            "raw": raw_metrics,
            "sigmoid": calibrated_metrics,
            "selected": selected,
        }
    fitted = FittedXTD(
        kind,
        architecture,
        pipeline,
        calibrator,
        calibrator is not None,
        fit_max,
        target_season - 1 if calibration is not None else None,
    )
    return fitted, diagnostic


def choose_model(logistic: dict[str, float], challenger: dict[str, float]) -> tuple[str, str]:
    """Require at least 1% relative gain on both proper scores and similar calibration."""
    better = (
        challenger["log_loss"] <= logistic["log_loss"] * 0.99
        and challenger["brier"] <= logistic["brier"] * 0.99
        and challenger["calibration_gap"] <= logistic["calibration_gap"] * 1.05
    )
    return (
        ("lightgbm", "at least 1% lower log loss and Brier with no material calibration loss")
        if better
        else ("logistic", "LightGBM did not meet the predeclared material-improvement rule")
    )
