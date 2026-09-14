"""2023-fitted market and football-disagreement diagnostics on 2024."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from nfl_td_model.phase7_analysis import _finite_mean, _graded
from nfl_td_model.phase7_stage_b import verify_frozen_rule


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_market = np.clip(np.array([r["p_market_raw_best"] for r in rows], dtype=float), 1e-5, 1 - 1e-5)
    p_football = np.array([r["p_football"] for r in rows], dtype=float)
    y = np.array([r["anytime_td"] for r in rows], dtype=int)
    x = np.column_stack((np.log(p_market / (1 - p_market)), 10 * (p_football - p_market)))
    return x, y, p_market


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "mean_prediction": float(p.mean()), "actual_td_rate": float(y.mean())}


def evaluate() -> Path:
    """Run only after the frozen 2023 rule and both season settlements exist."""
    verify_frozen_rule()
    train = _graded(2023)
    validation = _graded(2024)
    x_train, y_train, _ = _arrays(train)
    x_val, y_val, p_val = _arrays(validation)
    market = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    augmented = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    market.fit(x_train[:, :1], y_train)
    augmented.fit(x_train, y_train)
    train_fit = augmented.predict_proba(x_train)[:, 1]
    design = np.column_stack((np.ones(len(x_train)), x_train))
    information = design.T @ ((train_fit * (1 - train_fit))[:, None] * design)
    stderr = np.sqrt(np.diag(np.linalg.pinv(information)))
    edge_coef = float(augmented.coef_[0, 1])
    edge_se = float(stderr[2])
    z = edge_coef / edge_se if edge_se else float("nan")
    p_value = float(2 * norm.sf(abs(z)))
    calibrated_market = market.predict_proba(x_val[:, :1])[:, 1]
    augmented_val = augmented.predict_proba(x_val)[:, 1]
    comparison = {"raw_market": _metrics(y_val, p_val),
                  "2023_calibrated_market": _metrics(y_val, calibrated_market),
                  "2023_market_plus_edge": _metrics(y_val, augmented_val),
                  "football": _metrics(y_val, np.array([r["p_football"] for r in validation]))}
    by_position = []
    for pos in ("RB", "WR", "TE", "QB", "FB"):
        indices = np.array([i for i, r in enumerate(validation) if r["position"] == pos])
        if len(indices) == 0:
            continue
        by_position.append({"position": pos, "observations": len(indices),
                            "football": _metrics(y_val[indices], np.array([
                                validation[i]["p_football"] for i in indices])),
                            "raw_market": _metrics(y_val[indices], p_val[indices]),
                            "average_football_minus_market": _finite_mean([
                                validation[i]["edge_best"] for i in indices])})
    report = {"training_season": 2023, "validation_season": 2024,
              "training_rows": len(train), "validation_rows": len(validation),
              "market_calibration": {"intercept": float(market.intercept_[0]),
                                     "market_logit_coefficient": float(market.coef_[0, 0])},
              "residual_regression": {"intercept": float(augmented.intercept_[0]),
                                      "market_logit_coefficient": float(augmented.coef_[0, 0]),
                                      "edge_coefficient_per_10pp": edge_coef,
                                      "edge_standard_error": edge_se,
                                      "edge_z_score": z, "edge_p_value": p_value,
                                      "training_market_only_log_loss": _metrics(
                                          y_train, market.predict_proba(x_train[:, :1])[:, 1])["log_loss"],
                                      "training_market_plus_edge_log_loss": _metrics(y_train, train_fit)["log_loss"]},
              "validation": comparison, "by_position": by_position,
              "market_caveat": "one-sided Yes implied prices; no two-sided de-vigging"}
    output = Path("reports/phase7/market_residual_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(evaluate())
