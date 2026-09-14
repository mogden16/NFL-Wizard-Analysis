"""Distribution calibration diagnostics for the frozen usage-prop EWMA means."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nfl_td_model.usage_props import load_historical_prop_frame, target_column

CALIBRATION_BINS = (("<40%", 0.0, 0.40), ("40-45%", 0.40, 0.45),
                    ("45-50%", 0.45, 0.50), ("50-55%", 0.50, 0.55),
                    ("55-60%", 0.55, 0.60), ("60-65%", 0.60, 0.65),
                    ("65%+", 0.65, 1.01))


def negative_binomial_alpha(values: np.ndarray, means: np.ndarray) -> float:
    """Method-of-moments overdispersion estimated on development data only."""
    mean = float(np.mean(means))
    variance = float(np.var(values, ddof=1))
    if mean <= 0:
        return 0.0
    return max(0.0, (variance - mean) / max(mean * mean, 1e-9))


def count_pmf(method: str, mean: float, count: int, alpha: float = 0.0,
              residuals: np.ndarray | None = None) -> float:
    """Probability mass for Poisson, Negative Binomial, or residual mixture."""
    if mean < 0 or count < 0:
        raise ValueError("Count mean and count must be nonnegative")
    if method == "poisson":
        return math.exp(-mean + count * math.log(mean) - math.lgamma(count + 1)) if mean else float(count == 0)
    if method == "negative_binomial":
        if alpha <= 0:
            return count_pmf("poisson", mean, count)
        size = 1 / alpha
        probability = size / (size + mean)
        if probability >= 1:
            return float(count == 0)
        return math.exp(math.lgamma(count + size) - math.lgamma(size) - math.lgamma(count + 1)
                        + size * math.log(probability)
                        + count * math.log1p(-probability))
    if method == "empirical_residual":
        if residuals is None or residuals.size == 0:
            return count_pmf("poisson", mean, count)
        simulated = np.maximum(0, np.rint(mean + residuals)).astype(int)
        return float(np.mean(simulated == count))
    raise ValueError(f"Unknown distribution method: {method}")


def distribution_probabilities(method: str, mean: float, line: float,
                               alpha: float = 0.0,
                               residuals: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return Over, Under, Push probabilities for an integer or half line."""
    if line < 0:
        raise ValueError("Prop line cannot be negative")
    if method == "empirical_residual" and residuals is not None and residuals.size:
        simulated = np.maximum(0, np.rint(mean + residuals))
        if float(line).is_integer():
            push = float(np.mean(simulated == line))
            over = float(np.mean(simulated > line))
            under = float(np.mean(simulated < line))
        else:
            push = 0.0
            over = float(np.mean(simulated > line))
            under = float(np.mean(simulated <= line))
        total = over + under + push
        return over / total, under / total, push / total
    upper = max(100, math.ceil(mean + 12 * math.sqrt(mean + alpha * mean * mean + 1)))
    masses = np.array([count_pmf(method, mean, k, alpha, residuals) for k in range(upper)])
    masses = masses / masses.sum()
    if float(line).is_integer():
        index = int(line)
        push = float(masses[index]) if index < len(masses) else 0.0
        over = float(masses[index + 1:].sum())
        under = float(masses[:index].sum())
    else:
        push = 0.0
        over = float(masses[math.floor(line) + 1:].sum())
        under = float(masses[:math.floor(line) + 1].sum())
    total = over + under + push
    return over / total, under / total, push / total


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return {"log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "brier": float(np.mean((probability - y) ** 2)),
            "mean_prediction": float(probability.mean()), "actual_rate": float(y.mean())}


def _calibration_table(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    table = []
    for label, low, high in CALIBRATION_BINS:
        mask = (probability >= low) & (probability < high)
        table.append({"bucket": label, "observations": int(mask.sum()),
                      "mean_predicted_probability": float(probability[mask].mean()) if mask.any() else None,
                      "actual_over_rate": float(y[mask].mean()) if mask.any() else None})
    return table


def _residuals(frame: Any, prop: str) -> tuple[np.ndarray, np.ndarray]:
    target = target_column(prop)
    mean_field = "ewma_receptions_per_game" if prop == "receptions" else "ewma_carries_per_game"
    values = np.asarray(frame[target].fill_null(0).to_numpy(), dtype=float)
    means = np.asarray(frame[mean_field].fill_null(0).to_numpy(), dtype=float)
    return values, means


def _method_result(method: str, values: np.ndarray, means: np.ndarray,
                   alpha: float, residuals: np.ndarray) -> dict[str, Any]:
    line = np.floor(means).astype(int)
    probabilities = np.array([
        distribution_probabilities(method, float(mean), float(current_line), alpha, residuals)[0]
        for mean, current_line in zip(means, line, strict=True)
    ])
    outcome = (values > line).astype(int)
    return {"method": method, "probability_metrics": _metric(outcome, probabilities),
            "calibration": _calibration_table(outcome, probabilities)}


def calibration_report() -> Path:
    """Fit dispersion/residual behavior on 2017-2023 and validate on 2024."""
    frame = load_historical_prop_frame()
    report: dict[str, Any] = {"development_seasons": list(range(2017, 2024)),
                              "validation_season": 2024,
                              "market_matched_validation": {},
                              "market_data_note": "No historical player_receptions or player_rush_attempts lines are archived; no lines were fabricated."}
    for prop in ("receptions", "rushing_attempts"):
        train = frame.filter(frame["season"] < 2024)
        validation = frame.filter(frame["season"] == 2024)
        train_values, train_means = _residuals(train, prop)
        values, means = _residuals(validation, prop)
        report["market_matched_validation"][prop] = {
            "total_2024_player_games": validation.height,
            "market_matched_player_games": 0,
            "unique_players": 0,
            "positions": [],
            "positive_over_outcomes": None,
            "negative_over_outcomes": None,
            "note": "No archived timestamped prop line; outcome relative to a line is unavailable.",
        }
        alpha = negative_binomial_alpha(train_values, train_means)
        residuals = (train_values - train_means)
        if residuals.size > 5000:
            residuals = residuals[:: math.ceil(residuals.size / 5000)]
        methods = [_method_result("poisson", values, means, 0.0, residuals),
                   _method_result("negative_binomial", values, means, alpha, residuals),
                   _method_result("empirical_residual", values, means, 0.0, residuals)]
        variance_ratio = float(np.var(train_values, ddof=1) / max(np.mean(train_values), 1e-9))
        report[prop] = {"training_rows": train.height, "validation_rows": validation.height,
                        "training_variance_to_mean": variance_ratio, "negative_binomial_alpha": alpha,
                        "market_matched_rows": 0, "methods": methods,
                        "selected_method": "negative_binomial" if methods[1]["probability_metrics"]["log_loss"] < methods[0]["probability_metrics"]["log_loss"] else "poisson",
                        "status": "RESEARCH_ONLY"}
    output = Path("reports/usage_prop_calibration.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def selected_distribution(prop: str) -> tuple[str, float]:
    """Return the frozen diagnostic choice for live display, never a new mean model."""
    report = json.loads(Path("reports/usage_prop_calibration.json").read_text(encoding="utf-8"))
    entry = report[prop]
    method = str(entry["selected_method"])
    return method, float(entry["negative_binomial_alpha"] if method == "negative_binomial" else 0.0)
