"""Post-freeze descriptive best/median and CLV diagnostics for both seasons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from nfl_td_model.phase7_analysis import EDGE_BINS, EDGE_LABELS, _finite_mean, _graded, portfolio


def _clv(rows: list[dict[str, Any]], price: str) -> dict[str, Any]:
    values = [float(r[f"clv_{price}_probability"]) for r in rows
              if r[f"clv_{price}_probability"] is not None]
    return {"observations": len(values), "mean_probability_clv": _finite_mean(values),
            "median_probability_clv": sorted(values)[len(values) // 2] if values else None,
            "percent_beating_close": sum(value > 0 for value in values) / len(values) if values else None}


def _edge_table(rows: list[dict[str, Any]], price: str) -> list[dict[str, Any]]:
    entries = []
    for label, (lo, hi) in zip(EDGE_LABELS, EDGE_BINS, strict=True):
        subset = [r for r in rows if r[f"edge_{price}"] is not None and
                  lo < float(r[f"edge_{price}"]) <= hi]
        result = portfolio(subset)
        entries.append({"edge_bucket": label, "count": len(subset),
                        "football_probability": _finite_mean([r["p_football"] for r in subset]),
                        "market_probability": _finite_mean([r[f"p_market_raw_{price}"] for r in subset]),
                        "actual_td_rate": _finite_mean([r["anytime_td"] for r in subset]),
                        "mean_odds": _finite_mean([r["best_american"] for r in subset]),
                        "best_roi": result["best"]["roi"], "median_roi": result["median"]["roi"],
                        "clv": _clv(subset, price)})
    return entries


def _groups(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [{field: value, "portfolio": portfolio(subset), "clv": _clv(subset, "best")}
            for value in sorted({str(r[field]) for r in rows})
            if (subset := [r for r in rows if str(r[field]) == value])]


def describe() -> Path:
    out: dict[str, Any] = {}
    for season in (2023, 2024):
        rows = _graded(season)
        positive = [r for r in rows if r["edge_best"] > 0 and r["ev_best"] > 0]
        extreme = [r for r in rows if r["edge_best"] > 0.10]
        age_groups = ((-math.inf, 5, "<=5m"), (5, 15, "5-15m"),
                      (15, 30, "15-30m"), (30, math.inf, ">30m"))
        dispersion_groups = ((-math.inf, 0.05, "<=5%"), (0.05, 0.15, "5-15%"),
                             (0.15, 0.30, "15-30%"), (0.30, math.inf, ">30%"))
        out[str(season)] = {
            "modeled_graded_rows": len(rows),
            "all_quoted_players": len({(r["game"], r["player"]) for r in rows}),
            "best_edge_buckets": _edge_table(rows, "best"),
            "median_edge_buckets": _edge_table(rows, "median"),
            "positive_edge_and_ev": {"portfolio": portfolio(positive, bootstrap=True),
                                      "clv": _clv(positive, "best")},
            "extreme_positive_edge_diagnostic_not_rule": {
                "portfolio": portfolio(extreme, bootstrap=True),
                "clv": _clv(extreme, "best")},
            "positive_edge_quote_age": [
                {"bucket": label, "portfolio": portfolio(subset), "clv": _clv(subset, "best")}
                for lo, hi, label in age_groups
                if (subset := [r for r in positive if lo < r["best_quote_age_minutes"] <= hi])],
            "positive_edge_dispersion": [
                {"bucket": label, "portfolio": portfolio(subset), "clv": _clv(subset, "best")}
                for lo, hi, label in dispersion_groups
                if (subset := [r for r in positive if lo < r["best_vs_median_decimal_deviation"] <= hi])],
            "clv_by_sportsbook": _groups(rows, "best_sportsbook"),
            "clv_by_position": _groups(rows, "position"),
        }
    path = Path("reports/phase7/supplemental_diagnostics.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(describe())
