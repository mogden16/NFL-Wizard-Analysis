"""Predeclared Phase 7 betting diagnostics, with a hard 2023/2024 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl_td_model.phase7_stage_b import verify_frozen_rule, verify_stage_a

ROOT = Path("reports/phase7")
EDGES = (0.0, 0.025, 0.05, 0.075, 0.10)
EVS = (0.0, 0.05, 0.10, 0.15)
BOOKS = (1, 2, 3)
EDGE_BINS = ((-math.inf, -0.10), (-0.10, -0.05), (-0.05, -0.025),
             (-0.025, 0.0), (0.0, 0.025), (0.025, 0.05),
             (0.05, 0.10), (0.10, math.inf))
EDGE_LABELS = ("<=-10pp", "-10:-5pp", "-5:-2.5pp", "-2.5:0pp",
               "0:+2.5pp", "+2.5:+5pp", "+5:+10pp", ">+10pp")


def _graded(season: int) -> list[dict[str, Any]]:
    """Load one season only; development has no 2024 outcome dependency."""
    verify_stage_a()
    if season == 2024:
        verify_frozen_rule()
    elif season != 2023:
        raise ValueError("The 2025 holdout and 2026 exhibition are sealed")
    path = ROOT / f"settlement_{season}.csv"
    manifest = json.loads((ROOT / f"settlement_{season}_manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["settlement_sha256"]:
        raise ValueError("Frozen settlement was modified")
    return pl.read_csv(path).filter(pl.col("settlement_status") == "graded_played").to_dicts()


def _finite_mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def _bootstrap_roi(rows: list[dict[str, Any]], price: str) -> list[float] | None:
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_game[row["game"]].append(float(row[f"profit_{price}_units"]))
    if len(by_game) < 2:
        return None
    game_values = np.array([(sum(v), len(v)) for v in by_game.values()], dtype=float)
    rng = np.random.default_rng(1729)
    indices = rng.integers(0, len(game_values), size=(2000, len(game_values)))
    samples = game_values[indices].sum(axis=1)
    valid = samples[:, 1] > 0
    if not valid.any():
        return None
    return [float(x) for x in np.quantile(samples[valid, 0] / samples[valid, 1], [0.025, 0.975])]


def portfolio(rows: list[dict[str, Any]], bootstrap: bool = False) -> dict[str, Any]:
    """Use identical selections for best/median, one unit each."""
    ordered = sorted(rows, key=lambda r: (r["kickoff"], r["game"], r["player"]))
    wins = sum(int(row["anytime_td"]) for row in ordered)
    result: dict[str, Any] = {"bets": len(ordered), "wins": wins,
                              "losses": len(ordered) - wins,
                              "average_odds": _finite_mean([r["best_american"] for r in ordered]),
                              "average_clv_best": _finite_mean([r["clv_best_probability"] for r in ordered]),
                              "average_clv_median": _finite_mean([r["clv_median_probability"] for r in ordered]),
                              "clv_coverage": (sum(r["clv_best_probability"] is not None for r in ordered)
                                               / len(ordered) if ordered else None)}
    for price in ("best", "median"):
        profits = [float(r[f"profit_{price}_units"]) for r in ordered]
        units = sum(profits)
        peak = equity = max_drawdown = 0.0
        losing_streak = longest_streak = 0
        for profit in profits:
            equity += profit
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            losing_streak = losing_streak + 1 if profit < 0 else 0
            longest_streak = max(longest_streak, losing_streak)
        gross = sum(max(0.0, p) for p in profits)
        winning = sorted((p for p in profits if p > 0), reverse=True)
        result[price] = {"units": units, "roi": units / len(profits) if profits else None,
                         "max_drawdown": max_drawdown, "longest_losing_streak": longest_streak,
                         "largest_winner_gross_share": winning[0] / gross if gross else None,
                         "top_two_winner_gross_share": sum(winning[:2]) / gross if gross else None}
        if bootstrap:
            result[price]["roi_95_game_bootstrap"] = _bootstrap_roi(ordered, price)
    return result


def _rule_rows(rows: list[dict[str, Any]], min_edge: float, min_ev: float,
               min_books: int) -> list[dict[str, Any]]:
    return [r for r in rows if r["edge_best"] is not None and
            r["edge_best"] >= min_edge and r["ev_best"] >= min_ev and r["books"] >= min_books]


def _bucket(rows: list[dict[str, Any]], field: str, lo: float, hi: float) -> list[dict[str, Any]]:
    return [r for r in rows if r[field] is not None and lo < float(r[field]) <= hi]


def diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edge_rows = []
    for label, (lo, hi) in zip(EDGE_LABELS, EDGE_BINS, strict=True):
        subset = _bucket(rows, "edge_best", lo, hi)
        edge_rows.append({"bucket": label, "observations": len(subset),
                          "football_probability": _finite_mean([r["p_football"] for r in subset]),
                          "market_probability": _finite_mean([r["p_market_raw_best"] for r in subset]),
                          "actual_td_rate": _finite_mean([r["anytime_td"] for r in subset]),
                          "calibration_residual": _finite_mean([
                              r["anytime_td"] - r["p_market_raw_best"] for r in subset]),
                          "average_odds": _finite_mean([r["best_american"] for r in subset]),
                          "roi": portfolio(subset)["best"]["roi"],
                          "average_clv": _finite_mean([r["clv_best_probability"] for r in subset])})
    age_ranges = ((-math.inf, 5, "<=5m"), (5, 15, "5-15m"),
                  (15, 30, "15-30m"), (30, math.inf, ">30m"))
    age = [{"bucket": label, **portfolio(_bucket(rows, "best_quote_age_minutes", lo, hi))}
           for lo, hi, label in age_ranges]
    odds_ranges = ((-math.inf, 0, "negative"), (99, 199, "+100-199"),
                   (199, 299, "+200-299"), (299, 499, "+300-499"),
                   (499, math.inf, "+500+"))
    odds = [{"bucket": label, **portfolio(_bucket(rows, "best_american", lo, hi))}
            for lo, hi, label in odds_ranges]
    positions = [{"position": pos, **portfolio([r for r in rows if r["position"] == pos])}
                 for pos in ("RB", "WR", "TE", "QB", "FB")]
    books = [{"sportsbook": book, **portfolio([r for r in rows if r["best_sportsbook"] == book])}
             for book in sorted({str(r["best_sportsbook"]) for r in rows})]
    dispersion = [{"bucket": label, **portfolio(_bucket(rows, "best_vs_median_decimal_deviation", lo, hi))}
                  for lo, hi, label in ((-math.inf, 0.05, "<=5%"), (0.05, 0.15, "5-15%"),
                                         (0.15, 0.30, "15-30%"), (0.30, math.inf, ">30%"))]
    return {"edge_buckets": edge_rows, "quote_age": age, "odds_range": odds,
            "position": positions, "sportsbook": books, "dispersion": dispersion}


def develop() -> Path:
    """Finalize all 2023 choices and hash them before any 2024 settlement read."""
    rows = _graded(2023)
    path = ROOT / "development/frozen_rule.json"
    if path.exists():
        raise FileExistsError("2023 development rule is already frozen")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = []
    qualified = []
    for edge, ev, books in product(EDGES, EVS, BOOKS):
        selected = _rule_rows(rows, edge, ev, books)
        metrics = portfolio(selected)
        item = {"min_edge": edge, "min_ev": ev, "min_books": books, **metrics}
        grid.append(item)
        if (metrics["bets"] >= 100 and metrics["best"]["roi"] is not None and
            metrics["best"]["roi"] > 0 and metrics["median"]["roi"] is not None and
            metrics["median"]["roi"] >= 0 and metrics["average_clv_best"] is not None and
            metrics["average_clv_best"] >= 0 and metrics["clv_coverage"] is not None and
            metrics["clv_coverage"] >= 0.5 and
            metrics["best"]["largest_winner_gross_share"] is not None and
            metrics["best"]["largest_winner_gross_share"] <= 0.5):
            qualified.append(item)
    qualified.sort(key=lambda r: (r["median"]["roi"], r["average_clv_best"], r["bets"]),
                   reverse=True)
    shortlist = [{"min_edge": r["min_edge"], "min_ev": r["min_ev"],
                  "min_books": r["min_books"]} for r in qualified[:5]]
    report = {"season": 2023, "graded_rows": len(rows), "full_grid": grid,
              "shortlist_metrics": [{**rule, **portfolio(_rule_rows(rows, **rule), bootstrap=True)}
                                    for rule in shortlist], "diagnostics": diagnostics(rows)}
    (path.parent / "development_report.json").write_text(json.dumps(report, indent=2) + "\n")
    frozen = {"schema": "phase7-rule-v1", "development_season": 2023,
              "validation_season": 2024, "model_version": "phase5_hierarchical_frozen",
              "probability_source": "frozen_phase5_hierarchical", "prediction_offset_minutes": -60,
              "candidate_grid": {"min_edge": EDGES, "min_ev": EVS, "min_books": BOOKS},
              "shortlist": shortlist, "status": "DEVELOPMENT_CANDIDATES" if shortlist else "NO_BET",
              "quote_age_rule": None, "dispersion_rule": None,
              "eligible_positions": ["RB", "WR", "TE", "QB", "FB"],
              "odds_ranges": "all", "flat_stake_units": 1.0,
              "settlement_rule": "affirmative_play_only; other status policy_unknown",
              "eligibility_limitation": "historical active status unavailable; all strategies diagnostic proxy"}
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    (path.parent / "frozen_rule.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return path


def validate() -> Path:
    """Read 2024 outcomes only after the 2023 development choice is hashed."""
    rule = verify_frozen_rule()
    rows = _graded(2024)
    shortlist = []
    for candidate in rule["shortlist"]:
        selected = _rule_rows(rows, **candidate)
        summary = portfolio(selected, bootstrap=True)
        segment_units = []
        for lo, hi in ((2, 6), (7, 12), (13, 99)):
            segment = [r for r in selected if lo <= int(r["game"].split("_")[1]) <= hi]
            segment_units.append(portfolio(segment)["best"]["units"])
        positive_net = summary["best"]["units"]
        stale_net = sum(r["profit_best_units"] for r in selected if r["best_quote_age_minutes"] > 30)
        longshot_net = sum(r["profit_best_units"] for r in selected if r["best_american"] >= 500)
        promoted = (summary["bets"] >= 100 and summary["best"]["roi"] is not None and
                    summary["best"]["roi"] > 0 and summary["median"]["roi"] is not None and
                    summary["median"]["roi"] > 0 and summary["average_clv_best"] is not None and
                    summary["average_clv_best"] >= 0 and summary["clv_coverage"] is not None and
                    summary["clv_coverage"] >= 0.5 and sum(x >= 0 for x in segment_units) >= 2 and
                    summary["best"]["top_two_winner_gross_share"] is not None and
                    summary["best"]["top_two_winner_gross_share"] <= 0.5 and
                    stale_net <= positive_net / 2 and longshot_net <= positive_net / 2)
        shortlist.append({**candidate, **summary, "segment_units": segment_units,
                          "stale_quote_net_units": stale_net, "longshot_net_units": longshot_net,
                          "passes_predeclared_promotion": promoted})
    result = "CANDIDATE_ADVANCES" if any(s["passes_predeclared_promotion"] for s in shortlist) else "NO_ROBUST_EDGE"
    report = {"season": 2024, "graded_rows": len(rows), "PHASE7_RESULT": result,
              "shortlist": shortlist, "diagnostics": diagnostics(rows),
              "eligibility_limitation": "authoritative pregame active status absent; no operational rule can advance"}
    output = ROOT / "validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "validate"])
    args = parser.parse_args()
    print(develop() if args.stage == "develop" else validate())
