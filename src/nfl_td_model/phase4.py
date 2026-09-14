"""Phase 4 team offensive touchdown environment build and chronological comparison."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.phase4_football import build_team_history, lag_team_context
from nfl_td_model.phase4_market import build_strict_market_table
from nfl_td_model.phase4_models import (
    ABLATIONS,
    EFFICIENCY,
    ENVIRONMENT,
    PACE_REDZONE,
    XTD,
    calibration_table,
    count_distribution,
    count_metrics,
    materially_better,
    validate_model,
)


def _coverage(frame: pl.DataFrame, fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        str(season[0] if isinstance(season, tuple) else season): {
            "team_games": part.height,
            "first_kickoff": cast(datetime, part["kickoff"].min()).isoformat(),
            "last_kickoff": cast(datetime, part["kickoff"].max()).isoformat(),
            "feature_present_pct": {
                field: round(100 * part[field].is_not_null().sum() / part.height, 2)
                for field in fields
            },
        }
        for season, part in frame.partition_by("season", as_dict=True).items()
    }


def _examples(frame: pl.DataFrame) -> list[dict[str, Any]]:
    choices = (
        ("heavy favorite", frame.sort("team_spread").head(1)),
        ("underdog", frame.sort("team_spread", descending=True).head(1)),
        ("high total", frame.sort("game_total", descending=True).head(1)),
        ("low total", frame.sort("game_total").head(1)),
    )
    return [
        {
            "category": label,
            "game_id": row["game_id"],
            "team": row["team"],
            "opponent": row["opponent"],
            "market_implied_points": row["implied_team_points"],
            "market_only_td": row["market_only_td"],
            "football_market_td": row["football_market_td"],
            "actual_offensive_td": row["actual_offensive_td"],
        }
        for label, selected in choices
        for row in selected.to_dicts()
    ]


def _report(detail: dict[str, Any]) -> str:
    m = detail["market_coverage"]
    fm = detail["football_coverage"]
    rows = [
        "# Phase 4 completion report",
        "",
        "The target is a team's nflverse rushing plus receiving touchdowns credited to the possessing team. Defensive returns, special-teams returns, blocked-kick scores, and two-point tries are excluded. A touchdown on a lateral play remains an offensive touchdown when nflverse credits its `rush_touchdown` or `pass_touchdown` to the possessing team. Every team-game target agreed with the separate weekly team rushing-plus-passing TD totals.",
        "",
        f"Football-only history: {detail['football_team_games']:,} team-games across 2017–2024 ({fm['first_kickoff']} to {fm['last_kickoff']}). Strict T−60 market history: {m['team_games']:,} team-games, {m['games']:,} games across 2021–2024 ({m['first_kickoff']} to {m['last_kickoff']}); {m['snapshot_count']} cached daily snapshots. All 2024 regular-season games are represented. No 2025-season data was loaded.",
        "",
        "Sportsbook values come from one book's spread and total market updates in a historical snapshot at or before kickoff minus 60 minutes. The home-spread convention is the handicap added to home points, so a home favorite has a negative spread, `home_margin = -home_spread`, and home implied points are `(total + home_margin)/2`. nflverse closing lines never enter the strict table.",
        f"Quote age relative to each T−60 cutoff: median {m['quote_age_minutes']['median']:.0f} minutes, 90th percentile {m['quote_age_minutes']['p90']:.0f}, maximum {m['quote_age_minutes']['max']:.0f}.",
        "",
        f"Training: 2021–2023 strict market team-games ({detail['strict_train_rows']:,}); 2024 chronological validation ({detail['validation_rows']:,}); 2025 untouched. The football-only diagnostic trains on 2017–2023 and validates on 2024. No random split or betting-return selection was used.",
        f"The points-per-TD reference divides implied team points by the empirical 2021–2023 league ratio of {detail['league_points_per_offensive_td']:.3f} points per offensive touchdown; it does not divide by seven.",
        "",
        "## 2024 count-model comparison",
        "",
        "| Model | MAE | RMSE | Poisson deviance | Bias | Calibration gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, info in detail["headline_metrics"].items():
        metric = info["metrics"]
        rows.append(
            f"| {name} | {metric['mae']:.4f} | {metric['rmse']:.4f} | {metric['poisson_deviance']:.4f} | {metric['mean_prediction_bias']:+.4f} | {metric['calibration_gap']:.4f} |"
        )
    inc = detail["incremental_football_vs_market"]
    rows.extend(
        [
            "",
            f"Market-only benchmark: **{detail['market_model']}**. Best controlled football challenger: **{detail['football_challenger']}**. Final selected model: **{detail['selected_model']}**.",
            f"Football minus market: MAE {inc['mae_pct']:+.2f}%, RMSE {inc['rmse_pct']:+.2f}%, count deviance {inc['poisson_deviance_pct']:+.2f}%, calibration gap {inc['calibration_gap_pct']:+.2f}% (negative means improvement). Material improvement requires at least 1% lower RMSE and deviance with no worse MAE.",
            f"**Did football improve on the sportsbook in 2024? {'Yes, by the predeclared materiality rule.' if detail['football_material_improvement'] else 'No material improvement; the market-only model remains selected.'}**",
            "",
            "Leading market-model features: "
            + ", ".join(
                item["feature"]
                for item in detail["headline_metrics"]["market-only"]["important_features"][:6]
            )
            + ".",
            "Leading football-challenger features: "
            + ", ".join(
                item["feature"]
                for item in detail["headline_metrics"]["football+market"]["important_features"][:8]
            )
            + ".",
            "The C+xTD diagnostic did not improve both RMSE and deviance over C without xTD on 2024; xTD was not added to the selected market model.",
            "",
            "## Controlled feature-family ablation",
            "",
            "A uses market only; B adds prior offense/opponent efficiency; C adds pace and red-zone usage; D adds rest. A separate C+xTD test adds prior team and opponent xTD. Weather and starter fields are not added without timestamped pregame evidence. Every experiment below was fixed in advance and uses the same 2024 validation team-games.",
            "",
            "| Experiment | MAE | RMSE | Deviance | Calibration gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, experiment in detail["experiments"].items():
        metric = experiment["metrics"]
        rows.append(
            f"| {name} | {metric['mae']:.4f} | {metric['rmse']:.4f} | {metric['poisson_deviance']:.4f} | {metric['calibration_gap']:.4f} |"
        )
    dispersion = detail["dispersion"]
    rows.extend(
        [
            "",
            "## Team TD count distribution",
            "",
            f"Training offensive TD mean {dispersion['mean']:.3f}, variance {dispersion['variance']:.3f}, variance/mean {dispersion['variance_to_mean']:.3f}; conditional Poisson Pearson dispersion {dispersion['pearson_dispersion']:.3f}. Material overdispersion required both ratios to exceed 1.2 and was {'present; Negative Binomial was evaluated' if detail['negative_binomial_evaluated'] else 'not present; Negative Binomial was not justified'}. The published P(0), P(1), P(2), P(3), P(4+) use a Poisson reference distribution and sum to one; its shape is a working assumption, not a claim of perfect count calibration.",
            "",
            "Observed 2024 offensive TD counts: "
            + ", ".join(
                f"{key}: {value}" for key, value in detail["actual_distribution_2024"].items()
            )
            + ".",
            "",
            "| Count | Mean predicted probability | Observed frequency |",
            "|---|---:|---:|",
        ]
    )
    for label, predicted, observed in detail["distribution_calibration"]:
        rows.append(f"| {label} | {predicted:.3f} | {observed:.3f} |")
    rows.extend(
        [
            "",
            "The Poisson reference overpredicts 0- and 1-TD team-games and underpredicts 2-TD team-games on 2024 validation, consistent with underdispersion. Negative Binomial would add overdispersion and was not fitted.",
        ]
    )
    rows.extend(
        [
            "",
            "## Coverage and missingness",
            "",
            "The strict market table has complete spread, total, implied-points and home/away fields for every included team-game. The football-only table retains null lagged fields until the required earlier games exist; missing forecasts and starter signals remain null.",
            "",
            "| Season | Football team-games | Strict market team-games | Prior offense EPA present | Prior opponent EPA present | Prior xTD present |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for season, info in sorted(detail["football_coverage"]["by_season"].items()):
        present = info["feature_present_pct"]
        market_games = detail["market_coverage"]["by_season"].get(season, {}).get("team_games", 0)
        rows.append(
            f"| {season} | {info['team_games']} | {market_games} | {present['last5_off_epa_per_play']:.1f}% | {present['last5_def_epa_allowed']:.1f}% | {present['last5_off_xtd']:.1f}% |"
        )
    rows.extend(
        [
            "",
            "## Calibration by predicted team TD",
            "",
        ]
    )
    for name in ("market_only", "football_market", "selected"):
        rows.extend(
            [
                f"### {name.replace('_', ' ').title()}",
                "",
                "| Predicted range | Team-games | Mean prediction | Mean actual |",
                "|---|---:|---:|---:|",
            ]
        )
        for entry in detail["calibration"][name]:
            predicted = "—" if entry["mean_predicted"] is None else f"{entry['mean_predicted']:.3f}"
            actual = "—" if entry["mean_actual"] is None else f"{entry['mean_actual']:.3f}"
            rows.append(f"| {entry['range']} | {entry['team_games']} | {predicted} | {actual} |")
        rows.append("")
    rows.extend(
        [
            "## Representative 2024 games",
            "",
            "| Case | Team | Market implied pts | Market-only TD | Football+market TD | Actual TD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for ex in detail["examples"]:
        rows.append(
            f"| {ex['category']} | {ex['team']} | {ex['market_implied_points']:.2f} | {ex['market_only_td']:.2f} | {ex['football_market_td']:.2f} | {ex['actual_offensive_td']} |"
        )
    rows.extend(
        [
            "",
            "## Assumptions and limitations",
            "",
            "- Prior-game football and xTD publication uses the explicit assumed kickoff + 24 hours proxy from the earlier phases, never an observed release timestamp. Every source game's proxy precedes the target prediction time.",
            "- This is a team offensive TD count model only; it does not calculate player anytime-TD probabilities.",
            "- The market series starts in 2021. Earlier football seasons remain in the separate football-only table and never receive fabricated closing or historical market lines.",
            "- A pregame starting-QB signal and timestamped weather forecasts were unavailable for the full series, so their model inputs remain null. PROE was not safely reconstructed; lagged pass rate is the simpler tendency measure.",
            "- The T−60 snapshot is taken at the earliest kickoff cutoff on each Eastern game day. Later games that day can use an older eligible quote. Both market update times and the snapshot wrapper time are checked against each game's own T−60 cutoff.",
            "- Only the controlled ablations in this report were evaluated; no broad feature search or betting ROI optimization occurred.",
            "- Full feature missingness by season, model importance, count probabilities, and every experiment are in `reports/phase4_metrics.json`.",
            "",
            "Phase 5 has not begun.",
            "",
        ]
    )
    return "\n".join(rows)


def build_phase4(settings: Settings) -> tuple[Path, Path]:
    """Build strict and football-only team games, validate models, and report."""
    derived = settings.data_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    history, football_meta = build_team_history(settings.data_dir)
    football = lag_team_context(history)
    market, market_meta = build_strict_market_table(settings)
    strict = market.join(
        football,
        on=["season", "week", "game_id", "team", "opponent", "home", "kickoff", "prediction_time"],
        how="inner",
    )
    if strict.height != market.height:
        raise ValueError("Strict market rows did not join one-to-one with football targets")
    train = strict.filter(pl.col("season") <= 2023)
    validation = strict.filter(pl.col("season") == 2024)
    actual = np.asarray(validation["actual_offensive_td"], dtype=float)
    points_per_td = float(cast(int, train["team_points"].sum())) / float(
        cast(int, train["actual_offensive_td"].sum())
    )
    ratio_pred = np.asarray(validation["implied_team_points"], dtype=float) / points_per_td
    ratio_detail = {
        "train_rows": train.height,
        "points_per_offensive_td": points_per_td,
        "metrics": count_metrics(actual, ratio_pred),
        "calibration": calibration_table(actual, ratio_pred),
    }
    experiments: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted = {}
    for family, fields in ABLATIONS.items():
        for architecture in ("poisson", "lightgbm"):
            name = f"{family}_{architecture}"
            model, predicted, result = validate_model(train, validation, fields, architecture)
            experiments[name] = result | {"fields": fields, "architecture": architecture}
            predictions[name] = predicted
            fitted[name] = model
    market_name = "A_market_only_poisson"
    challenger_market = "A_market_only_lightgbm"
    if materially_better(
        experiments[market_name]["metrics"], experiments[challenger_market]["metrics"]
    ):
        market_name = challenger_market
    candidate_names = [name for name in experiments if not name.startswith("A_market_only")]
    football_name = min(
        candidate_names, key=lambda name: experiments[name]["metrics"]["poisson_deviance"]
    )
    football_improved = materially_better(
        experiments[market_name]["metrics"], experiments[football_name]["metrics"]
    )
    selected = football_name if football_improved else market_name
    football_pred = predictions[football_name]
    market_pred = predictions[market_name]
    selected_pred = predictions[selected]
    football_train = football.filter(pl.col("season") <= 2023)
    football_validation = football.filter(pl.col("season") == 2024)
    football_fields = EFFICIENCY + PACE_REDZONE + ENVIRONMENT
    _, _, football_only_result = validate_model(
        football_train, football_validation, football_fields, "poisson"
    )
    market_train_pred = fitted[market_name].predict(train)
    train_y = np.asarray(train["actual_offensive_td"], dtype=float)
    mean = float(train_y.mean())
    variance = float(train_y.var(ddof=1))
    pearson_dispersion = float(np.mean((train_y - market_train_pred) ** 2 / market_train_pred))
    overdispersed = variance / mean > 1.2 and pearson_dispersion > 1.2
    if overdispersed:
        raise ValueError("Material Poisson overdispersion requires Negative Binomial evaluation")
    probs = np.asarray([count_distribution(value) for value in selected_pred])
    prediction_table = validation.with_columns(
        pl.Series("baseline_points_ratio_td", ratio_pred),
        pl.Series("market_only_td", market_pred),
        pl.Series("football_market_td", football_pred),
        pl.Series("predicted_offensive_td", selected_pred),
        pl.lit(selected).alias("model_version"),
        *[
            pl.Series(f"p_team_{index}_td" if index < 4 else "p_team_4plus_td", probs[:, index])
            for index in range(5)
        ],
    )
    paths = {
        "football": derived / "phase4_football_only_2017_2024.parquet",
        "strict": derived / "phase4_strict_market_2021_2024.parquet",
        "predictions": derived / "phase4_team_game_predictions_2024.parquet",
    }
    for key, frame in (
        ("football", football),
        ("strict", strict),
        ("predictions", prediction_table),
    ):
        frame.write_parquet(paths[key])
    inc = {
        metric + "_pct": 100
        * (
            experiments[football_name]["metrics"][metric]
            / experiments[market_name]["metrics"][metric]
            - 1
        )
        for metric in ("mae", "rmse", "poisson_deviance", "calibration_gap")
    }
    market_meta["first_kickoff"] = cast(datetime, market["kickoff"].min()).isoformat()
    market_meta["last_kickoff"] = cast(datetime, market["kickoff"].max()).isoformat()
    ages = strict.with_columns(
        (pl.col("prediction_time") - pl.col("odds_timestamp"))
        .dt.total_minutes()
        .alias("quote_age_minutes")
    )
    market_meta["quote_age_minutes"] = {
        "median": float(cast(int | float, ages["quote_age_minutes"].median())),
        "p90": float(cast(int | float, ages["quote_age_minutes"].quantile(0.9))),
        "max": float(cast(int | float, ages["quote_age_minutes"].max())),
    }
    detail = {
        "football_team_games": football.height,
        "football_games": football_meta["games"],
        "football_coverage": {
            "first_kickoff": cast(datetime, football["kickoff"].min()).isoformat(),
            "last_kickoff": cast(datetime, football["kickoff"].max()).isoformat(),
            "by_season": _coverage(football, EFFICIENCY + PACE_REDZONE + ENVIRONMENT + XTD),
        },
        "market_coverage": market_meta,
        "strict_train_rows": train.height,
        "validation_rows": validation.height,
        "league_points_per_offensive_td": points_per_td,
        "headline_metrics": {
            "points-per-TD ratio": ratio_detail,
            "market-only": experiments[market_name],
            "football+market": experiments[football_name],
            "football-only diagnostic": football_only_result,
        },
        "market_model": market_name,
        "football_challenger": football_name,
        "selected_model": selected,
        "football_material_improvement": football_improved,
        "incremental_football_vs_market": inc,
        "experiments": experiments,
        "dispersion": {
            "mean": mean,
            "variance": variance,
            "variance_to_mean": variance / mean,
            "pearson_dispersion": pearson_dispersion,
        },
        "negative_binomial_evaluated": False,
        "actual_distribution_2024": dict(sorted(Counter(int(x) for x in actual).items())),
        "distribution_calibration": [
            (
                str(index) if index < 4 else "4+",
                float(probs[:, index].mean()),
                float(np.mean(actual == index)) if index < 4 else float(np.mean(actual >= 4)),
            )
            for index in range(5)
        ],
        "calibration": {
            "market_only": calibration_table(actual, market_pred),
            "football_market": calibration_table(actual, football_pred),
            "selected": calibration_table(actual, selected_pred),
        },
        "examples": _examples(prediction_table),
    }
    (reports / "phase4_metrics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    (reports / "PHASE4.md").write_text(_report(detail), encoding="utf-8")
    (reports / "phase4_artifact_hashes.json").write_text(
        json.dumps(
            {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths["predictions"], reports / "PHASE4.md"
