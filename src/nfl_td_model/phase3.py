"""Chronological Phase 3 play scoring, aggregation, and research reports."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.xtd_data import (
    RECEIVING_CATEGORICAL,
    RECEIVING_NUMERIC,
    RUSH_CATEGORICAL,
    RUSH_NUMERIC,
    extract_opportunities,
)
from nfl_td_model.xtd_features import aggregate_games, lagged_player_games
from nfl_td_model.xtd_models import (
    FittedXTD,
    calibration_table,
    choose_model,
    fit_chronological,
    metrics,
)

LOGGER = logging.getLogger(__name__)
KINDS = ("rushing", "receiving")
ARCHITECTURES = ("logistic", "lightgbm")


def _validation(
    opportunities: pl.DataFrame, kind: str, architecture: str
) -> tuple[FittedXTD, dict[str, Any]]:
    model, fit_details = fit_chronological(opportunities, kind, architecture, 2024)
    frame = opportunities.filter((pl.col("season") == 2024) & (pl.col("opportunity_type") == kind))
    probabilities = model.predict(frame)
    labels = np.asarray(frame["actual_td"], dtype=int)
    return model, {
        "architecture": architecture,
        "fit": fit_details,
        "validation_rows": frame.height,
        "validation_tds": int(labels.sum()),
        "metrics": metrics(labels, probabilities),
        "calibration": calibration_table(labels, probabilities),
        "top_features": model.top_features(),
    }


def _scenario_rows(kind: str) -> pl.DataFrame:
    common: dict[str, Any] = {
        "down": 1.0,
        "ydstogo": 10.0,
        "qtr": 2.0,
        "game_seconds_remaining": 1800.0,
        "half_seconds_remaining": 900.0,
        "score_differential": 0.0,
        "shotgun": 0.0,
        "no_huddle": 0.0,
        "position": "RB" if kind == "rushing" else "WR",
    }
    rows = []
    for yardline in (1, 2, 5, 10, 20, 35, 50):
        for variant in (0, 1):
            row = dict(common)
            row.update(
                yardline_100=float(yardline),
                log_yardline=float(np.log1p(yardline)),
                inside_5=int(yardline <= 5),
                inside_10=int(yardline <= 10),
                inside_20=int(yardline <= 20),
            )
            if kind == "rushing":
                row.update(
                    goal_to_go=float(variant),
                    qb_scramble=0.0,
                    run_location="middle",
                    run_gap="guard",
                    scenario=f"{yardline}-yard line; goal_to_go={variant}",
                )
            else:
                air_yards = float(yardline + 1 if variant else 3)
                row.update(
                    goal_to_go=0.0,
                    air_yards=air_yards,
                    relative_to_endzone=air_yards - yardline,
                    end_zone_target=int(air_yards >= yardline),
                    pass_location="middle",
                    scenario=f"{yardline}-yard line; air_yards={air_yards:g}",
                )
            rows.append(row)
    return pl.DataFrame(rows)


def _report(
    opportunities: pl.DataFrame,
    scored: pl.DataFrame,
    aggregates: pl.DataFrame,
    lagged: pl.DataFrame,
    coverage: dict[str, Any],
    comparisons: dict[str, Any],
    models: dict[str, FittedXTD],
    player_names: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    examples = {}
    sanity = {}
    for kind in KINDS:
        scenarios = _scenario_rows(kind)
        probs = models[kind].predict(scenarios)
        examples[kind] = [
            {"scenario": row["scenario"], "xtd": float(prob)}
            for row, prob in zip(scenarios.to_dicts(), probs, strict=True)
        ]
        if kind == "rushing":
            one = probs[1]
            twenty = probs[9]
            goal = probs[1]
            non_goal = probs[0]
            sanity[kind] = {
                "one_vs_twenty_ratio": float(one / twenty),
                "goal_to_go_one_vs_non_goal_one": float(goal / non_goal),
                "one_substantially_greater_than_twenty": bool(one >= twenty * 2),
                "goal_to_go_greater": bool(goal > non_goal),
            }
        else:
            sanity[kind] = {
                "end_zone_one_vs_short_fifty_ratio": float(probs[1] / probs[12]),
                "end_zone_target_greater": bool(probs[1] > probs[12]),
            }
    player_examples = (
        aggregates.with_columns((pl.col("actual_td") - pl.col("total_xtd")).alias("td_minus_xtd"))
        .group_by("season", "player_id")
        .agg(
            pl.col("actual_td").sum(),
            pl.col("total_xtd").sum(),
            pl.col("rushing_xtd").sum(),
            pl.col("receiving_xtd").sum(),
            pl.col("td_minus_xtd").sum(),
            pl.col("game").n_unique().alias("games"),
        )
    )
    over = player_examples.sort("td_minus_xtd", descending=True).head(8).to_dicts()
    under = player_examples.sort("td_minus_xtd").head(8).to_dicts()
    for example in over + under:
        example["player"] = player_names.get(example["player_id"], example["player_id"])
    feature_cols = [
        col
        for col in lagged.columns
        if col.startswith(("last3_", "last5_", "last8_", "season_", "ewma_"))
    ]
    feature_coverage: dict[str, Any] = {
        str(season): {
            "rows": frame.height,
            "with_xtd_history": int((frame["xtd_history_games"] > 0).sum()),
            "families": {
                col: {
                    "present": int(frame[col].is_not_null().sum()),
                    "missing": int(frame[col].is_null().sum()),
                    "coverage_pct": round(100 * frame[col].is_not_null().sum() / frame.height, 2),
                }
                for col in feature_cols
            },
        }
        for season, frame in lagged.partition_by("season", as_dict=True).items()
    }
    # Polars returns tuple keys for partition_by with as_dict.
    feature_coverage = {key.strip("(),"): value for key, value in feature_coverage.items()}
    context_coverage: dict[str, Any] = {}
    for season in range(2017, 2025):
        context_coverage[str(season)] = {}
        for kind in KINDS:
            frame = opportunities.filter(
                (pl.col("season") == season) & (pl.col("opportunity_type") == kind)
            )
            numeric = RUSH_NUMERIC if kind == "rushing" else RECEIVING_NUMERIC
            categorical = RUSH_CATEGORICAL if kind == "rushing" else RECEIVING_CATEGORICAL
            context_coverage[str(season)][kind] = {
                "rows": frame.height,
                "feature_present_pct": {
                    key: round(
                        100
                        * (
                            frame[key].is_not_null().sum()
                            if key in numeric
                            else frame.filter(pl.col(key) != "Unknown").height
                        )
                        / frame.height,
                        2,
                    )
                    for key in (*numeric, *categorical)
                },
            }
    detail = {
        "opportunities_by_season": coverage,
        "modeled_rows": {
            kind: int((opportunities["opportunity_type"] == kind).sum()) for kind in KINDS
        },
        "scored_rows": scored.height,
        "aggregate_rows": aggregates.height,
        "lagged_rows": lagged.height,
        "seasons": list(range(2017, 2025)),
        "comparison": comparisons,
        "sanity": sanity,
        "probability_examples": examples,
        "actual_minus_xtd_examples": {"over": over, "under": under},
        "lagged_feature_coverage": feature_coverage,
        "play_context_coverage": context_coverage,
    }
    lines = [
        "# Phase 3 completion report",
        "",
        "Play-level scoring opportunities span the 2017–2024 regular seasons. 2017 is training-only: it has no earlier season from which to fit a model. Scored and lagged xTD artifacts therefore span 2018–2024. The protected 2025 holdout is never loaded or scored.",
        "",
        f"Modeled opportunities: {int((opportunities['opportunity_type'] == 'rushing').sum()):,} rushing and {int((opportunities['opportunity_type'] == 'receiving').sum()):,} receiving. Scored opportunities: {scored.height:,}; player-game aggregates: {aggregates.height:,}; lagged player-game rows: {lagged.height:,}.",
        "",
        "2017–2022 fit and 2023 calibration checks precede chronological 2024 validation. Historical feature production uses the preregistered logistic baseline, independent of 2024 model-selection outcomes. Each target season is scored by a model fit on earlier seasons only. LightGBM must lower both 2024 log loss and Brier by at least 1% without materially worse calibration to be selected for future use. Accuracy is not a selection criterion.",
        "",
    ]
    for kind in KINDS:
        item = comparisons[kind]
        lines.extend([f"## {kind.title()}", ""])
        for architecture in ARCHITECTURES:
            v = item[architecture]
            m = v["metrics"]
            lines.append(
                f"{architecture}: log loss {m['log_loss']:.6f}; Brier {m['brier']:.6f}; calibration gap {m['calibration_gap']:.6f} ({v['validation_rows']:,} 2024 plays, {v['validation_tds']} TDs)."
            )
        lines.extend(
            [
                f"Selected: **{item['selected']}** — {item['selection_reason']}.",
                f"Sigmoid calibration applied: **{item[item['selected']]['fit']['calibration']['selected']}** (selected only if prior-season late-week log loss and Brier both improve).",
                "",
                "| Bin | Plays | Mean xTD | TD rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for bin_row in item[item["selected"]]["calibration"]:
            mean = "—" if bin_row["mean_predicted"] is None else f"{bin_row['mean_predicted']:.4f}"
            actual = (
                "—" if bin_row["actual_td_rate"] is None else f"{bin_row['actual_td_rate']:.4f}"
            )
            lines.append(f"| {bin_row['bin']} | {bin_row['observations']} | {mean} | {actual} |")
        lines.extend(
            [
                "",
                f"Leading fitted features: {', '.join(str(f['feature']) for f in item[item['selected']]['top_features'][:8])}.",
                f"Sanity: {json.dumps(sanity[kind])}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Actual touchdowns versus xTD",
            "",
            "These are regular-season player totals over modeled rushes and targets. They illustrate finishing above or below opportunity-based expectation; they are not estimates of future ability.",
            "",
            "| Player | Season | Actual TD | xTD | Difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for example in over[:5] + under[:5]:
        lines.append(
            f"| {example['player']} | {example['season']} | {example['actual_td']} | {example['total_xtd']:.2f} | {example['td_minus_xtd']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Lagged feature availability",
            "",
            "A window's `*_xtd` is mean xTD per game; `*_xtd_sum` is the total across eligible games. The season window resets each season. Every value uses earlier games only.",
            "",
            "| Season | Candidate rows | Rows with prior xTD | Last-3 total xTD present | Season-to-date total xTD present |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for season_label, values in sorted(feature_coverage.items()):
        lines.append(
            f"| {season_label} | {values['rows']:,} | {values['with_xtd_history']:,} | {values['families']['last3_total_xtd']['coverage_pct']:.2f}% | {values['families']['season_total_xtd']['coverage_pct']:.2f}% |"
        )
    lines.append("")
    lines.extend(
        [
            "## Assumptions and limitations",
            "",
            "- Only primary rushing and targeted receiving opportunities are modeled. Two-point plays, kneels, spikes, deleted plays and lateral/ambiguous scoring plays are excluded; the source coverage file counts excluded ambiguous TD plays.",
            "- Rusher/receiver position uses a prior same-team weekly player-stat position. Prior-game publication is an explicit **assumed kickoff + 24 hours** proxy, never an observed release timestamp. Unknown position is retained rather than filled from the target game.",
            "- No historical team implied total, market field, route participation, or player identity is used in the xTD model. Complete timestamped historical market coverage is unavailable.",
            "- Run gap is retained with an explicit Unknown category; it is unavailable on roughly 27% of rushing plays. Prior-game position is Unknown on roughly 9% of opportunities. Per-season and per-field model-input coverage is in `reports/phase3_metrics.json`.",
            "- Model inputs are context known at the play, suitable for opportunity-quality measurement. These current-play inputs are never used as pregame features for that same game; only earlier-game aggregates pass the strict lag filter.",
            "- Sigmoid calibration is fitted on weeks 1–9 of the prior season only and kept only when it improves both log loss and Brier on weeks 10+. The 2024 validation season does not fit calibration or model parameters.",
            "- The architecture comparison uses 2024 outcome labels, but historical xTD features are always produced by the preregistered logistic baseline. A later 2024 downstream model-selection exercise should still account for this reuse of the validation year. 2025 remains untouched for a final holdout.",
            "- Historical lagged xTD is available only after an earlier game with a modeled opportunity. Rates and shares are null when there is no observed denominator. Missing features remain null.",
            "- 2017 xTD is unavailable by design because no earlier training season is present. Phase 4 has not started.",
            "",
            "Detailed calibration for both models, example probability tables, per-season feature missingness, and actual TD versus xTD player examples are in `reports/phase3_metrics.json`.",
        ]
    )
    return "\n".join(lines) + "\n", detail


def build_phase3(settings: Settings) -> tuple[Path, Path]:
    """Build reproducible 2017–2024 opportunity evidence and lagged xTD features."""
    data_dir = settings.data_dir
    derived = data_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    opportunities, coverage = extract_opportunities(data_dir)
    LOGGER.info("Extracted %s play opportunities", opportunities.height)
    comparisons: dict[str, Any] = {}
    chosen: dict[str, FittedXTD] = {}
    baseline: dict[str, FittedXTD] = {}
    for kind in KINDS:
        candidates = {
            architecture: _validation(opportunities, kind, architecture)
            for architecture in ARCHITECTURES
        }
        selected, reason = choose_model(
            candidates["logistic"][1]["metrics"], candidates["lightgbm"][1]["metrics"]
        )
        comparisons[kind] = {
            architecture: candidates[architecture][1] for architecture in ARCHITECTURES
        } | {"selected": selected, "selection_reason": reason}
        chosen[kind] = candidates[selected][0]
        baseline[kind] = candidates["logistic"][0]
        LOGGER.info("%s selected %s", kind, selected)
    scored_parts = []
    fit_manifest = {}
    for season in range(2018, 2025):
        for kind in KINDS:
            # Freeze historical features to the predeclared baseline; model selection
            # on 2024 outcomes must not change any pregame feature for 2024.
            architecture = "logistic"
            if season == 2024:
                model = baseline[kind]
                diagnostics = comparisons[kind][architecture]["fit"]
            else:
                model, diagnostics = fit_chronological(opportunities, kind, architecture, season)
            frame = opportunities.filter(
                (pl.col("season") == season) & (pl.col("opportunity_type") == kind)
            )
            scores = model.predict(frame)
            scored_parts.append(
                frame.with_columns(
                    pl.Series("xtd", scores),
                    pl.lit(model.architecture).alias("model_architecture"),
                    pl.lit(model.fit_max_season).alias("model_fit_max_season"),
                    pl.lit(model.calibration_season, dtype=pl.Int32).alias(
                        "model_calibration_season"
                    ),
                )
            )
            fit_manifest[f"{season}_{kind}"] = diagnostics | {
                "architecture": architecture,
                "calibration_selected": model.calibration_selected,
            }
        LOGGER.info("Scored season %s", season)
    scored = pl.concat(scored_parts).sort("season", "game", "play_id", "opportunity_type")
    aggregates = aggregate_games(scored)
    # Load only pre-holdout candidate rows. The 2025 Phase 2 table is never read.
    phase2 = pl.concat(
        [
            pl.read_parquet(derived / f"phase2_{season}_player_features.parquet")
            for season in range(2018, 2025)
        ]
    ).with_columns(
        pl.col("game").str.slice(0, 4).cast(pl.Int32).alias("season"),
        pl.col("game").str.slice(5, 2).cast(pl.Int32).alias("week"),
    )
    lagged = lagged_player_games(phase2, aggregates)
    paths = {
        "opportunities": derived / "phase3_2017_2024_opportunities.parquet",
        "scored": derived / "phase3_2018_2024_play_scores.parquet",
        "aggregates": derived / "phase3_2018_2024_player_game_xtd.parquet",
        "lagged": derived / "phase3_2018_2024_lagged_xtd_features.parquet",
    }
    for key, frame in (
        ("opportunities", opportunities),
        ("scored", scored),
        ("aggregates", aggregates),
        ("lagged", lagged),
    ):
        frame.write_parquet(paths[key])
    player_names = {
        row["player_id"]: row["player"]
        for row in phase2.select("player_id", "player").unique().to_dicts()
    }
    report, detail = _report(
        opportunities, scored, aggregates, lagged, coverage, comparisons, chosen, player_names
    )
    (reports / "PHASE3.md").write_text(report, encoding="utf-8")
    (reports / "phase3_metrics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    (reports / "phase3_fit_manifest.json").write_text(
        json.dumps(fit_manifest, indent=2), encoding="utf-8"
    )
    (reports / "phase3_artifact_hashes.json").write_text(
        json.dumps(
            {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths["lagged"], reports / "PHASE3.md"
