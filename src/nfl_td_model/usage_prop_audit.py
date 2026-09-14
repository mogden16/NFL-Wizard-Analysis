"""Audit the largest live usage-prop disagreements without reading current outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.phase46_predict import normalize_name
from nfl_td_model.usage_props import _season_stats


def audit_large_disagreements(
    report_path: Path = Path("reports/usage_prop_opportunities_2026-09-14.csv"),
    limit: int = 10,
) -> Path:
    """Join live names to frozen 2024 prior rows; current-season availability is unknown."""
    report = (pl.read_csv(report_path)
              .with_columns(pl.col("Model Minus Market").abs().alias("_abs_disagreement"))
              .sort("_abs_disagreement", descending=True, nulls_last=True)
              .drop("_abs_disagreement"))
    rows = report.head(limit).to_dicts()
    features = pl.read_parquet("data/derived/phase2_2024_player_features.parquet")
    stats = _season_stats(2024).select("game", "player_id", "carries", "receptions", "targets")
    feature_rows: dict[str, list[dict[str, Any]]] = {}
    for row in features.sort("kickoff_time").to_dicts():
        feature_rows.setdefault(normalize_name(str(row["player"])), []).append(row)
    output = []
    for live in rows:
        name = normalize_name(str(live["Player"]))
        candidates = feature_rows.get(name, [])
        ids = {row["player_id"] for row in candidates}
        prior_teams = sorted({row["team"] for row in candidates})
        latest = candidates[-1] if candidates else None
        history = []
        if latest:
            ids_used = [value for value in str(latest["player_source_game_ids"] or "").split(";") if value]
            history_rows = stats.filter(pl.col("player_id") == latest["player_id"]).filter(
                pl.col("game").is_in(ids_used)
            ).to_dicts()
            by_game = {row["game"]: row for row in history_rows}
            history = [{"game": game, "receptions": by_game.get(game, {}).get("receptions"),
                        "carries": by_game.get(game, {}).get("carries"),
                        "targets": by_game.get(game, {}).get("targets")} for game in ids_used]
        output.append({
            "player": live["Player"], "prop": live["Prop"], "line": live["Sportsbook Line"],
            "model_mean": live["Model Mean"], "model_probability_over": live["P(Over)"],
            "market_no_vig_over": live["Market No-Vig P(Over)"], "books": live["Books Quoting"],
            "quote_age_minutes": live["Quote Age"], "quote_time": live.get("Quote Timestamp"),
            "exact_line_group": {"player": live["Player"], "prop": live["Prop"], "line": live["Sportsbook Line"]},
            "current_team": live["Team"], "prior_season_teams": prior_teams,
            "current_position": latest["position"] if latest else None,
            "current_season_games_available": 0,
            "ewma_source_games": latest["player_history_games"] if latest else 0,
            "last_3_actual_values": history[-3:], "last_5_actual_values": history[-5:],
            "last_8_actual_values": history[-8:],
            "identity_ambiguity": len(ids) != 1,
            "team_identity_ambiguity": len(prior_teams) > 1,
            "line_identity_ambiguity": False,
            "diagnostic": "No current-game outcome or 2025 data loaded",
        })
    output_path = Path("reports/usage_prop_disagreement_audit.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(audit_large_disagreements())
