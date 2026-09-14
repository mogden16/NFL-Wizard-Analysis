"""Phase 7 chronological, quote and settlement integrity tests."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl_td_model.phase7_analysis import _graded, _rule_rows, portfolio
from nfl_td_model.phase7_stage_a import FORBIDDEN, PREGAME_FIELDS, _flatten
from nfl_td_model.phase7_stage_b import verify_frozen_rule, verify_stage_a


def _payload(snapshot: str, quote: str) -> dict[str, object]:
    return {"timestamp": snapshot, "data": {"bookmakers": [{"key": "testbook",
            "markets": [{"key": "player_anytime_td", "last_update": quote,
                         "outcomes": [{"name": "Yes", "description": "Sample Runner", "price": 180}]}]}]}}


def test_stage_a_rejects_post_cutoff_snapshot_and_quote() -> None:
    cutoff = datetime(2024, 10, 1, 16, tzinfo=UTC)
    with pytest.raises(ValueError, match="Later sportsbook snapshot"):
        _flatten(_payload("2024-10-01T16:01:00+00:00", "2024-10-01T15:59:00+00:00"), cutoff)
    assert _flatten(_payload("2024-10-01T15:59:00+00:00", "2024-10-01T16:01:00+00:00"), cutoff) == []
    assert len(_flatten(_payload("2024-10-01T15:59:00+00:00",
                                 "2024-10-01T15:58:00+00:00"), cutoff)) == 1


def test_stage_a_closed_schema_has_no_results() -> None:
    assert not FORBIDDEN.intersection(PREGAME_FIELDS)
    path = Path("data/phase7/pregame_player_view.parquet")
    if path.exists():
        assert set(pl.scan_parquet(path).collect_schema().names()) == PREGAME_FIELDS
    frozen = Path("reports/phase7/stage_a/predictions.csv")
    if frozen.exists():
        columns = set(pl.scan_csv(frozen).collect_schema().names())
        assert not FORBIDDEN.intersection(columns)
        assert "actual_td_count" not in columns
        assert "closing_same_book_decimal" not in columns


def test_stage_a_worker_does_not_import_result_or_settlement_modules() -> None:
    source = Path("src/nfl_td_model/phase7_stage_a.py").read_text(encoding="utf-8")
    modules = {alias.name for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Import) for alias in node.names}
    modules.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any("stage_b" in name or "settle" in name or "xtd_data" in name
                   or "nflreadpy" in name for name in modules)


def test_phase7_refuses_holdout_and_exhibition() -> None:
    for season in (2025, 2026):
        with pytest.raises(ValueError, match="sealed"):
            _graded(season)


def test_frozen_artifact_and_development_rule_hashes() -> None:
    if not Path("reports/phase7/stage_a/manifest.json").exists():
        pytest.skip("Phase 7 artifacts not built")
    manifest = verify_stage_a()
    assert manifest["seasons"] == [2023, 2024]
    rule = verify_frozen_rule()
    assert rule["development_season"] == 2023
    assert rule["validation_season"] == 2024
    assert rule["model_version"] == "phase5_hierarchical_frozen"


def test_rule_grid_uses_both_edge_and_ev_and_book_count() -> None:
    rows = [{"edge_best": 0.04, "ev_best": 0.11, "books": 2},
            {"edge_best": 0.06, "ev_best": 0.04, "books": 3},
            {"edge_best": 0.08, "ev_best": 0.15, "books": 1}]
    assert _rule_rows(rows, 0.025, 0.10, 2) == rows[:1]


def test_flat_stake_american_odds_settlement_math() -> None:
    sample = [{"game": "2023_01_A_B", "kickoff": "2023-09-01", "player": "Winner",
               "anytime_td": 1, "profit_best_units": 2.0, "profit_median_units": 1.5,
               "best_american": 200, "clv_best_probability": 0.02,
               "clv_median_probability": 0.01},
              {"game": "2023_02_A_B", "kickoff": "2023-09-02", "player": "Loser",
               "anytime_td": 0, "profit_best_units": -1.0, "profit_median_units": -1.0,
               "best_american": 150, "clv_best_probability": None,
               "clv_median_probability": None}]
    result = portfolio(sample, bootstrap=True)
    assert result["bets"] == 2
    assert result["best"]["units"] == 1
    assert result["best"]["roi"] == 0.5
    assert result["median"]["units"] == 0.5
    assert result["clv_coverage"] == 0.5


def test_settlement_unknown_policy_is_not_graded() -> None:
    manifest_path = Path("reports/phase7/settlement_2023_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Phase 7 settlement not built")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["policy_unknown"] > 0
    settled = pl.read_csv("reports/phase7/settlement_2023.csv")
    unknown = settled.filter(pl.col("settlement_status") == "policy_unknown")
    assert unknown["profit_best_units"].null_count() == unknown.height
    assert unknown["anytime_td"].null_count() == unknown.height
