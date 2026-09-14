"""Phase 6 nonlinear, position, OOF, and holdout integrity tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_td_model.phase6 import (
    BLENDS,
    POSITION_CORE,
    POSITION_XTD,
    SAFE_FIELDS,
    TREE_FIELDS,
    _blend,
    _cluster_interval,
    _fit_logistic,
    _fit_sigmoid,
    _fit_tree,
    _safe_matrix,
)
from nfl_td_model.phase6_verify import verify_phase6


@pytest.mark.parametrize("season", [2025, 2026])
def test_holdout_and_exhibition_seasons_cannot_enter_player_matrix(season: int) -> None:
    frame = pl.DataFrame({"season": [season], "last5_carry_share": [.2]})
    with pytest.raises(ValueError, match="firewall"):
        _safe_matrix(frame, ("last5_carry_share",))


@pytest.mark.parametrize("field", [
    "anytime_td", "actual_td_count", "p_market", "market_median_decimal",
    "current_game_xtd", "current_game_carries", "current_game_targets",
    "current_game_snaps", "closing_atd_odds", "last5_route_participation",
    "passing_touchdowns", "diagnostic_exhibition_slate",
])
def test_outcome_market_current_game_and_routes_are_not_features(field: str) -> None:
    frame = pl.DataFrame({"season": [2024], field: [1]})
    with pytest.raises(ValueError, match="undeclared or unsafe"):
        _safe_matrix(frame, (field,))


def test_fit_period_and_calibration_period_are_separate() -> None:
    future = pl.DataFrame({"season": [2023], "anytime_td": [1],
                           "last5_carry_share": [.2]})
    with pytest.raises(ValueError, match="only 2022"):
        _fit_logistic(future, ("last5_carry_share",))
    with pytest.raises(ValueError, match="preregistration"):
        _fit_tree(future, "conservative")
    with pytest.raises(ValueError, match="preregistration"):
        _fit_tree(future.with_columns(pl.lit(2022).alias("season")), "unregistered")
    with pytest.raises(ValueError, match="only 2023"):
        _fit_sigmoid(future.with_columns(pl.lit(2024).alias("season")), np.array([.2]))


def test_declared_features_exclude_unsafe_sources() -> None:
    assert all(field in SAFE_FIELDS for fields in POSITION_CORE.values() for field in fields)
    assert all(field in SAFE_FIELDS for fields in POSITION_XTD.values() for field in fields)
    assert all(field in SAFE_FIELDS for field in TREE_FIELDS)
    assert all("route" not in field and "current_game" not in field for field in SAFE_FIELDS)
    assert not {"p_market", "anytime_td", "actual_td_count"}.intersection(SAFE_FIELDS)


def test_fixed_blends_are_normalized_and_deterministic() -> None:
    component = np.array([[.2, .3, .4, .5], [.6, .5, .4, .3]])
    for weights in BLENDS.values():
        assert sum(weights) == pytest.approx(1)
        result = _blend(component, weights)
        assert np.array_equal(result, _blend(component, weights))
        assert np.all((result >= 0) & (result <= 1))
    with pytest.raises(ValueError, match="Unregistered"):
        _blend(component, (.5, .5, .5, -.5))


def test_game_cluster_interval_is_deterministic() -> None:
    games = ["g1", "g1", "g2", "g2", "g3", "g3"]
    labels = np.array([0, 1, 0, 0, 1, 0])
    base = np.full(6, .3)
    challenger = np.array([.2, .5, .2, .2, .5, .2])
    one = _cluster_interval(games, labels, base, challenger)
    assert one == _cluster_interval(games, labels, base, challenger)
    assert one["game_clusters"] == 3


def test_phase6_verifier_rebuilds_2023_oof_components() -> None:
    summary = verify_phase6()
    assert summary["oof_rows"] == 6321
    assert summary["validation_rows"] == 6320
    assert summary["champion"] == "phase5_hierarchical"


def test_qb_atd_labels_count_only_rushing_or_receiving_touchdowns() -> None:
    opportunities = pl.read_parquet(
        "data/derived/phase3_2017_2024_opportunities.parquet",
        columns=["season", "game", "player_id", "position", "opportunity_type", "actual_td"],
    ).filter((pl.col("season") == 2024) & (pl.col("position") == "QB"))
    assert set(opportunities["opportunity_type"].unique().to_list()) <= {"rushing", "receiving"}
    counts = opportunities.group_by("game", "player_id").agg(
        pl.col("actual_td").sum().alias("rushing_or_receiving_td")
    )
    predictions = pl.read_csv("reports/phase6_validation_predictions.csv").filter(
        pl.col("position") == "QB"
    ).join(counts, on=["game", "player_id"], how="left")
    assert predictions.filter(
        pl.col("anytime_td") != (pl.col("rushing_or_receiving_td").fill_null(0) > 0).cast(pl.Int64)
    ).is_empty()
