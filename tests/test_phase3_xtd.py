"""Targeted Phase 3 play-label, aggregation and same-game leakage regressions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nfl_td_model.xtd_data import (
    classify_opportunity,
    end_zone_target,
    play_context,
    relative_to_endzone,
)
from nfl_td_model.xtd_features import aggregate_games, summarize_xtd_history
from nfl_td_model.xtd_models import feature_dicts, fit_chronological


def test_opportunity_labels_classification_and_ambiguous_laterals() -> None:
    rush = {
        "rush_attempt": 1,
        "rusher_player_id": "r1",
        "rush_touchdown": 1,
        "td_player_id": "r1",
        "play_type": "run",
    }
    assert classify_opportunity(rush) == ("rushing", "r1", 1)
    assert classify_opportunity({**rush, "td_player_id": "other"}) is None
    assert classify_opportunity({**rush, "two_point_attempt": 1}) is None
    assert classify_opportunity({**rush, "qb_kneel": 1}) is None
    reception = {
        "pass_attempt": 1,
        "receiver_player_id": "w1",
        "pass_touchdown": 1,
        "td_player_id": "w1",
        "play_type": "pass",
    }
    assert classify_opportunity(reception) == ("receiving", "w1", 1)
    assert classify_opportunity({**reception, "lateral_receiver_player_id": "w2"}) is None
    assert classify_opportunity({**reception, "pass_touchdown": 0}) == ("receiving", "w1", 0)


def test_end_zone_target_and_relative_distance() -> None:
    assert relative_to_endzone(10, 12) == 2
    assert relative_to_endzone(10, 4) == -6
    assert end_zone_target(10, 10) == 1
    assert end_zone_target(10, 4) == 0
    assert end_zone_target(10, None) is None
    assert (
        play_context({"yardline_100": 10, "air_yards": 12}, "receiving", "WR")["end_zone_target"]
        == 1
    )


def test_aggregate_xtd_is_exact_play_sum_and_preserves_actual_td() -> None:
    kickoff = datetime(2023, 10, 1, 17, tzinfo=UTC)
    rows = [
        {
            "season": 2023,
            "game": "g",
            "kickoff_time": kickoff,
            "team": "A",
            "player_id": "p",
            "opportunity_type": kind,
            "xtd": xtd,
            "actual_td": td,
        }
        for kind, xtd, td in (("rushing", 0.2, 1), ("rushing", 0.1, 0), ("receiving", 0.05, 0))
    ]
    rows.append({**rows[0], "player_id": "q", "xtd": 0.3, "actual_td": 0})
    aggregate = aggregate_games(pl.DataFrame(rows))
    player = aggregate.filter(pl.col("player_id") == "p").to_dicts()[0]
    assert player["rushing_xtd"] == pytest.approx(0.3)
    assert player["receiving_xtd"] == pytest.approx(0.05)
    assert player["total_xtd"] == pytest.approx(0.35)
    assert player["actual_td"] == 1
    assert player["rushing_xtd_share"] == pytest.approx(0.5)


def test_same_game_xtd_cannot_enter_any_lagged_window_or_share() -> None:
    kickoff = datetime(2024, 9, 8, 17, tzinfo=UTC)
    target = {
        "season": 2024,
        "week": 1,
        "game": "current",
        "team": "A",
        "player_id": "p",
        "position": "RB",
        "prediction_time": kickoff - timedelta(hours=1),
    }
    base = {
        "season": 2024,
        "team": "A",
        "player_id": "p",
        "rushing_xtd": 0.2,
        "receiving_xtd": 0.1,
        "total_xtd": 0.3,
        "team_rushing_xtd": 0.4,
        "team_receiving_xtd": 0.2,
        "team_total_xtd": 0.6,
        "rushing_xtd_share": 0.5,
        "receiving_xtd_share": 0.5,
        "total_xtd_share": 0.5,
        "rushing_xtd_per_attempt": 0.1,
        "receiving_xtd_per_target": 0.1,
        "total_xtd_per_opportunity": 0.1,
        "rush_attempts": 2,
        "targets": 1,
    }
    prior = {**base, "game": "prior", "kickoff_time": kickoff - timedelta(days=7)}
    current = {**base, "game": "current", "kickoff_time": kickoff, "total_xtd": 99.0}
    before = summarize_xtd_history(target, [prior])
    after = summarize_xtd_history(target, [prior, current])
    assert before == after
    assert after["last3_total_xtd"] == pytest.approx(0.3)
    assert after["last3_total_xtd_sum"] == pytest.approx(0.3)
    assert after["last3_total_xtd_share"] == pytest.approx(0.5)
    assert after["xtd_source_game_ids"] == "prior"


def test_2025_excluded_from_fitting_and_scoring_deterministic() -> None:
    contexts = []
    for season in (2017, 2018, 2025):
        for index in range(120):
            row = play_context(
                {
                    "yardline_100": 1 if index % 10 == 0 else 25,
                    "goal_to_go": int(index % 10 == 0),
                    "down": 1,
                    "ydstogo": 3,
                    "qtr": 2,
                    "game_seconds_remaining": 1600,
                    "half_seconds_remaining": 700,
                    "score_differential": 0,
                    "shotgun": 0,
                    "no_huddle": 0,
                    "qb_scramble": 0,
                    "run_location": "middle",
                    "run_gap": "guard",
                },
                "rushing",
                "RB",
            )
            contexts.append(
                {
                    "season": season,
                    "week": 1,
                    "opportunity_type": "rushing",
                    "actual_td": int(index % 10 == 0),
                    "player_id": f"p{index}",
                    **row,
                }
            )
    frame = pl.DataFrame(contexts)
    assert "player_id" not in feature_dicts(frame.head(1), "rushing")[0]
    model_a, _ = fit_chronological(frame, "rushing", "logistic", 2018)
    model_b, _ = fit_chronological(frame, "rushing", "logistic", 2018)
    assert model_a.fit_max_season == model_b.fit_max_season == 2017
    scores_a = model_a.predict(frame.filter(pl.col("season") == 2018))
    scores_b = model_b.predict(frame.filter(pl.col("season") == 2018))
    assert np.array_equal(scores_a, scores_b)
    assert np.all((scores_a > 0) & (scores_a < 1))
    with pytest.raises(ValueError, match="2018–2024"):
        fit_chronological(frame, "rushing", "logistic", 2025)
