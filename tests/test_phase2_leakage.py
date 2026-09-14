"""Regression tests for same-game usage and denominator leakage."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_td_model.feature_windows import summarize_candidate
from nfl_td_model.phase2 import assert_complete_pbp, count_red_zone_plays, map_snap_counts


def source_game(game_id: str, kickoff: datetime, **values: float | None) -> dict[str, object]:
    return {
        "game_id": game_id,
        "kickoff": kickoff,
        "available_at_proxy": kickoff + timedelta(days=1),
        **values,
    }


def test_all_usage_shares_ignore_same_game_and_future_game() -> None:
    prediction = datetime(2024, 9, 29, 16, tzinfo=UTC)
    prior = source_game(
        "prior",
        prediction - timedelta(days=7),
        carries=4,
        targets=3,
        touches=6,
        snaps=40,
        red_zone_carries=1,
        inside_10_carries=1,
        inside_5_carries=1,
        goal_line_opportunities=2,
        red_zone_targets=2,
        end_zone_targets=1,
        team_carries=20,
        team_targets=30,
        team_touches=35,
        team_snaps=60,
        team_goal_line_opportunities=4,
        team_red_zone_targets=8,
        team_end_zone_targets=2,
    )
    current = source_game(
        "current",
        prediction + timedelta(hours=1),
        carries=900,
        targets=900,
        touches=900,
        snaps=900,
        red_zone_carries=900,
        inside_10_carries=900,
        inside_5_carries=900,
        goal_line_opportunities=900,
        red_zone_targets=900,
        end_zone_targets=900,
        team_carries=901,
        team_targets=901,
        team_touches=901,
        team_snaps=901,
        team_goal_line_opportunities=901,
        team_red_zone_targets=901,
        team_end_zone_targets=901,
    )
    team_prior = source_game(
        "prior",
        prediction - timedelta(days=7),
        offensive_plays=55,
        carries=20,
        targets=30,
        red_zone_carries=3,
        red_zone_targets=8,
        offensive_touchdowns=3,
    )
    team_current = source_game(
        "current",
        prediction + timedelta(hours=1),
        offensive_plays=900,
        carries=900,
        targets=900,
        red_zone_carries=900,
        red_zone_targets=900,
        offensive_touchdowns=900,
    )
    opp_prior = source_game(
        "opp_prior",
        prediction - timedelta(days=7),
        allowed_offensive_plays=60,
        allowed_carries=24,
        allowed_targets=29,
        allowed_red_zone_carries=4,
        allowed_red_zone_targets=5,
        allowed_offensive_touchdowns=2,
    )
    opp_current = source_game(
        "current",
        prediction + timedelta(hours=1),
        allowed_offensive_plays=900,
        allowed_carries=900,
        allowed_targets=900,
        allowed_red_zone_carries=900,
        allowed_red_zone_targets=900,
        allowed_offensive_touchdowns=900,
    )
    baseline = summarize_candidate(
        [prior], [team_prior], [opp_prior], prediction, game_id="current"
    )
    with_current = summarize_candidate(
        [prior, current],
        [team_prior, team_current],
        [opp_prior, opp_current],
        prediction,
        game_id="current",
    )
    assert with_current == baseline
    assert baseline["last3_carry_share"] == pytest.approx(4 / 20)
    assert baseline["last3_target_share"] == pytest.approx(3 / 30)
    assert baseline["last3_touch_share"] == pytest.approx(6 / 35)
    assert baseline["last3_snap_share"] == pytest.approx(40 / 60)
    assert baseline["last3_goal_line_opportunity_share"] == pytest.approx(2 / 4)
    assert baseline["last3_red_zone_target_share"] == pytest.approx(2 / 8)
    assert baseline["last3_end_zone_target_share"] == pytest.approx(1 / 2)
    assert baseline["last3_opponent_allowed_targets_per_game"] == 29
    assert baseline["player_source_game_ids"] == "prior"
    assert baseline["team_source_game_ids"] == "prior"
    assert baseline["opponent_source_game_ids"] == "opp_prior"
    assert baseline["feature_available_at_max"] < prediction


def test_recent_game_unavailable_at_prediction_is_excluded() -> None:
    prediction = datetime(2024, 9, 29, 16, tzinfo=UTC)
    yesterday = source_game(
        "yesterday", prediction - timedelta(hours=12), carries=8, team_carries=20
    )
    older = source_game("older", prediction - timedelta(days=7), carries=3, team_carries=15)
    features = summarize_candidate([older, yesterday], [], [], prediction, game_id="current")
    assert features["season_carries_per_game"] == 3
    assert features["season_carry_share"] == pytest.approx(3 / 15)
    assert features["player_source_game_ids"] == "older"
    assert features["availability_type"] == "assumed_kickoff_plus_24h"


def test_windows_and_ewma_keep_denominator_aligned_to_player_games() -> None:
    prediction = datetime(2024, 11, 1, 16, tzinfo=UTC)
    history = [
        source_game(
            f"week{i}",
            prediction - timedelta(days=7 * (9 - i)),
            carries=float(i),
            team_carries=float(i * 2),
        )
        for i in range(1, 9)
    ]
    features = summarize_candidate(history, [], [], prediction, game_id="current")
    assert features["last3_player_games"] == 3
    assert features["last5_player_games"] == 5
    assert features["last8_player_games"] == 8
    assert features["season_player_games"] == 8
    assert features["last3_carries_per_game"] == pytest.approx(7)
    for window in ("last3", "last5", "last8", "season", "ewma"):
        assert features[f"{window}_carry_share"] == pytest.approx(0.5)
        assert features[f"{window}_routes_per_game"] is None


def test_red_zone_counter_excludes_two_point_attempts() -> None:
    pbp = pl.DataFrame(
        {
            "game_id": ["old", "old", "old", "old"],
            "posteam": ["ATL"] * 4,
            "rush_attempt": [1, 0, 0, 0],
            "pass_attempt": [0, 1, 1, 1],
            "two_point_attempt": [0, 0, 0, 1],
            "qb_kneel": [0] * 4,
            "rusher_player_id": ["runner", None, None, None],
            "receiver_player_id": [None, "receiver", "receiver", "receiver"],
            "yardline_100": [4.0, 15.0, 4.0, 2.0],
            "air_yards": [None, 16.0, 1.0, None],
        }
    )
    player, team = count_red_zone_plays(pbp, {"old"})
    assert player[("old", "runner")]["inside_5_carries"] == 1
    assert player[("old", "receiver")]["red_zone_targets"] == 2
    assert player[("old", "receiver")]["end_zone_targets"] == 1
    assert team[("old", "ATL")]["goal_line_opportunities"] == 2


def test_unknown_air_yards_remains_missing_across_feature_window() -> None:
    pbp = pl.DataFrame(
        {
            "game_id": ["old"],
            "posteam": ["ATL"],
            "rush_attempt": [0],
            "pass_attempt": [1],
            "two_point_attempt": [0],
            "qb_kneel": [0],
            "rusher_player_id": [None],
            "receiver_player_id": ["receiver"],
            "yardline_100": [2.0],
            "air_yards": [None],
        }
    )
    player, team = count_red_zone_plays(pbp, {"old"})
    assert player[("old", "receiver")]["end_zone_targets"] is None
    assert team[("old", "ATL")]["end_zone_targets"] is None
    prediction = datetime(2024, 9, 29, 16, tzinfo=UTC)
    known = source_game(
        "known",
        prediction - timedelta(days=14),
        end_zone_targets=1,
        team_end_zone_targets=2,
    )
    unknown = source_game(
        "old",
        prediction - timedelta(days=7),
        end_zone_targets=None,
        team_end_zone_targets=None,
    )
    summary = summarize_candidate([known, unknown], [], [], prediction, game_id="current")
    assert summary["last3_end_zone_targets_per_game"] is None
    assert summary["last3_end_zone_target_share"] is None


def test_snap_mapping_requires_unique_crosswalk() -> None:
    snaps = pl.DataFrame(
        {
            "game_id": ["old", "old"],
            "team": ["ATL", "ATL"],
            "pfr_player_id": ["One", "Two"],
            "offense_snaps": [40.0, 60.0],
            "offense_pct": [0.5, 0.75],
        }
    )
    players = pl.DataFrame({"gsis_id": ["gsis-one", "gsis-two"], "pfr_id": ["One", "Two"]})
    player, team, coverage = map_snap_counts(snaps, players, {"old"})
    assert player[("old", "gsis-one")] == 40
    assert team[("old", "ATL")] == 80
    assert coverage["gsis_matched_rows"] == 2


def test_incomplete_pbp_cannot_turn_zone_usage_into_zero() -> None:
    game = {"game_id": "old", "home_score": 7, "away_score": 0}
    with pytest.raises(ValueError, match="Missing or incomplete PBP"):
        assert_complete_pbp(
            pl.DataFrame({"game_id": ["old"], "home_score": [0], "away_score": [0]}),
            [game],
        )
