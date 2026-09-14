"""Phase 4 target, market cutoff, lag, and holdout regression tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_td_model.config import Settings
from nfl_td_model.market_math import implied_team_points
from nfl_td_model.phase4_football import (
    OFFENSE_METRICS,
    build_team_history,
    forecast_is_eligible,
    lag_team_context,
    offensive_td_play,
)
from nfl_td_model.phase4_market import (
    MARKET_SEASONS,
    _event_quotes,
    normalize_home_spread,
    quote_is_eligible,
)
from nfl_td_model.phase4_models import MARKET, count_distribution, fit_count_model, validate_model


def test_offensive_td_target_excludes_non_offensive_scores() -> None:
    base = {"posteam": "ATL", "td_team": "ATL", "rush_touchdown": 1, "pass_touchdown": 0}
    assert offensive_td_play(base)
    assert offensive_td_play({**base, "rush_touchdown": 0, "pass_touchdown": 1})
    assert not offensive_td_play({**base, "td_team": "NO"})
    assert not offensive_td_play({**base, "rush_touchdown": 0, "pass_touchdown": 0})
    assert not offensive_td_play({**base, "two_point_attempt": 1})
    assert not offensive_td_play({**base, "play_deleted": 1})


def test_representative_2024_games_agree_with_pbp_and_weekly_team_stats() -> None:
    history, metadata = build_team_history(Settings().data_dir)
    assert metadata["target_mismatches"] == []
    expected = {
        ("2024_04_DAL_NYG", "DAL"): 2,
        ("2024_04_DAL_NYG", "NYG"): 0,
        ("2024_04_NO_ATL", "ATL"): 0,  # Falcons' non-offensive scores do not enter.
        ("2024_04_NO_ATL", "NO"): 3,
        ("2024_04_MIN_GB", "MIN"): 4,
        ("2024_04_MIN_GB", "GB"): 4,
    }
    actual = {
        (row["game_id"], row["team"]): row["actual_offensive_td"]
        for row in history.filter(pl.col("game_id").is_in({key[0] for key in expected})).to_dicts()
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("home_handicap", "total", "expected_home", "expected_away"),
    [(-7.0, 47.0, 27.0, 20.0), (7.0, 47.0, 20.0, 27.0), (0.0, 44.0, 22.0, 22.0)],
)
def test_canonical_spread_and_implied_points(
    home_handicap: float, total: float, expected_home: float, expected_away: float
) -> None:
    assert implied_team_points(total, home_handicap) == pytest.approx(
        (expected_home, expected_away)
    )
    home_margin = -home_handicap
    assert normalize_home_spread(home_margin, convention="home_margin") == home_handicap
    assert normalize_home_spread(home_handicap, convention="home_handicap") == home_handicap
    with pytest.raises(ValueError):
        normalize_home_spread(home_handicap, convention="ambiguous")


def test_quote_and_forecast_must_precede_t_minus_60() -> None:
    kickoff = datetime(2024, 9, 8, 17, tzinfo=UTC)
    cutoff = kickoff - timedelta(hours=1)
    assert quote_is_eligible(cutoff - timedelta(minutes=1), cutoff, cutoff, cutoff)
    assert not quote_is_eligible(cutoff, cutoff + timedelta(seconds=1), cutoff, cutoff)
    assert not quote_is_eligible(cutoff + timedelta(seconds=1), cutoff, cutoff, cutoff)
    assert forecast_is_eligible(cutoff - timedelta(minutes=30), cutoff)
    assert not forecast_is_eligible(cutoff + timedelta(seconds=1), cutoff)
    assert not forecast_is_eligible(None, cutoff)
    event = {
        "id": "event",
        "bookmakers": [
            {
                "key": "book",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": (cutoff + timedelta(minutes=1)).isoformat(),
                        "outcomes": [{"name": "Home", "point": -3}],
                    },
                    {
                        "key": "totals",
                        "last_update": cutoff.isoformat(),
                        "outcomes": [{"name": "Over", "point": 45}],
                    },
                ],
            }
        ],
    }
    assert _event_quotes(event, cutoff, "Home", cutoff) == []  # A later closing line is rejected.


def _team_row(
    game: str, team: str, opponent: str, kickoff: datetime, xtd: float
) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2024,
        "week": 2,
        "game_id": game,
        "team": team,
        "opponent": opponent,
        "home": 1,
        "kickoff": kickoff,
        "prediction_time": kickoff - timedelta(hours=1),
        "actual_offensive_td": 2,
        "team_points": 20,
    }
    row.update(dict.fromkeys(OFFENSE_METRICS, 1.0))
    row["off_xtd"] = xtd
    return row


def test_current_team_opponent_and_xtd_cannot_enter_own_rolling_features() -> None:
    kickoff = datetime(2024, 9, 15, 17, tzinfo=UTC)
    past = kickoff - timedelta(days=7)
    earlier = [
        _team_row("prior_A", "A", "C", past, 1.5),
        _team_row("prior_B", "X", "B", past, 2.0),
    ]
    current = _team_row("target", "A", "B", kickoff, 999.0)
    rival = _team_row("target", "B", "A", kickoff, 999.0)
    baseline = lag_team_context(pl.DataFrame(earlier + [current, rival], infer_schema_length=None))
    altered = lag_team_context(
        pl.DataFrame(
            earlier
            + [
                {**current, "off_xtd": -999.0, "off_epa_per_play": 999.0},
                {**rival, "off_xtd": -999.0, "off_epa_per_play": 999.0},
            ],
            infer_schema_length=None,
        )
    )
    target_base = baseline.filter(
        (pl.col("game_id") == "target") & (pl.col("team") == "A")
    ).to_dicts()[0]
    target_altered = altered.filter(
        (pl.col("game_id") == "target") & (pl.col("team") == "A")
    ).to_dicts()[0]
    assert target_base == target_altered
    assert target_base["off_source_game_ids"] == "prior_A"
    assert target_base["def_source_game_ids"] == "prior_B"
    assert target_base["last5_off_xtd"] == pytest.approx(1.5)
    assert target_base["last5_def_xtd_allowed"] == pytest.approx(2.0)


def test_2025_cannot_enter_training_or_validation_and_distribution_sums_to_one() -> None:
    assert max(MARKET_SEASONS) == 2024
    row = {"season": 2025, "actual_offensive_td": 2, **dict.fromkeys(MARKET, 1.0)}
    with pytest.raises(ValueError, match="2024 validation or 2025"):
        fit_count_model(pl.DataFrame([row]), MARKET, "poisson")
    with pytest.raises(ValueError, match="2024 only"):
        validate_model(
            pl.DataFrame([{**row, "season": 2023}]), pl.DataFrame([row]), MARKET, "poisson"
        )
    assert sum(count_distribution(2.5)) == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in count_distribution(2.5))
    with pytest.raises(ValueError):
        count_distribution(-0.01)
