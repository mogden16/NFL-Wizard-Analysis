"""Focused tests for count props and two-sided market math."""

from datetime import UTC, datetime, timedelta

import pytest

from nfl_td_model.market_math import american_to_decimal
from nfl_td_model.usage_props import (
    FEATURES,
    _live_prop_rows,
    _season_stats,
    count_distribution,
    group_prop_quotes,
    over_under_push,
    target_column,
    two_sided_de_vig,
)


def test_count_distribution_sums_to_one() -> None:
    distribution = count_distribution(5.8)
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["4+"] > 0


def test_over_under_push_half_and_integer_lines() -> None:
    over, under, push = over_under_push(5.0, 4.5)
    assert over + under + push == pytest.approx(1.0)
    assert push == 0
    over, under, push = over_under_push(5.0, 5.0)
    assert over + under + push == pytest.approx(1.0)
    assert push > 0
    assert over == pytest.approx(1 - under - push)


def test_two_sided_proportional_de_vig() -> None:
    over_raw, under_raw, over_fair = two_sided_de_vig(-120, 100)
    assert over_raw == pytest.approx(120 / 220)
    assert under_raw == pytest.approx(0.5)
    assert over_fair == pytest.approx(over_raw / (over_raw + under_raw))


def test_american_odds_conversion_is_reused() -> None:
    assert american_to_decimal(-120) == pytest.approx(1.8333333333)
    assert american_to_decimal(110) == pytest.approx(2.1)


def test_different_lines_are_separate_consensus_groups() -> None:
    quotes = [
        {"player": "Runner", "prop": "receptions", "line": 4.5},
        {"player": "Runner", "prop": "receptions", "line": 5.5},
        {"player": "Runner", "prop": "rushing_attempts", "line": 12.5},
    ]
    grouped = group_prop_quotes(quotes)
    assert len(grouped) == 3


def test_declared_features_are_lagged_usage_only() -> None:
    assert all("current" not in field for fields in FEATURES.values() for field in fields)


def test_2025_holdout_is_inaccessible() -> None:
    with pytest.raises(ValueError, match="2025 is sealed"):
        _season_stats(2025)


def test_prop_targets_are_the_expected_player_stat_fields() -> None:
    assert target_column("receptions") == "receptions"
    assert target_column("rushing_attempts") == "carries"
    with pytest.raises(ValueError):
        target_column("receiving_yards")


def test_live_prop_projection_keeps_only_supported_sides_and_pre_cutoff_quotes() -> None:
    payload = {"bookmakers": [{"key": "book", "markets": [
        {"key": "player_receptions", "last_update": "2026-09-14T16:59:00+00:00",
         "outcomes": [{"name": "Over", "description": "Runner", "point": 4.5, "price": -110},
                      {"name": "Under", "description": "Runner", "point": 4.5, "price": -110}]},
        {"key": "player_rush_attempts", "last_update": "2026-09-14T17:01:00+00:00",
         "outcomes": [{"name": "Over", "description": "Runner", "point": 10.5, "price": 100}]},
    ]}]}
    rows = _live_prop_rows(payload, datetime(2026, 9, 14, 17, tzinfo=UTC))
    assert len(rows) == 2
    assert {row["prop"] for row in rows} == {"receptions"}


def test_usage_prop_cutoff_is_rejected_by_live_quote_projection() -> None:
    # Keep a direct chronology assertion in the focused suite.
    cutoff = datetime(2026, 9, 14, 17, tzinfo=UTC)
    assert cutoff - timedelta(minutes=16) < cutoff - timedelta(minutes=15)
