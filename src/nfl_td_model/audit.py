"""Fail-closed temporal checks shared by batch and future live feature creation."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any


def prediction_timestamp(kickoff_time: datetime, minutes_to_kickoff: int = 60) -> datetime:
    if kickoff_time.tzinfo is None or minutes_to_kickoff <= 0:
        raise ValueError("Kickoff must be timezone-aware and lead time positive")
    return kickoff_time - timedelta(minutes=minutes_to_kickoff)


def select_quote(
    quotes: Sequence[Mapping[str, Any]], prediction_time: datetime
) -> Mapping[str, Any] | None:
    """Select the most recent quote at or before prediction time."""
    eligible = [q for q in quotes if q["quote_time"] <= prediction_time]
    return max(eligible, key=lambda q: q["quote_time"]) if eligible else None


def audit_row(
    *,
    kickoff_time: datetime,
    prediction_time: datetime,
    latest_player_game_used: datetime | None,
    latest_team_game_used: datetime | None,
    odds_timestamp: datetime | None,
    odds_snapshot_time: datetime | None = None,
    weather_timestamp: datetime | None = None,
) -> datetime:
    """Return max availability timestamp or reject incomplete/leaky observations."""
    if kickoff_time.tzinfo is None or prediction_time.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware")
    if prediction_time >= kickoff_time:
        raise ValueError("Prediction must precede kickoff")
    if latest_player_game_used is None or latest_player_game_used >= prediction_time:
        raise ValueError("Missing or future player history")
    if latest_team_game_used is None or latest_team_game_used >= prediction_time:
        raise ValueError("Missing or future team history")
    if odds_timestamp is None or odds_timestamp > prediction_time:
        raise ValueError("Missing or future odds")
    if odds_snapshot_time is not None and odds_snapshot_time > prediction_time:
        raise ValueError("Future odds snapshot")
    if weather_timestamp is not None and weather_timestamp > prediction_time:
        raise ValueError("Future weather forecast")
    return max(
        timestamp
        for timestamp in (
            latest_player_game_used,
            latest_team_game_used,
            odds_timestamp,
            odds_snapshot_time,
            weather_timestamp,
        )
        if timestamp is not None
    )
