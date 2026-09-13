import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nfl_td_model.audit import audit_row, select_quote
from nfl_td_model.odds import extract_market_rows

KICKOFF = datetime(2024, 10, 20, 17, tzinfo=UTC)
PREDICTION = KICKOFF - timedelta(minutes=60)


def test_select_quote_never_uses_later_snapshot() -> None:
    quotes = [
        {"quote_time": PREDICTION - timedelta(minutes=5), "price": 140},
        {"quote_time": PREDICTION + timedelta(minutes=1), "price": 150},
    ]
    assert select_quote(quotes, PREDICTION)["price"] == 140


def test_audit_fails_on_current_game_player_usage() -> None:
    with pytest.raises(ValueError, match="player history"):
        audit_row(
            kickoff_time=KICKOFF,
            prediction_time=PREDICTION,
            latest_player_game_used=KICKOFF,
            latest_team_game_used=PREDICTION - timedelta(days=7),
            odds_timestamp=PREDICTION - timedelta(minutes=5),
        )


def test_audit_fails_on_future_odds() -> None:
    with pytest.raises(ValueError, match="odds"):
        audit_row(
            kickoff_time=KICKOFF,
            prediction_time=PREDICTION,
            latest_player_game_used=PREDICTION - timedelta(days=7),
            latest_team_game_used=PREDICTION - timedelta(days=7),
            odds_timestamp=PREDICTION + timedelta(seconds=1),
        )


def test_audit_requires_quote_and_history() -> None:
    with pytest.raises(ValueError, match="odds"):
        audit_row(
            kickoff_time=KICKOFF,
            prediction_time=PREDICTION,
            latest_player_game_used=PREDICTION - timedelta(days=7),
            latest_team_game_used=PREDICTION - timedelta(days=7),
            odds_timestamp=None,
        )


def test_weather_issuance_after_prediction_is_rejected() -> None:
    with pytest.raises(ValueError, match="weather"):
        audit_row(
            kickoff_time=KICKOFF,
            prediction_time=PREDICTION,
            latest_player_game_used=PREDICTION - timedelta(days=7),
            latest_team_game_used=PREDICTION - timedelta(days=7),
            odds_timestamp=PREDICTION - timedelta(minutes=5),
            weather_timestamp=PREDICTION + timedelta(seconds=1),
        )


def test_market_update_later_than_snapshot_is_discarded() -> None:
    payload = {
        "timestamp": "2024-10-20T15:55:00Z",
        "data": {
            "bookmakers": [
                {
                    "key": "testbook",
                    "markets": [
                        {
                            "key": "player_anytime_td",
                            "last_update": "2024-10-20T15:56:00Z",
                            "outcomes": [{"name": "Yes", "description": "A Player", "price": 140}],
                        },
                    ],
                }
            ]
        },
    }
    assert extract_market_rows(payload, datetime(2024, 10, 20, 15, 55, tzinfo=UTC)) == []
    assert len(extract_market_rows(payload, PREDICTION)) == 1


def test_generated_ten_game_audit_is_temporally_safe() -> None:
    path = Path(__file__).resolve().parents[1] / "reports" / "phase1_2024_week4_audit.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["game"] for row in rows}) == 10
    assert len(rows) == 40
    for row in rows:
        prediction = datetime.fromisoformat(row["prediction_time"])
        kickoff = datetime.fromisoformat(row["kickoff_time"])
        assert prediction == kickoff - timedelta(minutes=60)
        assert datetime.fromisoformat(row["latest_player_game_used"]) < prediction
        assert datetime.fromisoformat(row["latest_team_game_used"]) < prediction
        assert datetime.fromisoformat(row["player_history_available_at"]) < prediction
        assert datetime.fromisoformat(row["team_history_available_at"]) < prediction
        assert datetime.fromisoformat(row["feature_available_at_max"]) <= prediction
        assert row["result"] in {"0", "1"}
        assert row["result_source"] in {"pbp+weekly_stats", "pbp_only"}
        if row["odds_timestamp"]:
            assert datetime.fromisoformat(row["odds_timestamp"]) <= prediction
        else:
            assert row["status"].startswith("BLOCKED_")
