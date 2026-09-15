import pytest

from nfl_td_model.receptions_market import consensus, normalize_quotes, pair_quotes


def _payload():
    return {"timestamp": "2024-09-08T15:00:00+00:00", "data": {"bookmakers": [
        {"key": "book-a", "markets": [{"key": "player_receptions", "last_update": "2024-09-08T14:59:00+00:00", "outcomes": [
            {"name": "Over", "description": "A. Receiver", "point": 4.5, "price": -120},
            {"name": "Under", "description": "A. Receiver", "point": 4.5, "price": 100},
        ]}]},
        {"key": "book-b", "markets": [{"key": "player_receptions", "last_update": "2024-09-08T14:58:00+00:00", "outcomes": [
            {"name": "Over", "description": "A. Receiver", "point": 4.5, "price": -110},
            {"name": "Under", "description": "A. Receiver", "point": 4.5, "price": -110},
        ]}]},
    ]}}


def test_normalize_pair_and_consensus_same_line_only():
    rows = normalize_quotes(_payload(), {"game_id": "g", "event_id": "e", "season": 2024, "week": 1,
                                        "kickoff_time": "2024-09-08T16:00:00+00:00",
                                        "prediction_time": "2024-09-08T15:00:00+00:00"},
                           {"areceiver": [{"player_id": "p1", "team": "ATL"}]})
    paired = pair_quotes(rows)
    assert len(rows) == 4 and len(paired) == 2
    assert all(r["match_status"] == "MATCHED" for r in paired)
    summary = consensus(paired)[0]
    assert summary["books_quoting"] == 2
    assert summary["line"] == 4.5
    assert summary["median_no_vig_over_probability"] == pytest.approx(.5, abs=.02)


def test_future_market_update_is_excluded():
    payload = _payload()
    payload["data"]["bookmakers"][0]["markets"][0]["last_update"] = "2024-09-08T15:01:00+00:00"
    rows = normalize_quotes(payload, {"prediction_time": "2024-09-08T15:00:00+00:00"})
    assert len(rows) == 2
    assert {row["sportsbook"] for row in rows} == {"book-b"}


def test_ambiguous_and_unmatched_are_explicit():
    game = {"prediction_time": "2024-09-08T15:00:00+00:00"}
    players = {"areceiver": [{"player_id": "p1"}, {"player_id": "p2"}]}
    rows = normalize_quotes(_payload(), game, players)
    assert {r["match_status"] for r in rows} == {"AMBIGUOUS"}
