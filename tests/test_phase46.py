"""Prospective replay infrastructure and strict pregame/settlement boundaries."""

import ast
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from nfl_td_model import phase46_collect, phase46_predict, phase46_settle

KICKOFF = datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=60)
CUTOFF = KICKOFF - timedelta(minutes=60)


def snapshot() -> dict[str, Any]:
    quotes = [
        {"player": "A Runner", "sportsbook": "book_a", "price": 200,
         "quote_time": (CUTOFF - timedelta(minutes=3)).isoformat(),
         "quote_age_seconds": 180,
         "market": "player_anytime_td", "name": "Yes"},
        {"player": "A Runner", "sportsbook": "book_b", "price": 300,
         "quote_time": (CUTOFF - timedelta(minutes=10)).isoformat(),
         "quote_age_seconds": 600,
         "market": "player_anytime_td", "name": "Yes"},
    ]
    coverage = [
        {"team": team, "source_url": "https://www.nfl.com/injuries/", "source_kind": "official_nfl",
         "published_at": (CUTOFF - timedelta(minutes=20)).isoformat(),
         "retrieved_at": CUTOFF.isoformat(), "complete": True}
        for team in ("ATL", "CAR")
    ]
    return {
        "schema": "phase46-pregame-v1", "event_id": "event_future",
        "kickoff": KICKOFF.isoformat(), "prediction_time": CUTOFF.isoformat(),
        "captured_at": CUTOFF.isoformat(), "home": "ATL", "away": "CAR",
        "team_expected_td": {"ATL": 2.0, "CAR": 1.5},
        "players": [{
            "player": "A Runner", "player_id": "gsis-1", "position": "RB", "team": "ATL",
            "allocation_score": 1.0, "history_games": 8,
            "latest_prior_available_at": (CUTOFF - timedelta(days=6)).isoformat(),
            "total_xtd_share": .4, "rushing_xtd_share": .5, "receiving_xtd_share": .1,
            "goal_line_opportunities_share": .6, "inside_5_carries_per_game": .5,
            "inside_10_carries_per_game": .8, "red_zone_targets_share": .1,
            "end_zone_targets_share": .1, "carry_share": .4, "target_share": .1,
            "snap_share": None,
        }],
        "quotes": quotes, "availability": [{
            "player": "A Runner", "team": "ATL", "status": "active",
            "source_url": "https://www.nfl.com/news/official-game-roster", "source_kind": "official_nfl",
            "published_at": (CUTOFF - timedelta(minutes=20)).isoformat(),
            "retrieved_at": CUTOFF.isoformat(),
        }], "availability_coverage": coverage,
        "model_version": "frozen_phase4_phase3_phase45_allocation",
        "diagnostic_exhibition_slate": False,
        "quote_snapshot_sha256": hashlib.sha256(json.dumps(quotes, sort_keys=True).encode()).hexdigest(),
    }


@pytest.mark.parametrize("field", [
    "final_score", "current_game_pbp", "current_game_player_stats", "touchdown_results",
    "closing_odds", "settlement_status",
])
def test_stage_a_rejects_result_fields_at_every_level(field: str) -> None:
    original = snapshot()
    for path in ((), ("players", 0), ("quotes", 0), ("availability_coverage", 0)):
        item = deepcopy(original)
        target = item
        for key in path:
            target = target[key]  # type: ignore[index]
        target[field] = 99  # type: ignore[index]
        with pytest.raises(ValueError, match="Forbidden"):
            phase46_predict.validate_snapshot(item)


def test_stage_a_worker_has_no_result_or_network_imports() -> None:
    source = Path(phase46_predict.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert not imports.intersection({"httpx", "nflreadpy", "polars", "phase46_settle", "phase45_settle"})
    assert "site.api.espn.com" not in source and "load_pbp(" not in source


def test_known_inactive_excluded_but_quote_remains() -> None:
    case = snapshot()
    case["availability"].append({
        "player": "A Runner", "team": "ATL", "status": "inactive",
        "source_url": "https://www.nfl.com/injuries/", "source_kind": "official_nfl",
        "published_at": (CUTOFF - timedelta(minutes=10)).isoformat(),
        "retrieved_at": CUTOFF.isoformat(),
    })
    row = phase46_predict.predict(case)[0]
    assert row["inactive_known_at_prediction_time"] is True
    assert row["eligible_at_prediction_time"] is False
    assert row["diagnostic_bet_best"] is False
    assert row["model_atd_probability"] is not None


def test_missing_official_coverage_fails_closed_for_bets() -> None:
    case = snapshot()
    case["availability_coverage"] = []
    row = phase46_predict.predict(case)[0]
    assert row["availability_verified_at_prediction_time"] is False
    assert row["diagnostic_bet_best"] is False
    assert row["eligible_at_prediction_time"] is False


def test_missing_positive_active_status_fails_closed_for_bets() -> None:
    case = snapshot()
    case["availability"] = []
    row = phase46_predict.predict(case)[0]
    assert row["availability_verified_at_prediction_time"] is True
    assert row["active_roster_confirmed_at_prediction_time"] is False
    assert row["eligible_at_prediction_time"] is False
    assert row["diagnostic_bet_best"] is False


def test_quote_quality_and_quote_age_are_preserved() -> None:
    row = phase46_predict.predict(snapshot())[0]
    assert row["books_quoting"] == 2
    assert row["best_sportsbook"] == "book_b"
    assert row["best_odds"] == 300
    assert row["best_quote_age_seconds"] == 600
    assert row["median_odds"] == 250
    assert row["best_vs_median_decimal_deviation"] > 0
    assert row["extreme_best_price_flag"] is False


def test_extreme_best_price_is_flagged_without_filtering() -> None:
    case = snapshot()
    case["quotes"][1]["price"] = 2000
    row = phase46_predict.predict(case)[0]
    assert row["extreme_best_price_flag"] is True
    assert row["best_odds"] == 2000
    assert row["diagnostic_bet_best"] is True


def test_collector_drops_post_cutoff_live_market_updates() -> None:
    payload = {"bookmakers": [{"key": "book_a", "markets": [
        {"key": "player_anytime_td", "last_update": CUTOFF.isoformat(),
         "outcomes": [{"name": "Yes", "description": "A Runner", "price": 200}]},
        {"key": "player_anytime_td", "last_update": (CUTOFF + timedelta(seconds=1)).isoformat(),
         "outcomes": [{"name": "Yes", "description": "B Runner", "price": 300}]},
    ]}]}
    rows = phase46_collect._live_quote_rows(payload, CUTOFF)
    assert len(rows) == 1
    assert rows[0]["description"] == "A Runner"


def test_post_cutoff_quote_and_availability_rejected() -> None:
    case = snapshot()
    case["quotes"][0]["quote_time"] = (CUTOFF + timedelta(seconds=1)).isoformat()  # type: ignore[index]
    with pytest.raises(ValueError, match="Post-cutoff sportsbook"):
        phase46_predict.predict(case)
    case = snapshot()
    case["availability_coverage"][0]["retrieved_at"] = (CUTOFF + timedelta(seconds=1)).isoformat()  # type: ignore[index]
    with pytest.raises(ValueError, match="Post-cutoff availability"):
        phase46_predict.predict(case)


def test_later_dnp_preserves_prediction_and_uses_book_rule() -> None:
    original = phase46_predict.predict(snapshot())[0]
    assert original["diagnostic_bet_best"] is True
    assert original["eligible_at_prediction_time"] is True
    assert phase46_settle.settle_one(original, 0, False, "void_if_dnp")["settlement_status"] == "void_dnp"
    assert phase46_settle.settle_one(original, 0, False, "void_if_dnp")["pl_best_units"] == 0
    assert phase46_settle.settle_one(original, 0, False, "action_if_dnp")["pl_best_units"] == -1
    assert phase46_settle.settle_one(original, 0, False, None)["settlement_status"] == "pending_book_rule"
    assert phase46_settle.settle_one(original, 0, None, None)["settlement_status"] == "pending_participation"
    assert "settlement_status" not in original
    assert "ultimately_played" not in original


def test_stage_b_hash_barrier_precedes_result_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = snapshot()
    event_dir = tmp_path / "future"
    event_dir.mkdir()
    pregame = event_dir / "pregame.json"
    pregame.write_text(json.dumps(case), encoding="utf-8")
    phase46_predict.freeze_prediction(pregame, event_dir)

    def forbidden(_snapshot: object) -> None:
        raise AssertionError("Result endpoint accessed before hash verification")

    monkeypatch.setattr(phase46_settle, "_final_game", forbidden)
    (event_dir / "predictions.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        phase46_settle.settle_event(event_dir)


def test_offline_stage_a_runs_as_separate_process(tmp_path: Path) -> None:
    event_dir = tmp_path / "future"
    event_dir.mkdir()
    pregame = event_dir / "pregame.json"
    pregame.write_text(json.dumps(snapshot()), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "nfl_td_model.phase46_predict", str(pregame), str(event_dir)],
        check=True, capture_output=True, text=True,
    )
    manifest = phase46_settle.verify_frozen_event(event_dir)
    assert manifest["event_id"] == "event_future"
    assert manifest["quotes_sha256"] == hashlib.sha256((event_dir / "quotes.json").read_bytes()).hexdigest()
    frozen = json.loads((event_dir / "predictions.json").read_text(encoding="utf-8"))
    assert "settlement_status" not in frozen[0]
    assert "ultimately_played" not in frozen[0]
    assert "actual_td" not in frozen[0]


def test_stage_b_writes_outside_stage_a_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event_dir = tmp_path / "stage_a" / "future"
    event_dir.mkdir(parents=True)
    pregame = event_dir / "pregame.json"
    pregame.write_text(json.dumps(snapshot()), encoding="utf-8")
    phase46_predict.freeze_prediction(pregame, event_dir)
    game = {"event": {"competitions": [{"competitors": [
        {"homeAway": "home", "team": {"abbreviation": "ATL"}},
        {"homeAway": "away", "team": {"abbreviation": "CAR"}},
    ]}]}, "summary": {"boxscore": {"players": []}, "scoringPlays": []}}
    monkeypatch.setattr(phase46_settle, "_final_game", lambda _: game)
    output = phase46_settle.settle_event(event_dir, settlement_root=tmp_path / "stage_b")
    assert output.is_file()
    assert not (event_dir / "settlement.json").exists()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["rows"][0]["settlement_status"] == "pending_participation"
    assert result["rows"][0]["ultimately_played"] is None


def test_september_13_exhibition_cannot_be_replayed() -> None:
    case = snapshot()
    case["kickoff"] = datetime(2026, 9, 14, 0, 20, tzinfo=UTC).isoformat()
    case["prediction_time"] = datetime(2026, 9, 13, 23, 20, tzinfo=UTC).isoformat()
    case["captured_at"] = case["prediction_time"]
    with pytest.raises(ValueError, match="September 13"):
        phase46_predict.validate_snapshot(case)
