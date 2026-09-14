"""Exhibition replay firewall and immutable-stage regression tests."""

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Self

import pytest

from nfl_td_model import phase45, phase45_settle
from nfl_td_model.config import Settings


def test_stage_a_source_allowlist_and_no_settlement_import() -> None:
    tree = ast.parse(Path(phase45.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "nflreadpy" not in imported
    assert "phase45_settle" not in imported
    source = Path(phase45.__file__).read_text(encoding="utf-8")
    assert "load_pbp(2026)" not in source
    assert "load_player_stats(2026)" not in source
    assert "load_schedules(2026)" not in source
    with pytest.raises(ValueError, match="not allowlisted"):
        phase45.source_2025(Settings(), "settlement")


def test_frozen_prediction_hash_and_all_market_cutoffs() -> None:
    manifest = phase45_settle.verify_frozen()
    assert manifest["model_fit_max_season"] <= 2023
    assert manifest["diagnostic_exhibition_slate"] is True
    with phase45.PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest["quoted_players"]
    assert len({row["event_id"] for row in rows}) == 13
    for row in rows:
        cutoff = row["prediction_time"]
        assert row["best_quote_time"] <= cutoff
        assert row["spread_quote_time"] <= cutoff
        assert row["total_quote_time"] <= cutoff
        assert row["odds_snapshot_time"] <= cutoff
        if row["latest_prior_available_at"]:
            assert row["latest_prior_available_at"] < cutoff
        assert row["diagnostic_exhibition_slate"] == "True"


def test_settlement_rejects_tampered_predictions_before_outcome_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.csv"
    quotes = tmp_path / "quotes.csv"
    manifest_path = tmp_path / "manifest.json"
    predictions.write_text("frozen\n", encoding="utf-8")
    quotes.write_text("quotes\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({
            "slate_date": phase45.SLATE_DATE,
            "diagnostic_exhibition_slate": True,
            "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "quotes_sha256": hashlib.sha256(quotes.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(phase45_settle, "PREDICTIONS", predictions)
    monkeypatch.setattr(phase45_settle, "QUOTES", quotes)
    monkeypatch.setattr(phase45_settle, "MANIFEST", manifest_path)

    def forbidden_fetch() -> None:
        raise AssertionError("Outcome access happened before hash verification")

    monkeypatch.setattr(phase45_settle, "_fetch_final_games", forbidden_fetch)
    predictions.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        phase45_settle.settle_stage_b(Settings())


def test_allocation_is_fixed_and_nonnegative() -> None:
    assert abs(sum(phase45.FAMILIES.values()) - 1) < 1e-9
    players = [
        {"total_xtd_share": 0.4, "inside_5_carries_per_game": 2.0},
        {"total_xtd_share": 0.2, "inside_5_carries_per_game": 1.0},
    ]
    scores = [phase45.allocation_score(player, players) for player in players]
    assert scores[0] > scores[1] > 0


def test_median_portfolio_uses_median_decimal_price() -> None:
    rows = [{
        "ev_median": 0.1, "edge_median": 0.03, "median_odds": 201.5,
        "raw_implied_median": 0.33, "actual_td": 1,
        "model_atd_probability": 0.36,
    }]
    result = phase45_settle._portfolio(rows, "median", 0.05, False)
    assert result["bets"] == 1
    assert result["gross_winnings_units"] == pytest.approx(1 / 0.33 - 1)


def test_best_portfolio_uses_quoted_american_prices_and_flat_stakes() -> None:
    rows = [
        {"ev_best": .10, "edge_best": .03, "best_odds": 250, "actual_td": 1,
         "model_atd_probability": .35},
        {"ev_best": .06, "edge_best": .03, "best_odds": -120, "actual_td": 0,
         "model_atd_probability": .60},
        {"ev_best": .50, "edge_best": .02, "best_odds": 400, "actual_td": 1,
         "model_atd_probability": .22},
    ]
    result = phase45_settle._portfolio(rows, "best", .05, False)
    assert result["bets"] == 2
    assert result["wins"] == 1
    assert result["risked_units"] == 2
    assert result["gross_winnings_units"] == pytest.approx(2.5)
    assert result["net_units"] == pytest.approx(1.5)
    assert result["roi"] == pytest.approx(.75)


def test_settlement_counts_only_rushing_and_receiving_tds() -> None:
    game = {
        "event": {"competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "PHI"}},
            {"homeAway": "away", "team": {"abbreviation": "WSH"}},
        ]}]},
        "summary": {
            "boxscore": {"players": [{"team": {"abbreviation": "WSH"}, "statistics": [
                {"name": "rushing", "labels": ["CAR", "TD"], "athletes": [
                    {"athlete": {"displayName": "A. Runner"}, "stats": ["4", "1"]}]},
                {"name": "receiving", "labels": ["REC", "TD"], "athletes": [
                    {"athlete": {"displayName": "B. Catcher"}, "stats": ["2", "1"]}]},
                {"name": "passing", "labels": ["TD"], "athletes": [
                    {"athlete": {"displayName": "C. Passer"}, "stats": ["1"]}]},
            ]}]},
            "scoringPlays": [
                {"type": {"text": "Rushing Touchdown"}},
                {"type": {"text": "Passing Touchdown"}},
                {"type": {"text": "Interception Return Touchdown"}},
            ],
            "injuries": [{"injuries": [{
                "athlete": {"displayName": "D. Inactive"}, "status": "Out",
                "date": "2026-09-13T15:50:00Z",
                "details": {"fantasyStatus": {"description": "INACTIVE"}},
            }]}],
        },
    }
    scores, checks = phase45_settle._outcome_map([game])
    assert scores == {
        ("WAS@PHI", "WAS", "arunner"): 1,
        ("WAS@PHI", "WAS", "bcatcher"): 1,
    }
    assert checks["WAS@PHI"]["rushing_receiving_box_tds"] == 2
    assert phase45_settle._injury_map([game])[("WAS@PHI", "dinactive")]["timestamp"] == "2026-09-13T15:50:00Z"


def test_settlement_waits_for_all_final_games_before_summary_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"events": [
                {"status": {"type": {"name": "STATUS_FINAL"}}} for _ in range(12)
            ] + [{"status": {"type": {"name": "STATUS_IN_PROGRESS"}}}]}

    class Client:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def get(self, url: str, **_: object) -> Response:
            calls.append(url)
            return Response()

    monkeypatch.setattr(phase45_settle.httpx, "Client", Client)
    with pytest.raises(RuntimeError, match="All 13 games"):
        phase45_settle._fetch_final_games()
    assert len(calls) == 1
    assert "scoreboard" in calls[0]
