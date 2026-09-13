import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl_td_model.verify import (
    verify_audit_rows,
    verify_lineage,
    verify_market_quotes,
    verify_outcomes,
    verify_source_hashes,
)


def complete_row() -> dict[str, str]:
    kickoff = datetime(2024, 9, 29, 17, tzinfo=UTC)
    prediction = kickoff - timedelta(minutes=60)
    history_game = kickoff - timedelta(days=7)
    available = history_game + timedelta(days=1)
    quote = prediction - timedelta(minutes=5)
    return {
        "game": "2024_04_NO_ATL",
        "player": "Example Player",
        "team": "ATL",
        "selection_policy": "top_prior_usage_with_ATD_quote",
        "kickoff_time": kickoff.isoformat(),
        "prediction_time": prediction.isoformat(),
        "minutes_to_kickoff": "60",
        "latest_player_game_used": history_game.isoformat(),
        "latest_team_game_used": history_game.isoformat(),
        "player_history_available_at": available.isoformat(),
        "team_history_available_at": available.isoformat(),
        "odds_timestamp": quote.isoformat(),
        "odds_snapshot_time": prediction.isoformat(),
        "atd_quote_time": quote.isoformat(),
        "spread_quote_time": quote.isoformat(),
        "total_quote_time": quote.isoformat(),
        "odds_event_id": "example-event",
        "feature_available_at_max": prediction.isoformat(),
        "status": "PASS",
        "result": "0",
        "result_source": "pbp+weekly_stats",
        "sportsbook": "examplebook",
        "market_sportsbook": "examplebook",
        "atd_american_odds": "+140",
        "matched_odds_player_name": "Example Player",
        "name_match_method": "exact",
        "home_spread": "-3",
        "game_total": "47",
        "implied_team_points": "25",
    }


def complete_rows() -> list[dict[str, str]]:
    rows = []
    for team, player in (("ATL", "A"), ("ATL", "B"), ("NO", "C"), ("NO", "D")):
        row = complete_row()
        row["team"] = team
        row["player"] = player
        row["matched_odds_player_name"] = player
        row["implied_team_points"] = "25" if team == "ATL" else "22"
        rows.append(row)
    return rows


def test_complete_row_passes_gate() -> None:
    rows = complete_rows()
    verify_audit_rows(rows, expected_games=1)


def test_future_quote_rejected_even_with_pass_status() -> None:
    rows = complete_rows()
    row = rows[0]
    row["odds_timestamp"] = row["kickoff_time"]
    row["atd_quote_time"] = row["kickoff_time"]
    row["odds_snapshot_time"] = row["kickoff_time"]
    row["feature_available_at_max"] = row["kickoff_time"]
    with pytest.raises(ValueError, match="after prediction"):
        verify_audit_rows(rows, expected_games=1)


def test_wrong_implied_score_rejected() -> None:
    rows = complete_rows()
    row = rows[0]
    row["implied_team_points"] = "22"
    with pytest.raises(ValueError, match="implied team points"):
        verify_audit_rows(rows, expected_games=1)


def test_missing_quote_cannot_pass_acceptance_gate() -> None:
    rows = complete_rows()
    row = rows[0]
    row["odds_timestamp"] = ""
    with pytest.raises(ValueError, match="missing odds_timestamp"):
        verify_audit_rows(rows, expected_games=1)


def test_current_artifact_has_ten_distinct_games() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "reports" / "phase1_2024_week4_audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["game"] for row in rows}) == 10
    verify_lineage(rows, root / "reports" / "phase1_full_histories.json")


def test_source_hash_verifier_checks_immutable_files(tmp_path: Path) -> None:
    labels = {
        "schedules": "schedules-2024",
        "player_stats": "player-stats-2024",
        "team_stats": "team-stats-2024",
        "pbp": "pbp-2024",
        "teams": "teams",
    }
    source_dir = tmp_path / "raw" / "nflverse"
    source_dir.mkdir(parents=True)
    hashes = {}
    for key, label in labels.items():
        content = f"fixture-{key}".encode()
        digest = hashlib.sha256(content).hexdigest()
        hashes[key] = digest
        (source_dir / f"{label}-{digest}.parquet").write_bytes(content)
    manifest = tmp_path / "hashes.json"
    manifest.write_text(json.dumps(hashes), encoding="utf-8")
    verify_source_hashes(tmp_path, manifest)
    (source_dir / f"teams-{hashes['teams']}.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="altered raw source artifact: teams"):
        verify_source_hashes(tmp_path, manifest)


def test_outcome_verifier_recomputes_pbp_label(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "nflverse"
    source_dir.mkdir(parents=True)
    frames = {
        "schedules": (
            "schedules-2024",
            pl.DataFrame({"game_id": ["2024_04_NO_ATL"], "home_score": [7], "away_score": [0]}),
        ),
        "player_stats": (
            "player-stats-2024",
            pl.DataFrame(
                {
                    "game_id": ["2024_04_NO_ATL"],
                    "player_id": ["player-1"],
                    "rushing_tds": [1],
                    "receiving_tds": [0],
                }
            ),
        ),
        "pbp": (
            "pbp-2024",
            pl.DataFrame(
                {
                    "game_id": ["2024_04_NO_ATL"],
                    "home_score": [7],
                    "away_score": [0],
                    "rush_touchdown": [1],
                    "pass_touchdown": [0],
                    "td_player_id": ["player-1"],
                }
            ),
        ),
    }
    hashes = {}
    for key, (label, frame) in frames.items():
        temporary = source_dir / f"{label}.parquet"
        frame.write_parquet(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        hashes[key] = digest
        temporary.rename(source_dir / f"{label}-{digest}.parquet")
    manifest = tmp_path / "hashes.json"
    manifest.write_text(json.dumps(hashes), encoding="utf-8")
    row = {
        "game": "2024_04_NO_ATL",
        "player_id": "player-1",
        "result": "1",
        "result_source": "pbp+weekly_stats",
    }
    verify_outcomes([row], tmp_path, manifest)
    row["result"] = "0"
    with pytest.raises(ValueError, match="Outcome disagrees with PBP"):
        verify_outcomes([row], tmp_path, manifest)


def test_lineage_rejects_usage_changed_after_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "reports" / "phase1_2024_week4_audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["historical_carries"] = str(int(rows[0]["historical_carries"]) + 1)
    with pytest.raises(ValueError, match="Player usage disagrees"):
        verify_lineage(rows, root / "reports" / "phase1_full_histories.json")


def test_market_verifier_reconciles_and_rejects_tampering(tmp_path: Path) -> None:
    row = complete_row()
    payload = {
        "timestamp": row["odds_snapshot_time"],
        "data": {
            "id": row["odds_event_id"],
            "commence_time": row["kickoff_time"],
            "home_team": "Atlanta Falcons",
            "bookmakers": [
                {
                    "key": "examplebook",
                    "markets": [
                        {
                            "key": "player_anytime_td",
                            "last_update": row["atd_quote_time"],
                            "outcomes": [
                                {"name": "Yes", "description": "Example Player", "price": 140}
                            ],
                        },
                        {
                            "key": "spreads",
                            "last_update": row["spread_quote_time"],
                            "outcomes": [{"name": "Atlanta Falcons", "point": -3.0}],
                        },
                        {
                            "key": "totals",
                            "last_update": row["total_quote_time"],
                            "outcomes": [{"name": "Over", "point": 47.0}],
                        },
                    ],
                }
            ],
        },
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(content).hexdigest()
    source_dir = tmp_path / "raw" / "the_odds_api"
    source_dir.mkdir(parents=True)
    source = source_dir / f"{digest}.json"
    source.write_bytes(content)
    manifest = tmp_path / "odds_hashes.json"
    manifest.write_text(json.dumps({row["odds_event_id"]: digest}), encoding="utf-8")
    verify_market_quotes([row], tmp_path, manifest)
    row["atd_american_odds"] = "+150"
    with pytest.raises(ValueError, match="ATD quote disagrees"):
        verify_market_quotes([row], tmp_path, manifest)
    row["atd_american_odds"] = "+140"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="altered raw odds artifact"):
        verify_market_quotes([row], tmp_path, manifest)
