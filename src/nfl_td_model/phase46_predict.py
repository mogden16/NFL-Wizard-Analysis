"""Restricted, offline Stage A worker for prospective Phase 4.6 events.

This module has no network, historical-data, scoreboard, or settlement reader.
Its only input is a closed-schema pregame snapshot produced by the collector.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from nfl_td_model.market_math import (
    american_to_decimal,
    expected_value,
    poisson_at_least_one,
    raw_implied_probability,
)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Pregame timestamp has no timezone")
    return parsed


def normalize_name(name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", name.casefold())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)

SNAPSHOT_KEYS = {
    "schema", "event_id", "kickoff", "prediction_time", "captured_at", "home", "away",
    "team_expected_td", "players", "quotes", "availability", "availability_coverage", "model_version",
    "diagnostic_exhibition_slate", "quote_snapshot_sha256",
}
PLAYER_KEYS = {
    "player", "player_id", "position", "team", "allocation_score", "history_games",
    "latest_prior_available_at", "total_xtd_share", "rushing_xtd_share",
    "receiving_xtd_share", "goal_line_opportunities_share", "inside_5_carries_per_game",
    "inside_10_carries_per_game", "red_zone_targets_share", "end_zone_targets_share",
    "carry_share", "target_share", "snap_share",
}
QUOTE_KEYS = {"player", "sportsbook", "price", "quote_time", "quote_age_seconds", "market", "name"}
AVAILABILITY_KEYS = {
    "player", "team", "status", "source_url", "source_kind", "published_at", "retrieved_at",
}
COVERAGE_KEYS = {"team", "source_url", "source_kind", "published_at", "retrieved_at", "complete"}
INACTIVE = {"inactive", "out"}


def _closed(row: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = set(row) - allowed
    if extra:
        raise ValueError(f"Forbidden {label} fields: {sorted(extra)}")


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Reject unknown fields at every level, including leaked result columns."""
    _closed(snapshot, SNAPSHOT_KEYS, "snapshot")
    if snapshot.get("schema") != "phase46-pregame-v1":
        raise ValueError("Unsupported pregame schema")
    kickoff = parse_time(snapshot["kickoff"])
    cutoff = parse_time(snapshot["prediction_time"])
    captured = parse_time(snapshot["captured_at"])
    if cutoff != kickoff - timedelta(minutes=60) or not cutoff - timedelta(minutes=2) <= captured <= cutoff + timedelta(seconds=45):
        raise ValueError("Invalid prediction cutoff or capture time")
    if kickoff.astimezone(ZoneInfo("America/New_York")).date().isoformat() <= "2026-09-13" or snapshot["diagnostic_exhibition_slate"]:
        raise ValueError("September 13 exhibition cannot be regenerated")
    if set(snapshot["team_expected_td"]) != {snapshot["home"], snapshot["away"]}:
        raise ValueError("Team expectation mapping differs from event teams")
    if any(float(v) < 0 for v in snapshot["team_expected_td"].values()):
        raise ValueError("Negative expected team touchdowns")
    if not snapshot["quotes"]:
        raise ValueError("No pregame ATD quotes")
    for player in snapshot["players"]:
        _closed(player, PLAYER_KEYS, "player")
        if player["team"] not in snapshot["team_expected_td"]:
            raise ValueError("Historical player assigned outside this event")
        if parse_time(player["latest_prior_available_at"]) > cutoff:
            raise ValueError("Post-cutoff player history")
    for quote in snapshot["quotes"]:
        _closed(quote, QUOTE_KEYS, "quote")
        if quote["market"] != "player_anytime_td" or quote["name"] != "Yes":
            raise ValueError("Non-ATD quote entered prediction input")
        if parse_time(quote["quote_time"]) > cutoff:
            raise ValueError("Post-cutoff sportsbook quote")
        age = (cutoff - parse_time(quote["quote_time"])).total_seconds()
        if float(quote["quote_age_seconds"]) != age:
            raise ValueError("Quote age does not match T-60 timestamp")
        american_to_decimal(int(quote["price"]))
    for record in snapshot["availability"]:
        _closed(record, AVAILABILITY_KEYS, "availability")
        source = urlparse(record["source_url"])
        if record["source_kind"] != "official_nfl" or source.scheme != "https" or source.hostname not in {"www.nfl.com", "nfl.com"}:
            raise ValueError("Availability must have an official NFL source")
        if parse_time(record["published_at"]) > cutoff or parse_time(record["retrieved_at"]) > cutoff:
            raise ValueError("Post-cutoff availability record")
        if record["team"] not in snapshot["team_expected_td"]:
            raise ValueError("Availability team differs from event teams")
    for coverage in snapshot["availability_coverage"]:
        _closed(coverage, COVERAGE_KEYS, "availability coverage")
        source = urlparse(coverage["source_url"])
        if coverage["source_kind"] != "official_nfl" or source.scheme != "https" or source.hostname not in {"www.nfl.com", "nfl.com"}:
            raise ValueError("Availability coverage requires an official NFL source")
        if parse_time(coverage["published_at"]) > cutoff or parse_time(coverage["retrieved_at"]) > cutoff:
            raise ValueError("Post-cutoff availability coverage")
        if coverage["team"] not in snapshot["team_expected_td"]:
            raise ValueError("Availability coverage team differs from event teams")


def predict(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the unchanged fixed Phase 4.5 allocation and diagnostic rule."""
    validate_snapshot(snapshot)
    cutoff = parse_time(snapshot["prediction_time"])
    players: dict[tuple[str, str], list[dict[str, Any]]] = {}
    team_denominator: dict[str, float] = {}
    for player in snapshot["players"]:
        key = (normalize_name(player["player"]), player["team"])
        players.setdefault(key, []).append(player)
        team_denominator[player["team"]] = team_denominator.get(player["team"], 0) + float(
            player["allocation_score"]
        )
    unavailable: set[tuple[str, str]] = set()
    confirmed_active: set[tuple[str, str]] = set()
    verified_teams = {c["team"] for c in snapshot["availability_coverage"] if c["complete"] is True}
    for record in snapshot["availability"]:
        key = (normalize_name(record["player"]), record["team"])
        if record["status"].lower() in INACTIVE:
            unavailable.add(key)
        elif record["status"].lower() == "active":
            confirmed_active.add(key)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for quote in snapshot["quotes"]:
        by_name.setdefault(normalize_name(quote["player"]), []).append(quote)
    rows = []
    for normalized, quotes in sorted(by_name.items()):
        matches = [p for (name, _), group in players.items() if name == normalized for p in group]
        player = matches[0] if len(matches) == 1 else None
        quote_name = quotes[0]["player"]
        prices = [int(q["price"]) for q in quotes]
        best = max(prices, key=american_to_decimal)
        best_quotes = [q for q in quotes if int(q["price"]) == best]
        best_book = min(q["sportsbook"] for q in best_quotes)
        best_quote = max((q for q in best_quotes if q["sportsbook"] == best_book),
                         key=lambda q: q["quote_time"])
        median_decimal = statistics.median(american_to_decimal(p) for p in prices)
        median_american = statistics.median(prices)
        team: str | None = str(player["team"]) if player else None
        known_inactive = (normalized, team) in unavailable if team else False
        active_confirmed = (normalized, team) in confirmed_active if team else False
        # A complete official team status view and positive active-roster
        # confirmation are both required; a negative inactive signal overrides.
        availability_verified = team in verified_teams if team else False
        eligible = bool(player and availability_verified and active_confirmed and not known_inactive)
        share = None
        expected_team = float(snapshot["team_expected_td"][team]) if team else None
        expected_player = None
        probability = None
        if player and team is not None and team_denominator[team] > 0 and float(player["allocation_score"]) > 0:
            share = float(player["allocation_score"]) / team_denominator[team]
            assert expected_team is not None
            expected_player = expected_team * share
            probability = poisson_at_least_one(expected_player)
        implied_best = raw_implied_probability(best)
        implied_median = 1 / median_decimal
        edge_best = probability - implied_best if probability is not None else None
        edge_median = probability - implied_median if probability is not None else None
        ev_best = expected_value(probability, american_to_decimal(best)) if probability is not None else None
        ev_median = expected_value(probability, median_decimal) if probability is not None else None
        best_decimal = american_to_decimal(best)
        # Descriptive quote-quality flag, not an outcome-fitted exclusion rule.
        deviation = best_decimal / median_decimal - 1
        rows.append({
            "event_id": snapshot["event_id"], "player": quote_name,
            "player_id": player["player_id"] if player else None,
            "position": player["position"] if player else None, "team": team,
            "kickoff": snapshot["kickoff"], "prediction_time": snapshot["prediction_time"],
            "expected_team_td": expected_team, "expected_player_td": expected_player,
            "model_atd_probability": probability, "player_td_opportunity_share": share,
            "rolling_xtd_share": player["total_xtd_share"] if player else None,
            "best_sportsbook": best_quote["sportsbook"], "best_odds": best,
            "best_quote_time": best_quote["quote_time"],
            "best_quote_age_seconds": (cutoff - parse_time(best_quote["quote_time"])).total_seconds(),
            "median_odds": median_american, "median_decimal_odds": median_decimal,
            "books_quoting": len({q["sportsbook"] for q in quotes}),
            "best_vs_median_decimal_deviation": deviation,
            "extreme_best_price_flag": deviation >= 0.25,
            "raw_implied_best": implied_best, "raw_implied_median": implied_median,
            "edge_best": edge_best, "edge_median": edge_median,
            "ev_best": ev_best, "ev_median": ev_median,
            "eligible_at_prediction_time": eligible,
            "inactive_known_at_prediction_time": known_inactive,
            "availability_verified_at_prediction_time": availability_verified,
            "active_roster_confirmed_at_prediction_time": active_confirmed,
            "diagnostic_bet_best": bool(eligible and ev_best is not None and ev_best >= .05
                                        and edge_best is not None and edge_best >= .025),
            "diagnostic_bet_median": bool(eligible and ev_median is not None and ev_median >= .05
                                          and edge_median is not None and edge_median >= .025),
            "model_label": "PHASE 4.5 HIERARCHICAL BASELINE",
            "diagnostic_exhibition_slate": False,
        })
    return rows


def freeze_prediction(snapshot_path: Path, output_dir: Path) -> dict[str, Any]:
    """Read only a sanitized snapshot; atomically freeze quotes and predictions."""
    if any((output_dir / name).exists() for name in ("quotes.json", "predictions.json", "manifest.json")):
        raise FileExistsError("Prospective event prediction already frozen")
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    rows = predict(snapshot)
    cutoff = parse_time(snapshot["prediction_time"])
    created = datetime.now(UTC)
    if not cutoff - timedelta(minutes=2) <= created <= cutoff + timedelta(minutes=3):
        raise ValueError("Prospective Stage A must freeze at T-60, not as a retrospective replay")
    quote_digest = hashlib.sha256(json.dumps(snapshot["quotes"], sort_keys=True).encode()).hexdigest()
    if quote_digest != snapshot["quote_snapshot_sha256"]:
        raise ValueError("Pregame quote snapshot hash mismatch")
    if not rows:
        raise ValueError("No quoted players")
    output_dir.mkdir(parents=True, exist_ok=True)
    quotes_path = output_dir / "quotes.json"
    predictions_path = output_dir / "predictions.json"
    manifest_path = output_dir / "manifest.json"
    quotes_path.write_text(json.dumps(snapshot["quotes"], sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(rows, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase46-frozen-v1", "created_at_utc": created.isoformat(),
        "event_id": snapshot["event_id"], "kickoff": snapshot["kickoff"],
        "prediction_time": snapshot["prediction_time"],
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "quotes_sha256": hashlib.sha256(quotes_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "model_version": snapshot["model_version"],
        "diagnostic_exhibition_slate": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline Phase 4.6 Stage A prediction worker")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_prediction(args.snapshot, args.output_dir), sort_keys=True))
