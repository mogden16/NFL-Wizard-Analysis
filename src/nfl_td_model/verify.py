"""Independent Phase 1 artifact checks; a green unit suite is not acceptance."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.market_math import american_to_decimal, implied_team_points
from nfl_td_model.odds import extract_market_rows, parse_time
from nfl_td_model.phase1 import normalize_name


def _timestamp(row: dict[str, str], column: str) -> datetime:
    raw = row.get(column)
    if not raw:
        raise ValueError(f"{row.get('game', '?')} {row.get('player', '?')}: missing {column}")
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError(f"{row.get('game', '?')} {row.get('player', '?')}: naive {column}")
    return parsed


def verify_audit_rows(rows: list[dict[str, str]], expected_games: int = 10) -> None:
    """Reject any missing or future input in the ten-game proof of concept."""
    games = {row["game"] for row in rows}
    if len(games) != expected_games or not rows:
        raise ValueError(f"Expected {expected_games} games; found {len(games)}")
    if len(rows) != expected_games * 4:
        raise ValueError(f"Expected four audited players per game; found {len(rows)} rows")
    counts = Counter((row["game"], row["team"]) for row in rows)
    if any(count != 2 for count in counts.values()) or len(counts) != expected_games * 2:
        raise ValueError("Expected two audited players per team-game")
    for row in rows:
        label = f"{row['game']} {row['player']}"
        kickoff = _timestamp(row, "kickoff_time")
        prediction = _timestamp(row, "prediction_time")
        lead = int(row["minutes_to_kickoff"])
        if lead <= 0 or prediction != kickoff - timedelta(minutes=lead):
            raise ValueError(f"{label}: invalid prediction lead time")
        player_game = _timestamp(row, "latest_player_game_used")
        team_game = _timestamp(row, "latest_team_game_used")
        player_available = _timestamp(row, "player_history_available_at")
        team_available = _timestamp(row, "team_history_available_at")
        quote = _timestamp(row, "odds_timestamp")
        snapshot = _timestamp(row, "odds_snapshot_time")
        atd_quote = _timestamp(row, "atd_quote_time")
        spread_quote = _timestamp(row, "spread_quote_time")
        total_quote = _timestamp(row, "total_quote_time")
        maximum = _timestamp(row, "feature_available_at_max")
        if player_game >= prediction or team_game >= prediction:
            raise ValueError(f"{label}: current or future game in history")
        if player_available < player_game or team_available < team_game:
            raise ValueError(f"{label}: history availability precedes its game")
        if max(player_available, team_available, quote, snapshot) > prediction:
            raise ValueError(f"{label}: feature or quote after prediction time")
        if quote != max(atd_quote, spread_quote, total_quote):
            raise ValueError(f"{label}: odds_timestamp does not match constituent quotes")
        # Market-level last_update can slightly exceed the wrapper snapshot time;
        # each timestamp must independently precede the prediction time.
        if maximum != max(player_available, team_available, quote, snapshot):
            raise ValueError(f"{label}: incorrect feature_available_at_max")
        if row["status"] != "PASS" or row["result"] not in {"0", "1"}:
            raise ValueError(f"{label}: incomplete result or audit status")
        if row.get("result_source") not in {"pbp+weekly_stats", "pbp_only"}:
            raise ValueError(f"{label}: missing outcome provenance")
        if row.get("selection_policy") != "top_prior_usage_with_ATD_quote":
            raise ValueError(f"{label}: audit sample is not quote eligible")
        for column in (
            "odds_event_id",
            "sportsbook",
            "market_sportsbook",
            "atd_american_odds",
            "home_spread",
            "game_total",
            "matched_odds_player_name",
            "name_match_method",
        ):
            if not row.get(column):
                raise ValueError(f"{label}: missing {column}")
        american_to_decimal(int(row["atd_american_odds"]))
        matched_name = row["matched_odds_player_name"]
        if normalize_name(matched_name) != normalize_name(row["player"]):
            raise ValueError(f"{label}: matched ATD name disagrees with player")
        expected_match = (
            "exact"
            if matched_name.casefold() == row["player"].casefold()
            else "terminal_suffix_alias"
        )
        if row["name_match_method"] != expected_match:
            raise ValueError(f"{label}: incorrect odds name-match method")
        expected_home, expected_away = implied_team_points(
            float(row["game_total"]), float(row["home_spread"])
        )
        # NFL game IDs are season_week_away_home. A team must be one of those two.
        _, _, away, home = row["game"].split("_", 3)
        if row["team"] not in {away, home}:
            raise ValueError(f"{label}: player team does not match game")
        expected_points = expected_home if row["team"] == home else expected_away
        if abs(float(row["implied_team_points"]) - expected_points) > 1e-9:
            raise ValueError(f"{label}: implied team points disagree with spread convention")


def verify_source_hashes(data_dir: Path, manifest_path: Path) -> None:
    """Ensure each frozen nflverse Parquet artifact still matches the recorded hash."""
    hashes: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = {
        "schedules": "schedules-2024",
        "player_stats": "player-stats-2024",
        "team_stats": "team-stats-2024",
        "pbp": "pbp-2024",
        "teams": "teams",
    }
    if set(hashes) != set(labels):
        raise ValueError("Source hash manifest has missing or unexpected entries")
    for key, label in labels.items():
        digest = hashes[key]
        path = data_dir / "raw" / "nflverse" / f"{label}-{digest}.parquet"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or altered raw source artifact: {key}")


def verify_phase1(audit_path: Path, manifest_path: Path, data_dir: Path) -> None:
    """Check source integrity and strict row-level acceptance evidence."""
    verify_source_hashes(data_dir, manifest_path)
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    verify_lineage(rows, audit_path.parent / "phase1_full_histories.json")
    verify_outcomes(rows, data_dir, manifest_path)
    verify_audit_rows(rows)
    verify_market_quotes(rows, data_dir, audit_path.parent / "phase1_odds_hashes.json")


def verify_market_quotes(
    rows: list[dict[str, str]], data_dir: Path, odds_manifest_path: Path
) -> None:
    """Reconcile every reported quote with an integrity-checked API response."""
    hashes: dict[str, str] = json.loads(odds_manifest_path.read_text(encoding="utf-8"))
    events = {row["odds_event_id"] for row in rows}
    if set(hashes) != events:
        raise ValueError("Odds source manifest does not cover audited events exactly")
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(row["odds_event_id"], []).append(row)
    for event_id, event_rows in by_event.items():
        digest = hashes[event_id]
        path = data_dir / "raw" / "the_odds_api" / f"{digest}.json"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or altered raw odds artifact: {event_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        event = payload.get("data") or {}
        if event.get("id") != event_id:
            raise ValueError(f"Odds event ID disagrees with raw response: {event_id}")
        snapshot = parse_time(payload["timestamp"])
        kickoff = parse_time(event["commence_time"])
        for row in event_rows:
            label = f"{row['game']} {row['player']}"
            prediction = _timestamp(row, "prediction_time")
            if snapshot != _timestamp(row, "odds_snapshot_time"):
                raise ValueError(f"{label}: odds snapshot disagrees with raw response")
            if kickoff != _timestamp(row, "kickoff_time"):
                raise ValueError(f"{label}: kickoff disagrees with raw event")
            market_rows = extract_market_rows(payload, prediction)
            player_quotes = [
                quote
                for quote in market_rows
                if quote["market"] == "player_anytime_td"
                and quote["name"] == "Yes"
                and normalize_name(str(quote["description"] or "")) == normalize_name(row["player"])
            ]
            if not player_quotes:
                raise ValueError(f"{label}: no matching raw ATD quote")
            matching_names = {
                str(quote["description"])
                for quote in player_quotes
                if quote["sportsbook"] == row["sportsbook"]
            }
            if matching_names != {row["matched_odds_player_name"]}:
                raise ValueError(f"{label}: ambiguous or incorrect raw ATD player name")
            matching_atd = [
                quote
                for quote in player_quotes
                if quote["sportsbook"] == row["sportsbook"]
                and quote["description"] == row["matched_odds_player_name"]
                and quote["price"] == int(row["atd_american_odds"])
                and quote["quote_time"] == _timestamp(row, "atd_quote_time")
            ]
            if (
                not matching_atd
                or max(q["quote_time"] for q in player_quotes) != matching_atd[0]["quote_time"]
            ):
                raise ValueError(f"{label}: ATD quote disagrees with latest raw quote")
            for market, name, point_column, time_column in (
                ("spreads", event["home_team"], "home_spread", "spread_quote_time"),
                ("totals", "Over", "game_total", "total_quote_time"),
            ):
                matching = [
                    quote
                    for quote in market_rows
                    if quote["sportsbook"] == row["market_sportsbook"]
                    and quote["market"] == market
                    and quote["name"] == name
                    and quote["point"] == float(row[point_column])
                    and quote["quote_time"] == _timestamp(row, time_column)
                ]
                if not matching:
                    raise ValueError(f"{label}: {market} disagrees with raw quote")


def verify_outcomes(rows: list[dict[str, str]], data_dir: Path, manifest_path: Path) -> None:
    """Reconcile settlement labels with frozen PBP and player-stat source files."""
    hashes: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = data_dir / "raw" / "nflverse"
    schedules = pl.read_parquet(source_dir / f"schedules-2024-{hashes['schedules']}.parquet")
    player_stats = pl.read_parquet(
        source_dir / f"player-stats-2024-{hashes['player_stats']}.parquet"
    )
    pbp = pl.read_parquet(source_dir / f"pbp-2024-{hashes['pbp']}.parquet")
    sampled_games = {row["game"] for row in rows}
    schedule_by_game = {
        row["game_id"]: row
        for row in schedules.filter(pl.col("game_id").is_in(sampled_games))
        .select("game_id", "home_score", "away_score")
        .to_dicts()
    }
    sample_pbp = pbp.filter(pl.col("game_id").is_in(sampled_games))
    for game_id in sampled_games:
        plays = sample_pbp.filter(pl.col("game_id") == game_id)
        schedule = schedule_by_game.get(game_id)
        if (
            schedule is None
            or plays.is_empty()
            or tuple(plays.select("home_score", "away_score").tail(1).row(0))
            != (schedule["home_score"], schedule["away_score"])
        ):
            raise ValueError(f"Incomplete PBP final score: {game_id}")
    scoring = sample_pbp.filter((pl.col("rush_touchdown") == 1) | (pl.col("pass_touchdown") == 1))
    if scoring.filter(pl.col("td_player_id").is_null()).height:
        raise ValueError("A rushing/receiving TD lacks scorer ID")
    scorer_pairs = {
        (item["game_id"], item["td_player_id"])
        for item in scoring.select("game_id", "td_player_id").to_dicts()
    }
    weekly_by_pair = {
        (item["game_id"], item["player_id"]): item
        for item in player_stats.filter(pl.col("game_id").is_in(sampled_games))
        .select("game_id", "player_id", "rushing_tds", "receiving_tds")
        .to_dicts()
    }
    for row in rows:
        pair = (row["game"], row["player_id"])
        expected = int(pair in scorer_pairs)
        if row["result"] != str(expected):
            raise ValueError(f"Outcome disagrees with PBP: {pair}")
        weekly = weekly_by_pair.get(pair)
        expected_source = "pbp+weekly_stats" if weekly is not None else "pbp_only"
        if row.get("result_source") != expected_source:
            raise ValueError(f"Outcome provenance disagrees with source files: {pair}")
        if (
            weekly is not None
            and int((weekly["rushing_tds"] or 0) + (weekly["receiving_tds"] or 0) > 0) != expected
        ):
            raise ValueError(f"Weekly stats and PBP outcome disagree: {pair}")


def verify_lineage(rows: list[dict[str, str]], lineage_path: Path) -> None:
    """Recompute usage features from every recorded prior player and team game."""
    lineage: list[dict[str, Any]] = json.loads(lineage_path.read_text(encoding="utf-8"))
    keyed = {(item["game"], item["player_id"]): item for item in lineage}
    if len(keyed) != len(lineage) or len(keyed) != len(rows):
        raise ValueError("Full history lineage does not cover each audit row exactly once")
    for row in rows:
        label = (row["game"], row["player_id"])
        history = keyed.get(label)
        if history is None:
            raise ValueError(f"Missing player history lineage: {label}")
        prediction = _timestamp(row, "prediction_time")
        player_games = history["source_games"]
        team_games = history["team_source_games"]
        if not player_games or not team_games:
            raise ValueError(f"Missing player or team source games: {label}")
        player_ids = {game["game_id"] for game in player_games}
        team_ids = {game["game_id"] for game in team_games}
        if player_ids != team_ids:
            raise ValueError(f"Player and team usage windows differ: {label}")
        for source in player_games + team_games:
            if source["game_id"] == row["game"]:
                raise ValueError(f"Current game used as feature: {label}")
            kickoff = datetime.fromisoformat(source["kickoff"])
            available = datetime.fromisoformat(source["available_at_proxy"])
            if kickoff >= prediction or available < kickoff or available >= prediction:
                raise ValueError(f"Future or unavailable source game: {label}")
        carries = sum(game["carries"] or 0 for game in player_games)
        targets = sum(game["targets"] or 0 for game in player_games)
        denominator = sum((game["carries"] or 0) + (game["targets"] or 0) for game in team_games)
        if int(row["historical_carries"]) != carries or int(row["historical_targets"]) != targets:
            raise ValueError(f"Player usage disagrees with source games: {label}")
        expected_share = (carries + targets) / denominator if denominator else None
        reported_share = (
            float(row["historical_touch_share"]) if row["historical_touch_share"] else None
        )
        if expected_share is None and reported_share is None:
            continue
        if (
            expected_share is None
            or reported_share is None
            or abs(expected_share - reported_share) > 1e-12
        ):
            raise ValueError(f"Touch share disagrees with source games: {label}")
