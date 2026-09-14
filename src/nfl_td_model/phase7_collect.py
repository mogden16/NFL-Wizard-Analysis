"""Quote-only historical ATD collection; never reads current-game results."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.odds import HistoricalOddsClient, extract_market_rows, parse_time

LOGGER = logging.getLogger(__name__)
REPORTS = Path("reports")
RAW = Path("data/raw/the_odds_api")


def planned_games() -> pl.DataFrame:
    """Project only IDs and timestamps from frozen 2023/2024 pregame sources."""
    universe = pl.scan_parquet(
        "data/derived/phase5_2017_2024_player_features.parquet"
    ).select("season", "game").filter(pl.col("season").is_in([2023, 2024])).unique().collect()
    market = pl.scan_parquet(
        "data/derived/phase4_strict_market_2021_2024.parquet"
    ).select("season", "game_id", "event_id", "prediction_time", "kickoff").filter(
        pl.col("season").is_in([2023, 2024])
    ).unique("game_id").collect().rename({"game_id": "game"})
    joined = universe.join(market, on=["season", "game"], how="inner").sort("season", "kickoff", "game")
    if joined.height != 512 or joined.group_by("season").len().sort("season")["len"].to_list() != [256, 256]:
        raise ValueError("Expected 256 source-supported games in each 2023/2024 season")
    if joined.filter(pl.col("prediction_time") != pl.col("kickoff") - pl.duration(minutes=60)).height:
        raise ValueError("Historical quote cutoff is not kickoff minus 60 minutes")
    return joined


def _frozen_phase1() -> dict[str, str]:
    with (REPORTS / "phase1_2024_week4_audit.csv").open(newline="", encoding="utf-8") as handle:
        events = {row["odds_event_id"] for row in csv.DictReader(handle)}
    hashes = json.loads((REPORTS / "phase1_odds_hashes.json").read_text(encoding="utf-8"))
    return {event: hashes[event] for event in events}


def _payload_digest(payload: dict[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def collect_t60_atd_quotes(settings: Settings) -> Path:
    """Cache one-market T-60 odds, reusing immutable Phase 1 captures."""
    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required for historical ATD snapshots")
    games = planned_games()
    phase1 = _frozen_phase1()
    client = HistoricalOddsClient(settings.odds_api_key, settings.data_dir, settings.odds_regions)
    output = REPORTS / "phase7_t60_quote_manifest.json"
    partial = REPORTS / "phase7_t60_quote_manifest.partial.json"
    records: list[dict[str, Any]] = []
    try:
        for index, game in enumerate(games.iter_rows(named=True), 1):
            event = game["event_id"]
            cutoff = game["prediction_time"]
            if event in phase1:
                digest = phase1[event]
                path = RAW / f"{digest}.json"
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Frozen Phase 1 ATD snapshot was modified")
                payload = json.loads(path.read_text(encoding="utf-8"))
                source = "frozen_phase1"
            else:
                payload = client._get(
                    f"/events/{event}/odds", cutoff, regions=settings.odds_regions,
                    markets="player_anytime_td", oddsFormat="american",
                )
                digest = _payload_digest(payload)
                source = "historical_event_odds"
            if parse_time(payload["timestamp"]) > cutoff:
                raise ValueError("Future snapshot entered T-60 quote manifest")
            quotes = [row for row in extract_market_rows(payload, cutoff)
                      if row["market"] == "player_anytime_td" and row["name"] == "Yes"]
            records.append({"season": game["season"], "game": game["game"],
                            "event_id": event, "kickoff": game["kickoff"].isoformat(),
                            "prediction_time": cutoff.isoformat(),
                            "snapshot_time": payload["timestamp"], "payload_sha256": digest,
                            "quote_count": len(quotes),
                            "book_count": len({row["sportsbook"] for row in quotes}),
                            "source": source})
            if index % 20 == 0 or index == games.height:
                partial.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                LOGGER.info("Collected T-60 ATD snapshots %s/%s", index, games.height)
    finally:
        client.http.close()
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    return output


def collect_closing_atd_quotes(settings: Settings, season: int) -> Path:
    """Post-Stage-A closing capture for Stage B CLV only."""
    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required for historical closing ATD snapshots")
    if season not in (2023, 2024):
        raise ValueError("Phase 7 closing capture is restricted to 2023/2024")
    frozen = json.loads((REPORTS / "phase7/stage_a/manifest.json").read_text(encoding="utf-8"))
    prediction_file = REPORTS / "phase7/stage_a/predictions.csv"
    if hashlib.sha256(prediction_file.read_bytes()).hexdigest() != frozen["prediction_sha256"]:
        raise ValueError("Stage A predictions must be frozen before closing capture")
    if season == 2024:
        rule = REPORTS / "phase7/development/frozen_rule.json"
        rule_hash = REPORTS / "phase7/development/frozen_rule.sha256"
        if not rule.exists() or not rule_hash.exists() or hashlib.sha256(rule.read_bytes()).hexdigest() != rule_hash.read_text().strip():
            raise ValueError("2023 rule must be frozen before 2024 closing capture")
    games = planned_games().filter(pl.col("season") == season)
    client = HistoricalOddsClient(settings.odds_api_key, settings.data_dir, settings.odds_regions)
    output = REPORTS / f"phase7_closing_{season}_manifest.json"
    partial = REPORTS / f"phase7_closing_{season}_manifest.partial.json"
    records: list[dict[str, Any]] = []
    try:
        for index, game in enumerate(games.iter_rows(named=True), 1):
            cutoff = game["kickoff"]
            payload = client._get(
                f"/events/{game['event_id']}/odds", cutoff, regions=settings.odds_regions,
                markets="player_anytime_td", oddsFormat="american",
            )
            digest = _payload_digest(payload)
            quotes = [row for row in extract_market_rows(payload, cutoff)
                      if row["market"] == "player_anytime_td" and row["name"] == "Yes"]
            records.append({"season": season, "game": game["game"],
                            "event_id": game["event_id"], "kickoff": cutoff.isoformat(),
                            "snapshot_time": payload["timestamp"], "payload_sha256": digest,
                            "quote_count": len(quotes),
                            "book_count": len({row["sportsbook"] for row in quotes})})
            if index % 20 == 0 or index == games.height:
                partial.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                LOGGER.info("Collected %s closing ATD snapshots %s/%s", season, index, games.height)
    finally:
        client.http.close()
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    return output
