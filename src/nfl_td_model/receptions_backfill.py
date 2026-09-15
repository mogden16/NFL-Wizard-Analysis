"""Quota-safe planning and resumable orchestration for receptions backfill."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

BACKFILL_MARKET = "player_receptions"
CREDIT_PER_CALL = 10
TARGET_SEASONS = (2023, 2024)


@dataclass(frozen=True)
class BackfillPlan:
    seasons: tuple[int, ...]
    games: int
    cached_event_ids: int
    missing_event_ids: int
    cached_market_calls: int
    remaining_market_calls: int
    estimated_credits: int
    manifest_path: str


def _schedule_file(data_dir: Path, season: int) -> Path:
    files = sorted((data_dir / "raw" / "nflverse").glob(f"schedules-{season}-*.parquet"))
    files = [path for path in files if "2023-2024" not in path.name]
    if not files:
        raise FileNotFoundError(f"No schedule snapshot for {season}")
    return files[0]


def _games(data_dir: Path) -> list[dict[str, Any]]:
    eastern = ZoneInfo("America/New_York")
    games: list[dict[str, Any]] = []
    for season in TARGET_SEASONS:
        frame = pl.read_parquet(_schedule_file(data_dir, season)).filter(pl.col("game_type") == "REG")
        for row in frame.select("game_id", "season", "week", "gameday", "gametime").to_dicts():
            kickoff = datetime.fromisoformat(f"{row['gameday']}T{row['gametime']}").replace(tzinfo=eastern).astimezone(UTC)
            games.append({**row, "kickoff_time": kickoff.isoformat(),
                          "prediction_time": (kickoff - timedelta(minutes=60)).isoformat()})
    return games


def _cached_payloads(data_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in (data_dir / "raw" / "the_odds_api").glob("*.json"):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _cached_target_requests(data_dir: Path) -> int:
    count = 0
    for payload in _cached_payloads(data_dir):
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        markets = {market.get("key") for book in data.get("bookmakers", [])
                   for market in book.get("markets", [])}
        if BACKFILL_MARKET in markets:
            count += 1
    return count


def _cached_event_ids(data_dir: Path, games: list[dict[str, Any]]) -> set[str]:
    by_kickoff: dict[str, datetime] = {
        game["kickoff_time"]: datetime.fromisoformat(game["prediction_time"])
        for game in games
    }
    found: set[str] = set()
    for payload in _cached_payloads(data_dir):
        if not isinstance(payload.get("data"), list):
            continue
        try:
            snapshot = datetime.fromisoformat(str(payload["timestamp"]))
        except (KeyError, ValueError):
            continue
        for event in payload["data"]:
            kickoff = event.get("commence_time")
            event_id = event.get("id")
            cutoff = by_kickoff.get(kickoff)
            if event_id and cutoff is not None and snapshot <= cutoff:
                found.add(str(event_id))
    return found


def build_plan(data_dir: Path = Path("data"), reports_dir: Path = Path("reports")) -> BackfillPlan:
    """Build a no-network plan from target schedules and immutable local cache."""
    games = _games(data_dir)
    cached_ids = _cached_event_ids(data_dir, games)
    cached_calls = _cached_target_requests(data_dir)
    remaining = max(0, len(games) - cached_calls)
    manifest = reports_dir / "receptions_backfill_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"market": BACKFILL_MARKET, "games": games}, indent=2) + "\n", encoding="utf-8")
    plan = BackfillPlan(TARGET_SEASONS, len(games), len(cached_ids),
                        len(games) - len(cached_ids), cached_calls, remaining,
                        remaining * CREDIT_PER_CALL, str(manifest))
    (reports_dir / "RECEPTIONS_BACKFILL_PLAN.json").write_text(
        json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")
    return plan


def require_budget(plan: BackfillPlan, available_credits: int) -> None:
    """Fail closed before any request if the remaining quota cannot finish."""
    if available_credits < plan.estimated_credits:
        raise RuntimeError(
            f"Backfill requires {plan.estimated_credits} credits but only "
            f"{available_credits} are available; no requests made"
        )


def request_cache_key(event_id: str, prediction_time: str) -> str:
    """Stable key for one event/market/timestamp request, excluding credentials."""
    value = json.dumps([event_id, BACKFILL_MARKET, prediction_time], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()
