"""Historical Stage B settlement; 2024 is gated on a frozen 2023 rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.market_math import american_to_decimal
from nfl_td_model.phase2 import map_snap_counts
from nfl_td_model.phase7_stage_a import _flatten, _time
from nfl_td_model.xtd_data import classify_opportunity

REPORTS = Path("reports/phase7")
RAW = Path("data/raw/nflverse")
PBP_SETTLEMENT_COLUMNS = (
    "game_id", "rush_attempt", "pass_attempt", "rush_touchdown", "pass_touchdown",
    "td_player_id", "rusher_player_id", "receiver_player_id", "lateral_rusher_player_id",
    "lateral_receiver_player_id", "two_point_attempt", "play_deleted", "qb_kneel",
    "qb_spike", "play_type",
)


def verify_stage_a() -> dict[str, Any]:
    """Stage B cannot run against unfrozen or modified predictions/quotes."""
    root = REPORTS / "stage_a"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "phase7-stage-a-v1" or manifest["seasons"] != [2023, 2024]:
        raise ValueError("Phase 7 Stage A manifest is incomplete")
    for name, key in (("predictions.csv", "prediction_sha256"),
                      ("all_quotes.csv", "quotes_sha256")):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Phase 7 frozen Stage A artifact changed")
    return manifest


def verify_frozen_rule() -> dict[str, Any]:
    path = REPORTS / "development/frozen_rule.json"
    digest = (REPORTS / "development/frozen_rule.sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("2023 development rule hash changed")
    rule = json.loads(path.read_text(encoding="utf-8"))
    if rule.get("development_season") != 2023 or rule.get("validation_season") != 2024:
        raise ValueError("Frozen rule chronology is invalid")
    return rule


def _season_result_sources(season: int, games: set[str]) -> tuple[dict[tuple[str, str], int], set[tuple[str, str]], dict[str, int]]:
    """Read only season-specific PBP, weekly stats and positive PFR snap evidence."""
    if season not in (2023, 2024):
        raise ValueError("Phase 7 settlement season must be 2023 or 2024")
    source = json.loads(Path("reports/phase2_historical_source_hashes.json").read_text(
        encoding="utf-8"
    ))[str(season)]
    pbp = pl.read_parquet(RAW / f"pbp-{season}-{source['pbp']}.parquet",
                          columns=list(PBP_SETTLEMENT_COLUMNS))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    played: set[tuple[str, str]] = set()
    for play in pbp.iter_rows(named=True):
        game = play["game_id"]
        if game not in games:
            continue
        classified = classify_opportunity(play)
        if classified:
            _, player_id, touchdown = classified
            played.add((game, player_id))
            counts[(game, player_id)] += touchdown
    stats = pl.read_parquet(RAW / f"player-stats-{season}-{source['player_stats']}.parquet",
                            columns=["game_id", "player_id"])
    played.update((game, player) for game, player in stats.iter_rows() if game in games and player)
    snaps = pl.read_parquet(RAW / f"snap-counts-{season}-{source['snap_counts']}.parquet")
    crosswalk = pl.read_parquet(RAW / f"players-id-crosswalk-{source['player_id_crosswalk']}.parquet")
    mapped, _, snap_report = map_snap_counts(snaps, crosswalk, games)
    played.update(key for key, snap_count in mapped.items() if snap_count > 0)
    return counts, played, snap_report


def _closing_quotes(season: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    manifest_path = Path(f"reports/phase7_closing_{season}_manifest.json")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(records) != 256 or any(row["season"] != season for row in records):
        raise ValueError("Closing price manifest has incomplete season coverage")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        path = Path("data/raw/the_odds_api") / f"{record['payload_sha256']}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["payload_sha256"]:
            raise ValueError("Closing odds payload was modified")
        payload = json.loads(path.read_text(encoding="utf-8"))
        quotes = _flatten(payload, _time(record["kickoff"]))
        if len(quotes) != record["quote_count"]:
            raise ValueError("Closing quote count differs from collected snapshot")
        for quote in quotes:
            grouped[(record["game"], quote["normalized_player"])].append(quote)
    return grouped


def settle_season(season: int) -> Path:
    """Settle 2023 first; refuse 2024 until the 2023-only rule is frozen."""
    verify_stage_a()
    if season == 2024:
        verify_frozen_rule()
    elif season != 2023:
        raise ValueError("Only 2023/2024 Phase 7 settlement is supported")
    output = REPORTS / f"settlement_{season}.csv"
    if output.exists():
        raise FileExistsError("Phase 7 settlement is already frozen")
    predictions = pl.read_csv(REPORTS / "stage_a/predictions.csv").filter(
        pl.col("season") == season
    )
    games = set(predictions["game"].unique().to_list())
    counts, played, snap_report = _season_result_sources(season, games)
    closing = _closing_quotes(season)
    rows: list[dict[str, Any]] = []
    for row in predictions.to_dicts():
        key = (row["game"], row["player_id"])
        has_model = row["player_id"] is not None and row["p_football"] is not None
        affirmative_play = has_model and key in played
        status = "unmodeled" if not has_model else "graded_played" if affirmative_play else "policy_unknown"
        actual = counts.get(key, 0) if affirmative_play else None
        close = closing.get((row["game"], row["normalized_player"]), [])
        same_book = next((quote for quote in close if quote["sportsbook"] == row["best_sportsbook"]), None)
        same_book_decimal = None
        if same_book:
            same_book_decimal = american_to_decimal(same_book["price"])
        median_close = None
        if close:
            median_close = statistics.median(american_to_decimal(q["price"]) for q in close)
        graded = status == "graded_played"
        rows.append({
            **row, "ultimately_played": True if affirmative_play else None,
            "settlement_status": status, "actual_td_count": actual,
            "anytime_td": int(actual > 0) if actual is not None else None,
            "profit_best_units": (row["best_decimal"] - 1 if actual else -1) if graded else None,
            "profit_median_units": (row["median_decimal"] - 1 if actual else -1) if graded else None,
            "closing_same_book_decimal": same_book_decimal,
            "closing_median_decimal": median_close,
            "closing_same_book_time": same_book["quote_time"] if same_book else None,
            "clv_best_probability": (1 / same_book_decimal - 1 / row["best_decimal"])
            if same_book_decimal else None,
            "clv_median_probability": (1 / median_close - 1 / row["median_decimal"])
            if median_close else None,
        })
    REPORTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(output)
    manifest = {"season": season, "stage_a_sha256": verify_stage_a()["prediction_sha256"],
                "settlement_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "rows": len(rows), "graded_played": sum(r["settlement_status"] == "graded_played" for r in rows),
                "policy_unknown": sum(r["settlement_status"] == "policy_unknown" for r in rows),
                "unmodeled": sum(r["settlement_status"] == "unmodeled" for r in rows),
                "positive_snap_mapping": snap_report,
                "frozen_rule_sha256": hashlib.sha256(
                    (REPORTS / "development/frozen_rule.json").read_bytes()
                ).hexdigest() if season == 2024 else None}
    (REPORTS / f"settlement_{season}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 post-Stage-A season settlement")
    parser.add_argument("season", type=int, choices=[2023, 2024])
    arguments = parser.parse_args()
    print(settle_season(arguments.season))
