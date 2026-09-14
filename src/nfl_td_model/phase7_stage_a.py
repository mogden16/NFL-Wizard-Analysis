"""Offline Phase 7 T-60 worker. Its inputs have no results or closing prices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.market_math import american_to_decimal

FORBIDDEN = frozenset({"anytime_td", "actual_td_count", "final_score", "current_game_pbp",
                       "current_game_carries", "current_game_targets", "current_game_snaps",
                       "current_game_xtd", "closing_odds", "settlement_status", "p_market"})
PREGAME_FIELDS = frozenset({"season", "game", "team", "opponent", "player_id", "player",
                            "position", "prediction_time", "expected_team_td", "p_football",
                            "expected_player_td", "event_id", "kickoff"})


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Odds timestamp lacks timezone")
    return result


def _name(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def _flatten(payload: dict[str, Any], cutoff: datetime) -> list[dict[str, Any]]:
    """Reject post-cutoff snapshot and individual market updates."""
    if _time(payload["timestamp"]) > cutoff:
        raise ValueError("Later sportsbook snapshot entered Stage A")
    rows = []
    for book in (payload.get("data") or {}).get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "player_anytime_td":
                continue
            quote_time = _time(market["last_update"])
            if quote_time > cutoff:
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") != "Yes" or not outcome.get("description"):
                    continue
                price = int(outcome["price"])
                american_to_decimal(price)
                rows.append({"sportsbook": book["key"], "player": outcome["description"],
                             "normalized_player": _name(outcome["description"]),
                             "price": price, "quote_time": quote_time.isoformat(),
                             "quote_age_minutes": (cutoff - quote_time).total_seconds() / 60})
    return rows


def build_stage_a(
    pregame_path: Path = Path("data/phase7/pregame_player_view.parquet"),
    quote_manifest_path: Path = Path("reports/phase7_t60_quote_manifest.json"),
    output_dir: Path = Path("reports/phase7/stage_a"),
) -> Path:
    """Freeze every quoted player and all T-60 quotes; fail if output exists."""
    if output_dir.exists():
        raise FileExistsError("Phase 7 Stage A artifact is already frozen")
    source_manifest = json.loads(Path("reports/phase7_pregame_view_manifest.json").read_text())
    if hashlib.sha256(pregame_path.read_bytes()).hexdigest() != source_manifest["sha256"]:
        raise ValueError("Restricted pregame input hash changed")
    players = pl.read_parquet(pregame_path)
    if set(players.columns) != PREGAME_FIELDS or FORBIDDEN.intersection(players.columns):
        raise ValueError("Stage A pregame view has forbidden or unknown columns")
    if players.height != 12641 or sorted(players["season"].unique().to_list()) != [2023, 2024]:
        raise ValueError("Unexpected Stage A pregame universe")
    quote_manifest = json.loads(quote_manifest_path.read_text(encoding="utf-8"))
    if len(quote_manifest) != 512:
        raise ValueError("Historical T-60 quote collection is incomplete")
    players_by_game: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in players.to_dicts():
        players_by_game[row["game"]][_name(row["player"])].append(row)
    predictions: list[dict[str, Any]] = []
    all_quotes: list[dict[str, Any]] = []
    for record in quote_manifest:
        if record["season"] not in (2023, 2024):
            raise ValueError("2025/2026 entered Phase 7 quote manifest")
        cutoff = _time(record["prediction_time"])
        kickoff = _time(record["kickoff"])
        if (kickoff - cutoff).total_seconds() != 3600:
            raise ValueError("Historical prediction cutoff is not T-60")
        path = Path("data/raw/the_odds_api") / f"{record['payload_sha256']}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["payload_sha256"]:
            raise ValueError("T-60 quote payload was modified")
        payload = json.loads(path.read_text(encoding="utf-8"))
        quotes = _flatten(payload, cutoff)
        if len(quotes) != record["quote_count"]:
            raise ValueError("T-60 quote count changed after collection")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for quote in quotes:
            item = {"season": record["season"], "game": record["game"],
                    "event_id": record["event_id"], "prediction_time": cutoff.isoformat(),
                    **quote}
            all_quotes.append(item)
            grouped[quote["normalized_player"]].append(item)
        for normalized, options in sorted(grouped.items()):
            # A repeated person name in one game is ambiguous, not a silent match.
            matches = players_by_game[record["game"]].get(normalized, [])
            player = matches[0] if len(matches) == 1 else None
            by_book = {option["sportsbook"]: option for option in options}
            options = list(by_book.values())
            decimals = [american_to_decimal(option["price"]) for option in options]
            best = min(options, key=lambda option: (-american_to_decimal(option["price"]),
                                                    option["sportsbook"]))
            best_decimal = american_to_decimal(best["price"])
            median_decimal = statistics.median(decimals)
            probability = float(player["p_football"]) if player else None
            raw_best = 1 / best_decimal
            raw_median = 1 / median_decimal
            deviation = best_decimal / median_decimal - 1
            predictions.append({
                "season": record["season"], "game": record["game"],
                "event_id": record["event_id"], "player": best["player"],
                "normalized_player": normalized,
                "player_id": player["player_id"] if player else None,
                "position": player["position"] if player else None,
                "team": player["team"] if player else None,
                "opponent": player["opponent"] if player else None,
                "kickoff": kickoff.isoformat(), "prediction_time": cutoff.isoformat(),
                "model_version": "phase5_hierarchical_frozen",
                "expected_team_td": player["expected_team_td"] if player else None,
                "expected_player_td": player["expected_player_td"] if player else None,
                "p_football": probability, "best_sportsbook": best["sportsbook"],
                "best_american": best["price"], "best_decimal": best_decimal,
                "median_american_equivalent": statistics.median([o["price"] for o in options]),
                "median_decimal": median_decimal, "books": len(options),
                "best_quote_time": best["quote_time"],
                "best_quote_age_minutes": best["quote_age_minutes"],
                "median_quote_age_minutes": statistics.median(o["quote_age_minutes"] for o in options),
                "best_vs_median_decimal_deviation": deviation,
                "extreme_best_price_flag": deviation >= .25,
                "p_market_raw_best": raw_best, "p_market_raw_median": raw_median,
                "edge_best": probability - raw_best if probability is not None else None,
                "edge_median": probability - raw_median if probability is not None else None,
                "ev_best": probability * best_decimal - 1 if probability is not None else None,
                "ev_median": probability * median_decimal - 1 if probability is not None else None,
                "historical_role_proxy": player is not None,
                "eligible_at_prediction_time": False,
                "availability_status": "historical_active_status_unverified",
                "inactive_known_at_prediction_time": False,
            })
    output_dir.mkdir(parents=True)
    prediction_file = output_dir / "predictions.csv"
    quote_file = output_dir / "all_quotes.csv"
    pl.DataFrame(predictions, infer_schema_length=None).sort(
        "season", "game", "normalized_player"
    ).write_csv(prediction_file)
    pl.DataFrame(all_quotes, infer_schema_length=None).sort(
        "season", "game", "normalized_player", "sportsbook"
    ).write_csv(quote_file)
    manifest = {"schema": "phase7-stage-a-v1", "created_at_utc": datetime.now(UTC).isoformat(),
                "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                "quotes_sha256": hashlib.sha256(quote_file.read_bytes()).hexdigest(),
                "pregame_view_sha256": source_manifest["sha256"],
                "quote_manifest_sha256": hashlib.sha256(quote_manifest_path.read_bytes()).hexdigest(),
                "model_version": "phase5_hierarchical_frozen",
                "seasons": [2023, 2024], "quoted_players": len(predictions),
                "strict_live_eligibility_confirmed": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Phase 7 T-60 Stage A worker")
    parser.add_argument("--pregame", type=Path, default=Path("data/phase7/pregame_player_view.parquet"))
    parser.add_argument("--quotes", type=Path, default=Path("reports/phase7_t60_quote_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase7/stage_a"))
    args = parser.parse_args()
    print(build_stage_a(args.pregame, args.quotes, args.output))
