"""September 13, 2026 exhibition replay: pregame reconstruction only.

This module intentionally has no 2026 results/PBP/statistics reader. Settlement
is implemented in a separate module and requires the frozen digest first.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from nfl_td_model.config import Settings
from nfl_td_model.market_math import (
    american_to_decimal,
    expected_value,
    implied_team_points,
    poisson_at_least_one,
    raw_implied_probability,
)
from nfl_td_model.odds import HistoricalOddsClient, extract_market_rows, parse_time
from nfl_td_model.phase1 import _team_market, normalize_name
from nfl_td_model.phase4_models import MARKET, fit_count_model
from nfl_td_model.xtd_data import (
    PBP_COLUMNS,
    _asof_position,
    _position_history,
    classify_opportunity,
    play_context,
)
from nfl_td_model.xtd_features import aggregate_games
from nfl_td_model.xtd_models import fit_chronological

SLATE_DATE = "2026-09-13"
PREDICTIONS = Path("reports/phase45_stage_a_predictions.csv")
QUOTES = Path("reports/phase45_stage_a_all_quotes.csv")
MANIFEST = Path("reports/phase45_stage_a_manifest.json")
FAMILIES = {
    "total_xtd_share": 0.45,
    "rushing_xtd_share": 0.08,
    "receiving_xtd_share": 0.08,
    "goal_line_opportunities_share": 0.08,
    "inside_5_carries_per_game": 0.06,
    "inside_10_carries_per_game": 0.04,
    "red_zone_targets_share": 0.05,
    "end_zone_targets_share": 0.05,
    "carry_share": 0.05,
    "target_share": 0.05,
    "snap_share": 0.01,
}


def source_2025(settings: Settings, kind: str) -> pl.DataFrame:
    """Read only the frozen 2025 Phase 2 sources, all prior to this slate."""
    if kind not in {"schedules", "player_stats", "pbp"}:
        raise ValueError("Stage A historical source is not allowlisted")
    hashes = json.loads(Path("reports/phase2_historical_source_hashes.json").read_text())
    digest = hashes["2025"][kind]
    label = {"player_stats": "player-stats", "schedules": "schedules", "pbp": "pbp"}[kind]
    path = settings.data_dir / "raw/nflverse" / f"{label}-2025-{digest}.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen 2025 {kind} source hash mismatch")
    return pl.read_parquet(path, columns=list(PBP_COLUMNS) if kind == "pbp" else None)


def score_prior_xtd(settings: Settings) -> pl.DataFrame:
    """Score 2025 plays with frozen Phase 3 models, never fitting to 2025."""
    development = pl.read_parquet("data/derived/phase3_2017_2024_opportunities.parquet")
    schedules = source_2025(settings, "schedules")
    stats = source_2025(settings, "player_stats")
    games = {
        row["game_id"]: row
        for row in schedules.to_dicts()
        if row["season"] == 2025 and row["game_type"] == "REG"
    }
    positions = _position_history(stats, games)
    rows: list[dict[str, Any]] = []
    for play in source_2025(settings, "pbp").iter_rows(named=True):
        game = games.get(play["game_id"])
        if game is None:
            continue
        classified = classify_opportunity(play)
        if classified is None:
            continue
        kind, player_id, _ = classified
        kickoff = datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}").replace(
            tzinfo=ZoneInfo("America/New_York")
        ).astimezone(UTC)
        team = play["posteam"]
        if not team:
            continue
        rows.append(
            {
                "season": 2025,
                "game": play["game_id"],
                "kickoff_time": kickoff,
                "team": team,
                "player_id": player_id,
                "opportunity_type": kind,
                "actual_td": 0,  # Phase 4.5 scoring never needs prior TD outcomes.
                **play_context(play, kind, _asof_position(positions, player_id, team, kickoff)),
            }
        )
    frame = pl.DataFrame(rows)
    scored = []
    for kind in ("rushing", "receiving"):
        subset = frame.filter(pl.col("opportunity_type") == kind)
        model, _ = fit_chronological(development, kind, "logistic", 2024)
        if model.fit_max_season > 2022 or model.calibration_season != 2023:
            raise ValueError("Phase 3 model provenance changed")
        scored.append(subset.with_columns(pl.Series("xtd", model.predict(subset))))
    return aggregate_games(pl.concat(scored))


def _historical_players(settings: Settings) -> dict[str, dict[str, Any]]:
    stats = source_2025(settings, "player_stats")
    stats = stats.filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(["RB", "WR", "TE", "QB"])
    )
    xtd = score_prior_xtd(settings)
    xtd_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in xtd.to_dicts():
        xtd_map[(row["team"], row["player_id"])].append(row)
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stats.to_dicts():
        by_player[row["player_id"]].append(row)
    players: dict[str, dict[str, Any]] = {}
    for player_id, history in by_player.items():
        history.sort(key=lambda r: (r["week"], r["game_id"]))
        recent = history[-1]
        team = recent["team"]
        same_team = [r for r in history if r["team"] == team]
        if len(same_team) < 3:
            continue
        recent_xtd = sorted(xtd_map.get((team, player_id), []), key=lambda r: r["kickoff_time"])[-8:]
        if len(recent_xtd) < 3:
            continue
        # Per-family shares come from the same prior games, never 2026.
        values: dict[str, float | None] = {}
        for family in ("total", "rushing", "receiving"):
            numerator = sum(float(r[f"{family}_xtd"]) for r in recent_xtd)
            denominator = sum(float(r[f"team_{family}_xtd"]) for r in recent_xtd)
            values[f"{family}_xtd_share"] = numerator / denominator if denominator else None
        players[player_id] = {
            "player_id": player_id,
            "player": recent["player_display_name"] or recent["player_name"],
            "position": recent["position"],
            "team": team,
            "history_games": len(same_team),
            "latest_prior_game": recent["game_id"],
            "latest_prior_available_at": (recent_xtd[-1]["kickoff_time"] + timedelta(days=1)),
            **values,
        }
    # The final Phase 2 row for each player is a lagged summary prior to that
    # 2025 game. It cannot include the 2025 game itself, so use only xTD above
    # and separately calculate 2025 usage directly from prior completed games.
    pbp = source_2025(settings, "pbp")
    regular_games = {
        row["game_id"] for row in source_2025(settings, "schedules").to_dicts()
        if row["season"] == 2025 and row["game_type"] == "REG"
    }
    usage: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    team_usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for play in pbp.iter_rows(named=True):
        team = play.get("posteam")
        if not team or play["game_id"] not in regular_games:
            continue
        if play.get("play_deleted") == 1 or play.get("two_point_attempt") == 1:
            continue
        yardline = play.get("yardline_100")
        rush_id = play.get("rusher_player_id") if play.get("rush_attempt") == 1 else None
        target_id = play.get("receiver_player_id") if play.get("pass_attempt") == 1 else None
        if rush_id:
            team_usage[team]["carries"] += 1
            u = usage[(team, str(rush_id))]
            u["carries"] += 1
            if yardline is not None and yardline <= 10:
                u["inside_10_carries"] += 1
            if yardline is not None and yardline <= 5:
                u["inside_5_carries"] += 1
            if play.get("goal_to_go") == 1:
                u["goal_line_opportunities"] += 1
                team_usage[team]["goal_line_opportunities"] += 1
        if target_id:
            team_usage[team]["targets"] += 1
            u = usage[(team, str(target_id))]
            u["targets"] += 1
            if yardline is not None and yardline <= 20:
                u["red_zone_targets"] += 1
                team_usage[team]["red_zone_targets"] += 1
            air = play.get("air_yards")
            if air is not None and yardline is not None and air >= yardline:
                u["end_zone_targets"] += 1
                team_usage[team]["end_zone_targets"] += 1
    for player in players.values():
        team = player["team"]
        u = usage.get((team, player["player_id"]), {})
        t = team_usage[team]
        for key in ("goal_line_opportunities", "red_zone_targets", "end_zone_targets"):
            den = t.get(key, 0)
            player[key + "_share"] = u.get(key, 0) / den if den else None
        player["carry_share"] = u.get("carries", 0) / t["carries"] if t["carries"] else None
        player["target_share"] = u.get("targets", 0) / t["targets"] if t["targets"] else None
        player["inside_5_carries_per_game"] = u.get("inside_5_carries", 0) / player["history_games"]
        player["inside_10_carries_per_game"] = u.get("inside_10_carries", 0) / player["history_games"]
        player["snap_share"] = None  # Weekly snap counts lack reliable 2026 roster continuity.
    return players


def allocation_score(player: dict[str, Any], team_players: list[dict[str, Any]]) -> float:
    """Fixed, outcome-free weighted usage score; count features normalized by team."""
    total = 0.0
    weight_sum = 0.0
    for family, weight in FAMILIES.items():
        value = player.get(family)
        if value is None:
            continue
        if family.endswith("_per_game"):
            denominator = sum(float(p.get(family) or 0) for p in team_players)
            value = float(value) / denominator if denominator else 0.0
        total += weight * max(0.0, float(value))
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"Frozen artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temp = path.with_suffix(".temporary")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_stage_a(settings: Settings) -> dict[str, Any]:
    """Materialize immutable pregame predictions before any settlement access."""
    if any(p.exists() for p in (PREDICTIONS, QUOTES, MANIFEST)):
        raise FileExistsError("Stage A artifacts already exist; never overwrite them")
    if not settings.odds_api_key:
        raise ValueError("Historical The Odds API credential required")
    client = HistoricalOddsClient(settings.odds_api_key, settings.data_dir, settings.odds_regions)
    events = client.events(datetime(2026, 9, 13, 16, tzinfo=UTC))
    slate = [
        event for event in events["data"]
        if parse_time(event["commence_time"]).astimezone(ZoneInfo("America/New_York")).date().isoformat() == SLATE_DATE
    ]
    if len(slate) != 13:
        raise ValueError(f"Expected 13 September 13 games, found {len(slate)}")
    teams_file = next((settings.data_dir / "raw/nflverse").glob("teams-*.parquet"))
    names = {r["team_name"]: r["team_abbr"] for r in pl.read_parquet(teams_file).to_dicts()}
    players = _historical_players(settings)
    team_players: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for historical_player in players.values():
        team_players[historical_player["team"]].append(historical_player)
    for group in team_players.values():
        for historical_player in group:
            historical_player["allocation_score"] = allocation_score(historical_player, group)
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for historical_player in players.values():
        lookup[normalize_name(historical_player["player"])].append(historical_player)
    model_data = pl.read_parquet("data/derived/phase4_strict_market_2021_2024.parquet")
    model = fit_count_model(model_data.filter(pl.col("season") <= 2023), MARKET, "poisson")
    predictions: list[dict[str, Any]] = []
    quotes: list[dict[str, Any]] = []
    for event in slate:
        kickoff = parse_time(event["commence_time"])
        cutoff = kickoff - timedelta(minutes=60)
        raw = client.event_odds(event["id"], cutoff)
        market_rows = extract_market_rows(raw, cutoff)
        market = _team_market(market_rows, event["home_team"])
        if market is None:
            raise ValueError(f"No safe paired spread/total: {event['id']}")
        home = names[event["home_team"]]
        away = names[event["away_team"]]
        total = float(market["total"]["point"])
        spread = float(market["spread"]["point"])
        home_points, away_points = implied_team_points(total, spread)
        context = pl.DataFrame([
            {"implied_team_points": home_points, "team_spread": spread, "game_total": total, "home": 1},
            {"implied_team_points": away_points, "team_spread": -spread, "game_total": total, "home": 0},
        ])
        team_td = {home: float(model.predict(context)[0]), away: float(model.predict(context)[1])}
        yes_rows = [r for r in market_rows if r["market"] == "player_anytime_td" and r["name"] == "Yes"]
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in yes_rows:
            by_name[str(row["description"])].append(row)
            quotes.append({
                "event_id": event["id"], "game": f"{away}@{home}", "player": row["description"],
                "sportsbook": row["sportsbook"], "american_odds": row["price"],
                "quote_time": row["quote_time"].isoformat(), "snapshot_time": row["snapshot_time"].isoformat(),
                "prediction_time": cutoff.isoformat(),
            })
        for name, candidate_quotes in by_name.items():
            match = [p for p in lookup.get(normalize_name(name), []) if p["team"] in {home, away}]
            player: dict[str, Any] | None = match[0] if len(match) == 1 else None
            reason = None if player else "no_unambiguous_prior_same_team_history"
            if player and player["latest_prior_available_at"] >= cutoff:
                raise ValueError("Post-cutoff player history entered Stage A")
            prices = [int(q["price"]) for q in candidate_quotes]
            best = max(prices, key=american_to_decimal)
            best_book = min(q["sportsbook"] for q in candidate_quotes if int(q["price"]) == best)
            median_decimal = statistics.median(american_to_decimal(price) for price in prices)
            # Median American quote is descriptive; EV uses the exact median decimal payoff.
            median_american = statistics.median(prices)
            expected_team = team_td[player["team"]] if player else None
            share = None
            expected_player = None
            probability = None
            if player:
                group = team_players[player["team"]]
                denominator = sum(float(p["allocation_score"]) for p in group)
                if denominator > 0 and player["allocation_score"] > 0:
                    share = float(player["allocation_score"]) / denominator
                    assert expected_team is not None
                    expected_player = expected_team * share
                    probability = poisson_at_least_one(expected_player)
                else:
                    reason = "no_positive_prior_opportunity"
            implied_best = raw_implied_probability(best)
            implied_median = 1 / median_decimal
            edge_best = probability - implied_best if probability is not None else None
            edge_median = probability - implied_median if probability is not None else None
            ev_best = expected_value(probability, american_to_decimal(best)) if probability is not None else None
            ev_median = expected_value(probability, median_decimal) if probability is not None else None
            predictions.append({
                "event_id": event["id"], "game": f"{away}@{home}", "player": name,
                "player_id": player["player_id"] if player else None, "position": player["position"] if player else None,
                "team": player["team"] if player else None, "opponent": (away if player["team"] == home else home) if player else None,
                "kickoff": kickoff.isoformat(), "prediction_time": cutoff.isoformat(),
                "odds_snapshot_time": raw["timestamp"], "market_sportsbook": market["sportsbook"],
                "spread_quote_time": market["spread"]["quote_time"].isoformat(),
                "total_quote_time": market["total"]["quote_time"].isoformat(),
                "best_quote_time": max(q["quote_time"] for q in candidate_quotes if int(q["price"]) == best).isoformat(),
                "expected_team_td": expected_team, "allocation_score": player["allocation_score"] if player else None,
                "player_td_opportunity_share": share, "expected_player_td": expected_player,
                "model_atd_probability": probability, "rolling_xtd_share": player["total_xtd_share"] if player else None,
                "rushing_xtd_share": player["rushing_xtd_share"] if player else None,
                "receiving_xtd_share": player["receiving_xtd_share"] if player else None,
                "goal_line_share": player["goal_line_opportunities_share"] if player else None,
                "inside_5_carries_per_game": player["inside_5_carries_per_game"] if player else None,
                "inside_10_carries_per_game": player["inside_10_carries_per_game"] if player else None,
                "red_zone_target_share": player["red_zone_targets_share"] if player else None,
                "end_zone_target_share": player["end_zone_targets_share"] if player else None,
                "carry_share": player["carry_share"] if player else None,
                "target_share": player["target_share"] if player else None,
                "snap_share": player["snap_share"] if player else None,
                "prior_history_games": player["history_games"] if player else None,
                "latest_prior_game": player["latest_prior_game"] if player else None,
                "latest_prior_available_at": player["latest_prior_available_at"].isoformat() if player else None,
                "best_sportsbook": best_book, "best_odds": best, "median_odds": median_american,
                "books_quoting": len({q["sportsbook"] for q in candidate_quotes}),
                "raw_implied_best": implied_best, "raw_implied_median": implied_median,
                "edge_best": edge_best, "edge_median": edge_median,
                "ev_best": ev_best, "ev_median": ev_median,
                "diagnostic_bet_best": bool(ev_best is not None and ev_best >= 0.05 and edge_best is not None and edge_best >= 0.025),
                "diagnostic_bet_median": bool(ev_median is not None and ev_median >= 0.05 and edge_median is not None and edge_median >= 0.025),
                "ineligibility_reason": reason,
                "model_label": "PHASE 4.5 HIERARCHICAL BASELINE",
                "diagnostic_exhibition_slate": True,
            })
    if not predictions or not quotes:
        raise ValueError("No historical ATD quotes")
    # The manifest is written only after both artifacts are final, and itself
    # contains their cryptographic digests and the actual reconstruction time.
    quote_hash = _write_csv(QUOTES, quotes)
    prediction_hash = _write_csv(PREDICTIONS, predictions)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "slate_date": SLATE_DATE,
        "diagnostic_exhibition_slate": True,
        "prediction_file": str(PREDICTIONS), "prediction_sha256": prediction_hash,
        "quotes_file": str(QUOTES), "quotes_sha256": quote_hash,
        "model_version": "phase4-complete:A_market_only_poisson + phase3-complete:2024_logistic_inference_2025 + phase45_fixed_allocation_v1",
        "model_fit_max_season": model.fit_max_season,
        "allocation_weights": FAMILIES,
        "bet_rule": "EV >= 0.05 and model probability - raw implied >= 0.025; 1 unit flat",
        "games": len(slate), "quoted_players": len(predictions), "quotes": len(quotes),
        "eligible_players": sum(p["model_atd_probability"] is not None for p in predictions),
        "best_bets": sum(p["diagnostic_bet_best"] for p in predictions),
        "median_bets": sum(p["diagnostic_bet_median"] for p in predictions),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
