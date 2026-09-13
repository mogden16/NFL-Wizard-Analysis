"""Historical player-game feature generation from strictly lagged season sources."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import nflreadpy  # type: ignore[import-untyped]
import polars as pl

from nfl_td_model.audit import prediction_timestamp
from nfl_td_model.config import Settings
from nfl_td_model.feature_windows import WINDOWS, summarize_candidate
from nfl_td_model.phase1 import _snapshot_nfl_data, conservative_available_at, kickoff_utc

LOGGER = logging.getLogger(__name__)
RED_ZONE_COUNTS = (
    "red_zone_carries",
    "inside_10_carries",
    "inside_5_carries",
    "goal_line_opportunities",
    "red_zone_targets",
    "end_zone_targets",
)
PBP_COLUMNS = (
    "game_id",
    "posteam",
    "rush_attempt",
    "pass_attempt",
    "two_point_attempt",
    "qb_kneel",
    "rusher_player_id",
    "receiver_player_id",
    "yardline_100",
    "air_yards",
)


def count_red_zone_plays(
    pbp: pl.DataFrame, regular_games: set[str]
) -> tuple[
    dict[tuple[str, str], dict[str, int | None]],
    dict[tuple[str, str], dict[str, int | None]],
]:
    """Count ordinary rushing/target opportunities, excluding two-point tries."""
    player: dict[tuple[str, str], dict[str, int | None]] = defaultdict(
        lambda: dict.fromkeys(RED_ZONE_COUNTS, 0)
    )
    team: dict[tuple[str, str], dict[str, int | None]] = defaultdict(
        lambda: dict.fromkeys(RED_ZONE_COUNTS, 0)
    )

    def increment(field: str, player_key: tuple[str, str], team_key: tuple[str, str]) -> None:
        for bucket in (player[player_key], team[team_key]):
            current = bucket[field]
            if current is not None:
                bucket[field] = current + 1

    for play in pbp.select(PBP_COLUMNS).iter_rows(named=True):
        game_id = play["game_id"]
        offense = play["posteam"]
        yardline = play["yardline_100"]
        if (
            game_id not in regular_games
            or not offense
            or yardline is None
            or play["two_point_attempt"] == 1
        ):
            continue
        team_key = (game_id, offense)
        if play["rush_attempt"] == 1 and play["qb_kneel"] != 1:
            rusher = play["rusher_player_id"]
            if rusher:
                player_key = (game_id, rusher)
                for field, condition in (
                    ("red_zone_carries", yardline <= 20),
                    ("inside_10_carries", yardline <= 10),
                    ("inside_5_carries", yardline <= 5),
                    ("goal_line_opportunities", yardline <= 5),
                ):
                    if condition:
                        increment(field, player_key, team_key)
        if play["pass_attempt"] == 1:
            receiver = play["receiver_player_id"]
            if receiver:
                player_key = (game_id, receiver)
                if play["air_yards"] is None:
                    player[player_key]["end_zone_targets"] = None
                    team[team_key]["end_zone_targets"] = None
                for field, condition in (
                    ("red_zone_targets", yardline <= 20),
                    ("goal_line_opportunities", yardline <= 5),
                    (
                        "end_zone_targets",
                        play["air_yards"] is not None and play["air_yards"] >= yardline,
                    ),
                ):
                    if condition:
                        increment(field, player_key, team_key)
    return dict(player), dict(team)


def assert_complete_pbp(pbp: pl.DataFrame, games: list[dict[str, Any]]) -> None:
    """Prevent missing PBP from turning unavailable zone counts into false zeros."""
    last_scores = {
        row["game_id"]: row
        for row in pbp.group_by("game_id")
        .agg(
            pl.col("home_score").last().alias("pbp_home"),
            pl.col("away_score").last().alias("pbp_away"),
        )
        .to_dicts()
    }
    for game in games:
        score = last_scores.get(game["game_id"])
        if score is None or (score["pbp_home"], score["pbp_away"]) != (
            game["home_score"],
            game["away_score"],
        ):
            raise ValueError(f"Missing or incomplete PBP: {game['game_id']}")


def map_snap_counts(
    snap_counts: pl.DataFrame, players: pl.DataFrame, regular_games: set[str]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, int]]:
    """Join PFR game snaps to GSIS IDs and infer team snaps from PFR percentages."""
    id_rows = players.select("gsis_id", "pfr_id").drop_nulls().to_dicts()
    pfr_groups: dict[str, set[str]] = defaultdict(set)
    for row in id_rows:
        pfr_groups[row["pfr_id"]].add(row["gsis_id"])
    unique_pfr = {pfr: next(iter(ids)) for pfr, ids in pfr_groups.items() if len(ids) == 1}
    player_snaps: dict[tuple[str, str], float] = {}
    inferred: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts = {
        "offensive_pfr_rows": 0,
        "gsis_matched_rows": 0,
        "ambiguous_pfr_ids": len(pfr_groups) - len(unique_pfr),
    }
    for row in snap_counts.select(
        "game_id", "team", "pfr_player_id", "offense_snaps", "offense_pct"
    ).iter_rows(named=True):
        if row["game_id"] not in regular_games or row["offense_snaps"] is None:
            continue
        counts["offensive_pfr_rows"] += 1
        pct = row["offense_pct"]
        if pct is not None and pct > 0 and row["offense_snaps"] > 0:
            inferred[(row["game_id"], row["team"])].append(row["offense_snaps"] / pct)
        gsis = unique_pfr.get(row["pfr_player_id"])
        if gsis:
            counts["gsis_matched_rows"] += 1
            key = (row["game_id"], gsis)
            if key in player_snaps and player_snaps[key] != row["offense_snaps"]:
                raise ValueError(f"Conflicting PFR snap count identity: {key}")
            player_snaps[key] = row["offense_snaps"]
    team_snaps = {key: float(round(median(values))) for key, values in inferred.items()}
    return player_snaps, team_snaps, counts


def _read_season_sources(
    data_dir: Path, season: int
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Reuse frozen 2024 Phase 1 inputs; snapshot the same fields in other seasons."""
    labels = {
        "schedules": f"schedules-{season}",
        "player_stats": f"player-stats-{season}",
        "team_stats": f"team-stats-{season}",
        "pbp": f"pbp-{season}",
    }
    if season == 2024:
        hashes: dict[str, str] = json.loads(
            (Path("reports") / "phase1_source_hashes.json").read_text(encoding="utf-8")
        )
        frames = {
            key: pl.read_parquet(data_dir / "raw" / "nflverse" / f"{label}-{hashes[key]}.parquet")
            for key, label in labels.items()
        }
    else:
        frames = {
            "schedules": nflreadpy.load_schedules(season),
            "player_stats": nflreadpy.load_player_stats(season, summary_level="week"),
            "team_stats": nflreadpy.load_team_stats(season, summary_level="week"),
            "pbp": nflreadpy.load_pbp(season),
        }
        hashes = {
            key: _snapshot_nfl_data(data_dir, label, frame)
            for key, (label, frame) in ((key, (labels[key], frames[key])) for key in labels)
        }
    required = {
        "schedules": {"game_id", "season", "game_type", "week", "gameday", "gametime"},
        "player_stats": {
            "game_id",
            "player_id",
            "team",
            "position",
            "carries",
            "targets",
            "receptions",
        },
        "team_stats": {
            "game_id",
            "team",
            "opponent_team",
            "carries",
            "targets",
            "receptions",
            "attempts",
            "sacks_suffered",
            "rushing_tds",
            "passing_tds",
        },
        "pbp": set(PBP_COLUMNS) | {"home_score", "away_score"},
    }
    for key, columns in required.items():
        missing = columns - set(frames[key].columns)
        if missing:
            raise ValueError(f"Missing {season} {key} fields: {sorted(missing)}")
    return frames, hashes


def _build_one_season(
    settings: Settings, season: int, players: pl.DataFrame, crosswalk_hash: str
) -> tuple[Path, dict[str, str], dict[str, int], int]:
    """Create pregame-known player rows for one season without later-season inputs."""
    data_dir = settings.data_dir
    frames, source_hashes = _read_season_sources(data_dir, season)
    schedules = frames["schedules"].to_dicts()
    games = sorted(
        (game for game in schedules if game["season"] == season and game["game_type"] == "REG"),
        key=lambda game: (kickoff_utc(game), game["game_id"]),
    )
    game_by_id = {game["game_id"]: game for game in games}
    game_ids = set(game_by_id)
    if not game_ids:
        raise ValueError(f"No {season} regular-season games in nflverse schedule")
    assert_complete_pbp(frames["pbp"], games)
    snap_counts = nflreadpy.load_snap_counts(season)
    source_hashes = {
        **source_hashes,
        "snap_counts": _snapshot_nfl_data(data_dir, f"snap-counts-{season}", snap_counts),
        "player_id_crosswalk": crosswalk_hash,
    }
    if season == 2024:
        source_hashes["teams"] = json.loads(
            (Path("reports") / "phase1_source_hashes.json").read_text(encoding="utf-8")
        )["teams"]
    red_player, red_team = count_red_zone_plays(frames["pbp"], game_ids)
    player_snaps, team_snaps, snap_counts_report = map_snap_counts(snap_counts, players, game_ids)
    team_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    opponent_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_by_game: dict[tuple[str, str], dict[str, Any]] = {}
    for stats in frames["team_stats"].to_dicts():
        game = game_by_id.get(stats["game_id"])
        if game is None:
            continue
        game_id = stats["game_id"]
        team = stats["team"]
        opponent = stats["opponent_team"]
        kickoff = kickoff_utc(game)
        available = conservative_available_at(game)
        rz = red_team.get((game_id, team), dict.fromkeys(RED_ZONE_COUNTS, 0))
        carries = stats["carries"] or 0
        targets = stats["targets"] or 0
        receptions = stats["receptions"] or 0
        offensive_plays = carries + (stats["attempts"] or 0) + (stats["sacks_suffered"] or 0)
        touchdowns = (stats["rushing_tds"] or 0) + (stats["passing_tds"] or 0)
        record = {
            "game_id": game_id,
            "kickoff": kickoff,
            "available_at_proxy": available,
            "carries": carries,
            "targets": targets,
            "touches": carries + receptions,
            "snaps": team_snaps.get((game_id, team)),
            "offensive_plays": offensive_plays,
            "offensive_touchdowns": touchdowns,
            **rz,
        }
        team_records[team].append(record)
        team_by_game[(game_id, team)] = record
        opponent_records[opponent].append(
            {
                "game_id": game_id,
                "kickoff": kickoff,
                "available_at_proxy": available,
                "allowed_offensive_plays": offensive_plays,
                "allowed_carries": carries,
                "allowed_targets": targets,
                "allowed_red_zone_carries": rz["red_zone_carries"],
                "allowed_red_zone_targets": rz["red_zone_targets"],
                "allowed_offensive_touchdowns": touchdowns,
            }
        )
    player_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for stats in frames["player_stats"].to_dicts():
        game = game_by_id.get(stats["game_id"])
        if game is None or stats["position"] not in {"RB", "WR", "TE", "QB"}:
            continue
        game_id = stats["game_id"]
        team = stats["team"]
        team_game = team_by_game.get((game_id, team))
        if team_game is None:
            raise ValueError(f"Missing matching team-game stats: {game_id} {team}")
        player_id = stats["player_id"]
        rz = red_player.get((game_id, player_id), dict.fromkeys(RED_ZONE_COUNTS, 0))
        carries = stats["carries"] or 0
        targets = stats["targets"] or 0
        receptions = stats["receptions"] or 0
        player_records[(team, player_id)].append(
            {
                "game_id": game_id,
                "player_name": stats["player_display_name"],
                "position": stats["position"],
                "kickoff": kickoff_utc(game),
                "available_at_proxy": conservative_available_at(game),
                "carries": carries,
                "targets": targets,
                "receptions": receptions,
                "touches": carries + receptions,
                "snaps": player_snaps.get((game_id, player_id)),
                **rz,
                **{
                    f"team_{key}": team_game[key]
                    for key in (
                        "carries",
                        "targets",
                        "touches",
                        "snaps",
                        "goal_line_opportunities",
                        "red_zone_targets",
                        "end_zone_targets",
                    )
                },
            }
        )
    candidates_by_team: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (team, player_id), history in player_records.items():
        candidates_by_team[team].append((player_id, history))
    feature_rows: list[dict[str, Any]] = []
    for game in games:
        game_id = game["game_id"]
        kickoff = kickoff_utc(game)
        prediction = prediction_timestamp(kickoff, settings.prediction_lead_minutes)
        for team, opponent in (
            (game["home_team"], game["away_team"]),
            (game["away_team"], game["home_team"]),
        ):
            for player_id, history in candidates_by_team[team]:
                if not any(record["available_at_proxy"] < prediction for record in history):
                    continue
                features = summarize_candidate(
                    history,
                    team_records[team],
                    opponent_records[opponent],
                    prediction,
                    game_id=game_id,
                )
                if not features["player_history_games"]:
                    continue
                latest_player_record = max(
                    (
                        record
                        for record in history
                        if record["game_id"] != game_id
                        and record["available_at_proxy"] < prediction
                    ),
                    key=lambda record: (record["kickoff"], record["game_id"]),
                )
                available = features["feature_available_at_max"]
                if available is None or available >= prediction:
                    raise ValueError(f"Phase 2 source leakage: {game_id} {player_id}")
                feature_rows.append(
                    {
                        "game": game_id,
                        "player_id": player_id,
                        "player": latest_player_record["player_name"],
                        "position": latest_player_record["position"],
                        "team": team,
                        "opponent": opponent,
                        "kickoff_time": kickoff,
                        "prediction_time": prediction,
                        **features,
                    }
                )
    if not feature_rows:
        raise ValueError("No pregame-known player feature rows")
    feature_rows.sort(
        key=lambda row: (row["kickoff_time"], row["game"], row["team"], row["player_id"])
    )
    output = data_dir / "derived" / f"phase2_{season}_player_features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(feature_rows).write_parquet(output)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    if season == 2024:
        manifest = reports / "phase2_source_hashes.json"
        manifest.write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
        _write_phase2_reports(
            feature_rows,
            snap_counts_report,
            reports,
            player_records,
            team_records,
            opponent_records,
        )
    LOGGER.info(
        "Phase 2 %s features: %s rows across %s games", season, len(feature_rows), len(games)
    )
    return output, source_hashes, snap_counts_report, len(games)


FAMILY_FEATURES = {
    "carries": "last3_carries_per_game",
    "targets": "last3_targets_per_game",
    "touches": "last3_touches_per_game",
    "carry_share": "last3_carry_share",
    "target_share": "last3_target_share",
    "touch_share": "last3_touch_share",
    "snaps": "last3_snaps_per_game",
    "snap_share": "last3_snap_share",
    "red_zone_carries": "last3_red_zone_carries_per_game",
    "inside_10_carries": "last3_inside_10_carries_per_game",
    "inside_5_carries": "last3_inside_5_carries_per_game",
    "goal_line_opportunities": "last3_goal_line_opportunities_per_game",
    "goal_line_opportunity_share": "last3_goal_line_opportunity_share",
    "red_zone_targets": "last3_red_zone_targets_per_game",
    "red_zone_target_share": "last3_red_zone_target_share",
    "end_zone_targets": "last3_end_zone_targets_per_game",
    "end_zone_target_share": "last3_end_zone_target_share",
    "team_offensive_context": "last3_team_offensive_plays_per_game",
    "opponent_context": "last3_opponent_allowed_offensive_plays_per_game",
    "route_participation": "last3_route_participation",
}


def build_phase2_features(settings: Settings) -> tuple[Path, Path]:
    """Build and inventory 2017–2025 season files without using future-season games."""
    players = nflreadpy.load_players()
    crosswalk_hash = _snapshot_nfl_data(settings.data_dir, "players-id-crosswalk", players)
    source_manifest: dict[str, dict[str, str]] = {}
    season_paths: list[Path] = []
    matrix: list[dict[str, Any]] = []
    window_matrix: list[dict[str, Any]] = []
    for season in range(2017, 2026):
        path, hashes, snap_counts, schedule_games = _build_one_season(
            settings, season, players, crosswalk_hash
        )
        source_manifest[str(season)] = hashes
        season_paths.append(path)
        frame = pl.read_parquet(path)
        row: dict[str, Any] = {
            "season": season,
            "rows": frame.height,
            "schedule_games": schedule_games,
            "games_with_rows": frame["game"].n_unique(),
            "offensive_pfr_rows": snap_counts["offensive_pfr_rows"],
            "gsis_matched_rows": snap_counts["gsis_matched_rows"],
        }
        for family, column in FAMILY_FEATURES.items():
            present = frame[column].is_not_null().sum()
            row[f"{family}_present"] = present
            row[f"{family}_coverage_pct"] = round(100 * present / frame.height, 2)
            for window in WINDOWS:
                window_column = column.replace("last3_", f"{window}_", 1)
                observed = frame[window_column].is_not_null().sum()
                window_matrix.append(
                    {
                        "season": season,
                        "window": window,
                        "feature_family": family,
                        "present": observed,
                        "missing": frame.height - observed,
                        "coverage_pct": round(100 * observed / frame.height, 2),
                    }
                )
        matrix.append(row)
    combined = settings.data_dir / "derived" / "phase2_2017_2025_player_features.parquet"
    pl.scan_parquet([str(path) for path in season_paths]).sink_parquet(combined)
    reports = Path("reports")
    (reports / "phase2_artifact_hashes.json").write_text(
        json.dumps(
            {
                "combined": hashlib.sha256(combined.read_bytes()).hexdigest(),
                "by_season": {
                    str(season): hashlib.sha256(path.read_bytes()).hexdigest()
                    for season, path in zip(range(2017, 2026), season_paths, strict=True)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = reports / "phase2_historical_source_hashes.json"
    manifest.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    coverage = reports / "phase2_feature_availability_by_season.csv"
    with coverage.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    (reports / "phase2_feature_availability_by_season.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    with (reports / "phase2_feature_availability_by_season_window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_matrix[0]))
        writer.writeheader()
        writer.writerows(window_matrix)
    LOGGER.info(
        "Historical Phase 2: %s rows over %s seasons", sum(r["rows"] for r in matrix), len(matrix)
    )
    return combined, coverage


def _write_phase2_reports(
    rows: list[dict[str, Any]],
    snap_counts_report: dict[str, int],
    reports: Path,
    player_records: dict[tuple[str, str], list[dict[str, Any]]],
    team_records: dict[str, list[dict[str, Any]]],
    opponent_records: dict[str, list[dict[str, Any]]],
) -> None:
    feature_columns = [
        column
        for column in rows[0]
        if column.startswith(("last3_", "last5_", "last8_", "season_", "ewma_"))
    ]
    coverage = {
        "rows": len(rows),
        "games": len({row["game"] for row in rows}),
        "positions": {
            position: sum(row["position"] == position for row in rows)
            for position in ("RB", "WR", "TE", "QB")
        },
        "source_availability": "assumed_kickoff_plus_24h_not_observed_publication_time",
        "snap_id_mapping": snap_counts_report,
        "features": {
            column: {
                "present": sum(row[column] is not None for row in rows),
                "missing": sum(row[column] is None for row in rows),
            }
            for column in feature_columns
        },
    }
    (reports / "phase2_feature_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    examples: list[dict[str, Any]] = []
    for position in ("RB", "WR", "TE", "QB"):
        choices = [
            row
            for row in rows
            if row["position"] == position
            and row["game"].startswith("2024_04_")
            and row["player_history_games"] >= 2
        ]
        selected_games: set[str] = set()
        for row in choices:
            if row["game"] in selected_games:
                continue
            selected_games.add(row["game"])
            prediction = row["prediction_time"]
            player_ids = set(row["player_source_game_ids"].split(";"))
            team_ids = set(row["team_source_game_ids"].split(";"))
            opponent_ids = set(row["opponent_source_game_ids"].split(";"))
            summary = {
                key: value
                for key, value in row.items()
                if key
                in {
                    "game",
                    "player",
                    "player_id",
                    "position",
                    "team",
                    "opponent",
                    "prediction_time",
                    "feature_available_at_max",
                    "availability_type",
                    "last3_carries_per_game",
                    "last3_carry_share",
                    "last3_targets_per_game",
                    "last3_target_share",
                    "last3_red_zone_targets_per_game",
                    "last3_inside_5_carries_per_game",
                    "last3_snap_share",
                }
            }
            summary["player_source_games"] = sorted(
                (
                    record
                    for record in player_records[(row["team"], row["player_id"])]
                    if record["game_id"] in player_ids and record["available_at_proxy"] < prediction
                ),
                key=lambda record: record["kickoff"],
            )
            summary["team_source_games"] = sorted(
                (
                    record
                    for record in team_records[row["team"]]
                    if record["game_id"] in team_ids and record["available_at_proxy"] < prediction
                ),
                key=lambda record: record["kickoff"],
            )
            summary["opponent_source_games"] = sorted(
                (
                    record
                    for record in opponent_records[row["opponent"]]
                    if record["game_id"] in opponent_ids
                    and record["available_at_proxy"] < prediction
                ),
                key=lambda record: record["kickoff"],
            )
            examples.append(summary)
            if len(selected_games) == 3:
                break
    (reports / "phase2_representative_histories.json").write_text(
        json.dumps(examples, indent=2, default=lambda value: value.isoformat()), encoding="utf-8"
    )
    sample = [row for row in rows if row["game"].startswith("2024_04_")]
    columns = [
        "game",
        "player",
        "position",
        "team",
        "opponent",
        "prediction_time",
        "latest_player_game_used",
        "latest_team_game_used",
        "latest_opponent_game_used",
        "feature_available_at_max",
        "availability_type",
        "player_source_game_ids",
        "last3_carries_per_game",
        "last3_carry_share",
        "last3_targets_per_game",
        "last3_target_share",
        "last3_snap_share",
        "last3_red_zone_carries_per_game",
        "last3_red_zone_targets_per_game",
        "last3_end_zone_targets_per_game",
    ]
    with (reports / "phase2_week4_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in sample)
