"""Independent artifact and temporal checks for the Phase 2 feature table."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.feature_windows import WINDOWS
from nfl_td_model.phase1 import kickoff_utc
from nfl_td_model.phase2 import FAMILY_FEATURES

SOURCE_LABELS = {
    "schedules": "schedules-2024",
    "player_stats": "player-stats-2024",
    "team_stats": "team-stats-2024",
    "pbp": "pbp-2024",
    "teams": "teams",
    "snap_counts": "snap-counts-2024",
    "player_id_crosswalk": "players-id-crosswalk",
}


def verify_phase2_features(data_dir: Path, reports: Path) -> dict[str, int]:
    """Reject altered sources, current-game usage, and inconsistent coverage counts."""
    hashes: dict[str, str] = json.loads(
        (reports / "phase2_source_hashes.json").read_text(encoding="utf-8")
    )
    if set(hashes) != set(SOURCE_LABELS):
        raise ValueError("Phase 2 source manifest is incomplete")
    for key, label in SOURCE_LABELS.items():
        digest = hashes[key]
        path = data_dir / "raw" / "nflverse" / f"{label}-{digest}.parquet"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or altered Phase 2 source: {key}")
    phase1_hashes: dict[str, str] = json.loads(
        (reports / "phase1_source_hashes.json").read_text(encoding="utf-8")
    )
    if any(hashes[key] != digest for key, digest in phase1_hashes.items()):
        raise ValueError("Phase 2 changed the frozen Phase 1 source version")
    schedules = pl.read_parquet(
        data_dir / "raw" / "nflverse" / f"schedules-2024-{hashes['schedules']}.parquet"
    )
    player_stats = pl.read_parquet(
        data_dir / "raw" / "nflverse" / f"player-stats-2024-{hashes['player_stats']}.parquet"
    )
    team_stats = pl.read_parquet(
        data_dir / "raw" / "nflverse" / f"team-stats-2024-{hashes['team_stats']}.parquet"
    )
    schedule_by_id = {item["game_id"]: item for item in schedules.to_dicts()}
    player_source_keys = set(player_stats.select("game_id", "player_id", "team").iter_rows())
    team_source_keys = set(team_stats.select("game_id", "team").iter_rows())
    output = data_dir / "derived" / "phase2_2024_player_features.parquet"
    rows: list[dict[str, Any]] = pl.read_parquet(output).to_dicts()
    if not rows or len({(row["game"], row["team"], row["player_id"]) for row in rows}) != len(rows):
        raise ValueError("Phase 2 feature rows are missing or duplicated")
    for row in rows:
        label = f"{row['game']} {row['player_id']}"
        prediction = row["prediction_time"]
        kickoff = row["kickoff_time"]
        target_game = schedule_by_id.get(row["game"])
        if (
            target_game is None
            or kickoff != kickoff_utc(target_game)
            or {row["team"], row["opponent"]}
            != {target_game["home_team"], target_game["away_team"]}
        ):
            raise ValueError(f"Feature target game disagrees with frozen schedule: {label}")
        if prediction != kickoff - timedelta(minutes=60):
            raise ValueError(f"Invalid prediction timestamp: {label}")
        if row["availability_type"] != "assumed_kickoff_plus_24h":
            raise ValueError(f"Availability assumption is not explicit: {label}")
        if row["feature_available_at_max"] is None or row["feature_available_at_max"] >= prediction:
            raise ValueError(f"Feature available at or after prediction: {label}")
        expected_latest: dict[str, Any] = {}
        all_available = []
        for family, team in (
            ("player", row["team"]),
            ("team", row["team"]),
            ("opponent", row["opponent"]),
        ):
            raw_ids = row[f"{family}_source_game_ids"]
            ids = raw_ids.split(";") if raw_ids else []
            if len(ids) != row[f"{family}_history_games"] or len(set(ids)) != len(ids):
                raise ValueError(f"Inconsistent {family} source-game lineage: {label}")
            if family == "player" and not ids:
                raise ValueError(f"Player was not known before prediction: {label}")
            kickoffs = []
            for source_id in ids:
                game = schedule_by_id.get(source_id)
                if (
                    game is None
                    or source_id == row["game"]
                    or team not in (game["home_team"], game["away_team"])
                ):
                    raise ValueError(f"Invalid {family} source game: {label}")
                if (
                    family == "player"
                    and (source_id, row["player_id"], row["team"]) not in player_source_keys
                ):
                    raise ValueError(f"Player source game lacks same-team stats: {label}")
                if (source_id, team) not in team_source_keys:
                    raise ValueError(f"{family} source game lacks team stats: {label}")
                source_kickoff = kickoff_utc(game)
                assumed_available = source_kickoff + timedelta(days=1)
                if assumed_available >= prediction:
                    raise ValueError(f"Current or unavailable {family} game used: {label}")
                kickoffs.append(source_kickoff)
                all_available.append(assumed_available)
            expected_latest[family] = max(kickoffs, default=None)
        if not all_available or max(all_available) != row["feature_available_at_max"]:
            raise ValueError(f"Incorrect maximum feature availability: {label}")
        for family, expected in expected_latest.items():
            if row[f"latest_{family}_game_used"] != expected:
                raise ValueError(f"Incorrect latest {family} game: {label}")
        for key, value in row.items():
            if key.endswith("_share") and value is not None and not 0 <= value <= 1.000001:
                raise ValueError(f"Invalid share {key}: {label}")
            if key.endswith(("_routes_per_game", "_route_participation")) and value is not None:
                raise ValueError(f"Unverified route feature populated: {label}")
    coverage: dict[str, Any] = json.loads(
        (reports / "phase2_feature_coverage.json").read_text(encoding="utf-8")
    )
    if coverage["rows"] != len(rows) or coverage["games"] != len({row["game"] for row in rows}):
        raise ValueError("Phase 2 coverage totals disagree with feature table")
    for column, counts in coverage["features"].items():
        present = sum(row[column] is not None for row in rows)
        if counts != {"present": present, "missing": len(rows) - present}:
            raise ValueError(f"Phase 2 coverage disagrees for {column}")
    return {"rows": len(rows), "games": coverage["games"]}


def verify_historical_phase2(data_dir: Path, reports: Path) -> dict[str, int]:
    """Audit every 2017–2025 source and row, plus the combined coverage matrix."""
    artifact_hashes: dict[str, Any] = json.loads(
        (reports / "phase2_artifact_hashes.json").read_text(encoding="utf-8")
    )
    manifests: dict[str, dict[str, str]] = json.loads(
        (reports / "phase2_historical_source_hashes.json").read_text(encoding="utf-8")
    )
    with (reports / "phase2_feature_availability_by_season.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        matrix = {int(row["season"]): row for row in csv.DictReader(handle)}
    with (reports / "phase2_feature_availability_by_season_window.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        window_rows = list(csv.DictReader(handle))
    window_matrix = {
        (int(row["season"]), row["window"], row["feature_family"]): row for row in window_rows
    }
    seasons = set(range(2017, 2026))
    if set(artifact_hashes.get("by_season", {})) != {str(season) for season in seasons}:
        raise ValueError("Historical Phase 2 artifact manifest is incomplete")
    if set(map(int, manifests)) != seasons or set(matrix) != seasons:
        raise ValueError("Historical Phase 2 coverage omits a required season")
    if len(window_matrix) != len(window_rows) or len(window_matrix) != len(seasons) * len(
        WINDOWS
    ) * len(FAMILY_FEATURES):
        raise ValueError("Historical Phase 2 window-coverage matrix is incomplete")
    all_keys: set[tuple[str, str, str]] = set()
    total_rows = 0
    for season in sorted(seasons):
        hashes = manifests[str(season)]
        expected = {
            "schedules",
            "player_stats",
            "team_stats",
            "pbp",
            "snap_counts",
            "player_id_crosswalk",
        }
        if season == 2024:
            expected.add("teams")
        if set(hashes) != expected:
            raise ValueError(f"Incomplete {season} source manifest")
        for key, digest in hashes.items():
            label = (
                "players-id-crosswalk"
                if key == "player_id_crosswalk"
                else "teams"
                if key == "teams"
                else f"{SOURCE_LABELS[key].rsplit('-', 1)[0]}-{season}"
            )
            path = data_dir / "raw" / "nflverse" / f"{label}-{digest}.parquet"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Missing or changed {season} source: {key}")
        source_dir = data_dir / "raw" / "nflverse"
        feature_path = data_dir / "derived" / f"phase2_{season}_player_features.parquet"
        if (
            hashlib.sha256(feature_path.read_bytes()).hexdigest()
            != artifact_hashes["by_season"][str(season)]
        ):
            raise ValueError(f"Changed {season} feature artifact")
        schedules = pl.read_parquet(
            source_dir / f"schedules-{season}-{hashes['schedules']}.parquet"
        )
        player_stats = pl.read_parquet(
            source_dir / f"player-stats-{season}-{hashes['player_stats']}.parquet"
        )
        team_stats = pl.read_parquet(
            source_dir / f"team-stats-{season}-{hashes['team_stats']}.parquet"
        )
        schedule_by_id = {record["game_id"]: record for record in schedules.to_dicts()}
        player_keys = set(player_stats.select("game_id", "player_id", "team").iter_rows())
        team_keys = set(team_stats.select("game_id", "team").iter_rows())
        table = pl.read_parquet(feature_path)
        summary = matrix[season]
        if table.height != int(summary["rows"]):
            raise ValueError(f"Incorrect {season} row count in coverage matrix")
        game_count = schedules.filter(
            (pl.col("season") == season) & (pl.col("game_type") == "REG")
        ).height
        if game_count != int(summary["schedule_games"]):
            raise ValueError(f"Incorrect {season} schedule-game count")
        if table["game"].n_unique() != int(summary["games_with_rows"]):
            raise ValueError(f"Incorrect {season} feature-game count")
        for family, column in FAMILY_FEATURES.items():
            present = table[column].is_not_null().sum()
            if present != int(summary[f"{family}_present"]) or round(
                100 * present / table.height, 2
            ) != float(summary[f"{family}_coverage_pct"]):
                raise ValueError(f"Incorrect {season} {family} coverage")
            for window in WINDOWS:
                window_column = column.replace("last3_", f"{window}_", 1)
                observed = table[window_column].is_not_null().sum()
                entry = window_matrix[(season, window, family)]
                if (
                    observed != int(entry["present"])
                    or table.height - observed != int(entry["missing"])
                    or round(100 * observed / table.height, 2) != float(entry["coverage_pct"])
                ):
                    raise ValueError(f"Incorrect {season} {window} {family} coverage")
        metadata = table.select(
            "game",
            "team",
            "opponent",
            "player_id",
            "kickoff_time",
            "prediction_time",
            "availability_type",
            "feature_available_at_max",
            "player_source_game_ids",
            "team_source_game_ids",
            "opponent_source_game_ids",
            "player_history_games",
            "team_history_games",
            "opponent_history_games",
            "latest_player_game_used",
            "latest_team_game_used",
            "latest_opponent_game_used",
        )
        for row in metadata.iter_rows(named=True):
            row_key = (row["game"], row["team"], row["player_id"])
            if row_key in all_keys:
                raise ValueError(f"Duplicate historical feature row: {row_key}")
            all_keys.add(row_key)
            target = schedule_by_id.get(row["game"])
            prediction = row["prediction_time"]
            if (
                target is None
                or target["season"] != season
                or target["game_type"] != "REG"
                or row["kickoff_time"] != kickoff_utc(target)
                or prediction != row["kickoff_time"] - timedelta(minutes=60)
                or {row["team"], row["opponent"]} != {target["home_team"], target["away_team"]}
                or row["availability_type"] != "assumed_kickoff_plus_24h"
            ):
                raise ValueError(f"Invalid target-game metadata: {row_key}")
            availability = []
            for family, team in (
                ("player", row["team"]),
                ("team", row["team"]),
                ("opponent", row["opponent"]),
            ):
                source_ids = (
                    row[f"{family}_source_game_ids"].split(";")
                    if row[f"{family}_source_game_ids"]
                    else []
                )
                if len(source_ids) != row[f"{family}_history_games"] or len(source_ids) != len(
                    set(source_ids)
                ):
                    raise ValueError(f"Invalid {family} lineage count: {row_key}")
                if family == "player" and not source_ids:
                    raise ValueError(f"Player lacks pregame history: {row_key}")
                source_kickoffs = []
                for source_id in source_ids:
                    source = schedule_by_id.get(source_id)
                    if (
                        source is None
                        or source_id == row["game"]
                        or source["season"] != season
                        or source["game_type"] != "REG"
                        or team not in {source["home_team"], source["away_team"]}
                        or (source_id, team) not in team_keys
                        or (
                            family == "player"
                            and (source_id, row["player_id"], team) not in player_keys
                        )
                    ):
                        raise ValueError(f"Invalid {family} source: {row_key} {source_id}")
                    source_kickoff = kickoff_utc(source)
                    available = source_kickoff + timedelta(days=1)
                    if available >= prediction:
                        raise ValueError(f"Same-game or unavailable {family} history: {row_key}")
                    source_kickoffs.append(source_kickoff)
                    availability.append(available)
                if row[f"latest_{family}_game_used"] != max(source_kickoffs, default=None):
                    raise ValueError(f"Incorrect latest {family} source: {row_key}")
            if not availability or max(availability) != row["feature_available_at_max"]:
                raise ValueError(f"Incorrect feature availability maximum: {row_key}")
        for column in table.columns:
            if column.endswith("_share"):
                bad = table.filter(
                    pl.col(column).is_not_null()
                    & ((pl.col(column) < 0) | (pl.col(column) > 1.000001))
                ).height
                if bad:
                    raise ValueError(f"Invalid {season} share: {column}")
            if (
                column.endswith(("_routes_per_game", "_route_participation"))
                and table[column].is_not_null().any()
            ):
                raise ValueError(f"Unavailable {season} route feature populated: {column}")
        total_rows += table.height
    combined_path = data_dir / "derived" / "phase2_2017_2025_player_features.parquet"
    if hashlib.sha256(combined_path.read_bytes()).hexdigest() != artifact_hashes.get("combined"):
        raise ValueError("Changed combined historical feature artifact")
    combined = pl.read_parquet(
        combined_path,
        columns=["game", "team", "player_id"],
    )
    combined_keys = set(combined.iter_rows())
    if combined.height != total_rows or combined_keys != all_keys:
        raise ValueError("Combined historical feature table disagrees with season files")
    return {
        "rows": total_rows,
        "seasons": len(seasons),
        "games": sum(int(row["games_with_rows"]) for row in matrix.values()),
    }
