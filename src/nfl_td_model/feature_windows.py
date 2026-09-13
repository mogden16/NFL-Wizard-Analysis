"""Pure, strictly lagged summaries for candidate historical player features."""

from __future__ import annotations

from datetime import datetime
from typing import Any

WINDOWS: dict[str, int | None] = {"last3": 3, "last5": 5, "last8": 8, "season": None, "ewma": None}
EWMA_HALF_LIFE_GAMES = 3

PLAYER_COUNTS = (
    "carries",
    "targets",
    "receptions",
    "touches",
    "snaps",
    "red_zone_carries",
    "inside_10_carries",
    "inside_5_carries",
    "goal_line_opportunities",
    "red_zone_targets",
    "end_zone_targets",
)
SHARE_DENOMINATORS = {
    "carry_share": "carries",
    "target_share": "targets",
    "touch_share": "touches",
    "snap_share": "snaps",
    "goal_line_opportunity_share": "goal_line_opportunities",
    "red_zone_target_share": "red_zone_targets",
    "end_zone_target_share": "end_zone_targets",
}
TEAM_CONTEXT = (
    "offensive_plays",
    "carries",
    "targets",
    "red_zone_carries",
    "red_zone_targets",
    "offensive_touchdowns",
)
OPPONENT_CONTEXT = (
    "allowed_offensive_plays",
    "allowed_carries",
    "allowed_targets",
    "allowed_red_zone_carries",
    "allowed_red_zone_targets",
    "allowed_offensive_touchdowns",
)


def eligible_history(
    records: list[dict[str, Any]], prediction_time: datetime, *, excluded_game: str
) -> list[dict[str, Any]]:
    """Enforce the availability boundary before any window or denominator is computed."""
    return sorted(
        (
            record
            for record in records
            if record["game_id"] != excluded_game
            and record["kickoff"] < prediction_time
            and record["available_at_proxy"] < prediction_time
        ),
        key=lambda record: (record["kickoff"], record["game_id"]),
    )


def _weighted_value(records: list[dict[str, Any]], key: str, ewma: bool) -> tuple[float, float]:
    """Return weighted numerator and observed weight; missing is never treated as zero."""
    total = 0.0
    observed_weight = 0.0
    for index, record in enumerate(records):
        value = record.get(key)
        if value is None:
            continue
        weight = 0.5 ** ((len(records) - 1 - index) / EWMA_HALF_LIFE_GAMES) if ewma else 1.0
        total += float(value) * weight
        observed_weight += weight
    return total, observed_weight


def _average(records: list[dict[str, Any]], key: str, ewma: bool) -> float | None:
    if any(record.get(key) is None for record in records):
        return None
    total, weight = _weighted_value(records, key, ewma)
    return total / weight if weight else None


def _share(
    records: list[dict[str, Any]], numerator: str, denominator: str, ewma: bool
) -> float | None:
    if any(record.get(numerator) is None or record.get(denominator) is None for record in records):
        return None
    player_total = 0.0
    team_total = 0.0
    for index, record in enumerate(records):
        player = record.get(numerator)
        team = record.get(denominator)
        if player is None or team is None:
            continue
        weight = 0.5 ** ((len(records) - 1 - index) / EWMA_HALF_LIFE_GAMES) if ewma else 1.0
        player_total += float(player) * weight
        team_total += float(team) * weight
    return player_total / team_total if team_total > 0 else None


def summarize_candidate(
    player_history: list[dict[str, Any]],
    team_history: list[dict[str, Any]],
    opponent_history: list[dict[str, Any]],
    prediction_time: datetime,
    *,
    game_id: str,
) -> dict[str, Any]:
    """Build windows after filtering every source family by prediction time."""
    player = eligible_history(player_history, prediction_time, excluded_game=game_id)
    team = eligible_history(team_history, prediction_time, excluded_game=game_id)
    opponent = eligible_history(opponent_history, prediction_time, excluded_game=game_id)
    result: dict[str, Any] = {
        "player_source_game_ids": ";".join(record["game_id"] for record in player),
        "team_source_game_ids": ";".join(record["game_id"] for record in team),
        "opponent_source_game_ids": ";".join(record["game_id"] for record in opponent),
        "player_history_games": len(player),
        "team_history_games": len(team),
        "opponent_history_games": len(opponent),
        "latest_player_game_used": player[-1]["kickoff"] if player else None,
        "latest_team_game_used": team[-1]["kickoff"] if team else None,
        "latest_opponent_game_used": opponent[-1]["kickoff"] if opponent else None,
        "feature_available_at_max": max(
            (record["available_at_proxy"] for record in player + team + opponent),
            default=None,
        ),
        "availability_type": "assumed_kickoff_plus_24h",
    }
    for window, size in WINDOWS.items():
        p = player[-size:] if size else player
        t = team[-size:] if size else team
        o = opponent[-size:] if size else opponent
        ewma = window == "ewma"
        result[f"{window}_player_games"] = len(p)
        result[f"{window}_team_games"] = len(t)
        result[f"{window}_opponent_games"] = len(o)
        for metric in PLAYER_COUNTS:
            result[f"{window}_{metric}_per_game"] = _average(p, metric, ewma)
        for feature, metric in SHARE_DENOMINATORS.items():
            result[f"{window}_{feature}"] = _share(p, metric, f"team_{metric}", ewma)
        for metric in TEAM_CONTEXT:
            result[f"{window}_team_{metric}_per_game"] = _average(t, metric, ewma)
        for metric in OPPONENT_CONTEXT:
            result[f"{window}_opponent_{metric}_per_game"] = _average(o, metric, ewma)
        # FTN participation was published after 2024 postseason and identifies only
        # a primary receiver's route, so full receiver route counts are unavailable.
        result[f"{window}_routes_per_game"] = None
        result[f"{window}_route_participation"] = None
    return result
