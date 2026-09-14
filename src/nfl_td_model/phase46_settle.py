"""Postgame-only Phase 4.6 settlement process; never imported by Stage A."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

from nfl_td_model.market_math import american_to_decimal
from nfl_td_model.odds import parse_time
from nfl_td_model.phase1 import normalize_name
from nfl_td_model.phase45_settle import TEAM_ALIASES, _outcome_map, _special_td_map


def verify_frozen_event(event_dir: Path) -> dict[str, Any]:
    """Hash-check all Stage A bytes before any result request is possible."""
    manifest_path = event_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "phase46-frozen-v1" or manifest["diagnostic_exhibition_slate"]:
        raise ValueError("Invalid prospective frozen manifest")
    if parse_time(manifest["kickoff"]).astimezone(ZoneInfo("America/New_York")).date().isoformat() <= "2026-09-13":
        raise ValueError("September 13 cannot enter the prospective settlement path")
    for filename, field in (
        ("pregame.json", "snapshot_sha256"),
        ("quotes.json", "quotes_sha256"),
        ("predictions.json", "predictions_sha256"),
    ):
        if hashlib.sha256((event_dir / filename).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f"Frozen Stage A file modified: {filename}")
    if parse_time(manifest["created_at_utc"]) >= datetime.now(UTC):
        raise ValueError("Frozen manifest creation timestamp is in the future")
    return manifest


def _final_game(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read a game summary only after the matching scoreboard event is final."""
    date = parse_time(snapshot["kickoff"]).strftime("%Y%m%d")
    with httpx.Client(timeout=30) as client:
        response = client.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            params={"dates": date},
        )
        response.raise_for_status()
        matches = []
        for event in response.json().get("events", []):
            competitors = event["competitions"][0]["competitors"]
            teams = {TEAM_ALIASES.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                     for c in competitors}
            if teams == {snapshot["home"], snapshot["away"]}:
                matches.append(event)
        if len(matches) != 1:
            raise RuntimeError("Matching final-game scoreboard event is unavailable or ambiguous")
        event = matches[0]
        if event["status"]["type"]["name"] != "STATUS_FINAL":
            raise RuntimeError("Stage B waits until this game is final")
        summary = client.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
            params={"event": event["id"]},
        )
        summary.raise_for_status()
        return {"event": event, "summary": summary.json()}


def _participation(summary: dict[str, Any]) -> set[tuple[str, str]]:
    """Positive participation evidence only; absence from a boxscore is unknown."""
    seen = set()
    for team in summary.get("boxscore", {}).get("players", []):
        abbreviation = team["team"]["abbreviation"]
        abbreviation = TEAM_ALIASES.get(abbreviation, abbreviation)
        for group in team.get("statistics", []):
            for athlete in group.get("athletes", []):
                seen.add((abbreviation, normalize_name(athlete["athlete"]["displayName"])))
    return seen


def _official_participation(path: Path | None, kickoff: datetime) -> dict[tuple[str, str], bool]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"players"}:
        raise ValueError("Unexpected official participation fields")
    result = {}
    for row in data["players"]:
        if set(row) != {"player", "team", "played", "source_url", "published_at"}:
            raise ValueError("Unexpected official participation row")
        if not row["source_url"].startswith(("https://www.nfl.com/", "https://nfl.com/")):
            raise ValueError("DNP evidence requires an NFL source")
        if parse_time(row["published_at"]) < kickoff:
            raise ValueError("Participation evidence predates the game")
        result[(row["team"], normalize_name(row["player"]))] = bool(row["played"])
    return result


def _book_rules(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise TypeError("Book DNP rules must be a mapping")
    parsed = {}
    for book, record in rules.items():
        if set(record) != {"rule", "source_url", "retrieved_at"}:
            raise ValueError("Book DNP rule needs a source and retrieval time")
        source = urlparse(record["source_url"])
        if source.scheme != "https" or not source.hostname:
            raise ValueError("Book DNP rule requires an HTTPS source")
        if record["rule"] not in {"void_if_dnp", "action_if_dnp"}:
            raise ValueError("Book DNP rule must say void_if_dnp or action_if_dnp")
        parse_time(record["retrieved_at"])
        parsed[book] = record["rule"]
    return parsed


def settle_one(
    row: dict[str, Any], actual_td: int, played: bool | None, dnp_rule: str | None,
) -> dict[str, Any]:
    """Preserve the frozen decision; grade DNP only when both facts are known."""
    settled = dict(row)
    settled["actual_td"] = actual_td
    settled["ultimately_played"] = played
    if not row["diagnostic_bet_best"]:
        status, pl = "not_bet", 0.0
    elif actual_td > 0:
        status, pl = "won", american_to_decimal(int(row["best_odds"])) - 1
    elif played is True:
        status, pl = "lost", -1.0
    elif played is None:
        status, pl = "pending_participation", None
    elif dnp_rule == "void_if_dnp":
        status, pl = "void_dnp", 0.0
    elif dnp_rule == "action_if_dnp":
        status, pl = "lost_dnp_action", -1.0
    else:
        status, pl = "pending_book_rule", None
    settled["settlement_status"] = status
    settled["pl_best_units"] = pl
    return settled


def settle_event(
    event_dir: Path, participation_file: Path | None = None, rules_file: Path | None = None,
) -> Path:
    manifest = verify_frozen_event(event_dir)
    output = event_dir / "settlement.json"
    if output.exists():
        raise FileExistsError("Settlement already exists")
    snapshot = json.loads((event_dir / "pregame.json").read_text(encoding="utf-8"))
    frozen_rows = json.loads((event_dir / "predictions.json").read_text(encoding="utf-8"))
    game = _final_game(snapshot)
    offensive, _ = _outcome_map([game])
    special = _special_td_map([game])
    scores = dict(offensive)
    for key, count in special.items():
        scores[key] = scores.get(key, 0) + count
    game_code = f"{snapshot['away']}@{snapshot['home']}"
    observed_played = _participation(game["summary"])
    official_played = _official_participation(participation_file, parse_time(snapshot["kickoff"]))
    book_rules = _book_rules(rules_file)
    rows = []
    for row in frozen_rows:
        name = normalize_name(row["player"])
        team = row["team"]
        score_candidates = [(t, count) for (g, t, p), count in scores.items()
                            if g == game_code and p == name]
        if len(score_candidates) > 1:
            raise ValueError("Ambiguous touchdown scorer team")
        td = score_candidates[0][1] if score_candidates else 0
        participation_key = (team, name)
        played = official_played.get(participation_key)
        if played is None and participation_key in observed_played:
            played = True
        if td > 0:
            played = True
        rows.append(settle_one(row, td, played, book_rules.get(row["best_sportsbook"])))
    result = {
        "schema": "phase46-settlement-v1", "settled_at_utc": datetime.now(UTC).isoformat(),
        "event_id": manifest["event_id"], "stage_a_predictions_sha256": manifest["predictions_sha256"],
        "diagnostic_exhibition_slate": False, "rows": rows,
    }
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Separate postgame Phase 4.6 Stage B")
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("--participation-file", type=Path)
    parser.add_argument("--book-rules-file", type=Path)
    args = parser.parse_args()
    print(settle_event(args.event_dir, args.participation_file, args.book_rules_file))
